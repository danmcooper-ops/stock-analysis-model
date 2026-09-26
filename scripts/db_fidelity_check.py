# scripts/db_fidelity_check.py
"""Round-trip snapshots through the Supabase schema and compare every row.

Each snapshot row is split with the column registry and codec, loaded into
``core.results`` (plus ``core.edgar_blobs``) with ``COPY``, read back, rebuilt
with ``join_row`` and compared with the source row under the codec's fidelity
contract (``data/db/codec.rows_equivalent``). Every date runs in its own
transaction that is rolled back, so the database is left unchanged.

With ``--published`` nothing is written: rows already published for each
date (by ``scripts/db_publish.py``) are read back and compared, and
``core.runs.source_sha256`` is checked against the file. This is how a
backfill is verified (P3).

Needs a direct connection to a database with ``supabase/migrations`` applied
(local ``supabase start``/``supabase db start``, or a dev machine on the session
pooler); the cloud container cannot reach Postgres over TCP.

    python scripts/db_fidelity_check.py --dsn postgresql://postgres:postgres@127.0.0.1:54322/postgres \\
        output/results_2026-09-*.json.gz

Exit code 1 when any row differs.
"""
import argparse
import collections
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.db.publish import canonical_sha256  # noqa: E402
from data.db.codec import (blob_sha, decode, dumps, encode, join_row, rows_equivalent, split_row,  # noqa: E402
                           values_equal)
from data.db.columns import COLUMNS  # noqa: E402
from data.db.connect import connect  # noqa: E402
from data.db.schema import quote_ident  # noqa: E402
from data.snapshot_store import (load_snapshot_file, read_snapshot, snapshot_date_from_path,  # noqa: E402
                                 split_snapshot)

COLS = list(COLUMNS)
_INSERT_NAMES = ', '.join(['run_date', 'ticker_id', *map(quote_ident, COLS), 'extra', 'edgar_history_sha'])
_SELECT = ', '.join(['t.ticker', *(f'r.{quote_ident(c)}' for c in COLS), 'r.extra', 'b.value'])


def check_snapshot(con, path, max_report=5):
    """``(n_rows, mismatches, cast_failures)`` for one snapshot, rolled back."""
    run_date = snapshot_date_from_path(path)
    _, rows = load_snapshot_file(path)
    by_ticker = {}
    for r in rows:                          # last one wins, as the stores dedupe
        by_ticker[r['ticker']] = r
    split = {t: split_row(r, COLUMNS) for t, r in by_ticker.items()}
    failures = collections.Counter(k for sr in split.values() for k in sr.cast_failures)
    mismatches = []
    with con.transaction(force_rollback=True):
        con.execute('INSERT INTO core.runs (run_date) VALUES (%s)', (run_date,))
        with con.cursor() as cur:
            ids = {}
            for t in by_ticker:
                cur.execute('INSERT INTO core.tickers (ticker, first_seen, last_seen) VALUES (%s, %s, %s) '
                            'ON CONFLICT (ticker) DO UPDATE SET last_seen = EXCLUDED.last_seen RETURNING ticker_id',
                            (t, run_date, run_date))
                ids[t] = cur.fetchone()[0]
            shas = {}
            with cur.copy('COPY core.edgar_blobs (sha, value) FROM STDIN') as cp:
                seen = set()
                for t, sr in split.items():
                    if sr.blob:
                        body = dumps(sr.blob)
                        sha = shas[t] = blob_sha(body)
                        if sha not in seen:
                            seen.add(sha)
                            cp.write_row((sha, body))
            with cur.copy(f'COPY core.results ({_INSERT_NAMES}) FROM STDIN') as cp:
                for t, sr in split.items():
                    cp.write_row((run_date, ids[t], *sr.typed, dumps(sr.extra) if sr.extra else None, shas.get(t)))
        got = con.execute(
            f'SELECT {_SELECT} FROM core.results r JOIN core.tickers t USING (ticker_id) '
            'LEFT JOIN core.edgar_blobs b ON b.sha = r.edgar_history_sha WHERE r.run_date = %s',
            (run_date,)).fetchall()
    mismatches += _compare(got, by_ticker, max_report)
    return len(by_ticker), mismatches, failures


def _compare(got, by_ticker, max_report=5):
    mismatches = []
    if len(got) != len(by_ticker):
        mismatches.append(('<row count>', [f'{len(got)} in the database, {len(by_ticker)} in the file']))
    for rec in got:
        ticker = rec[0]
        if ticker not in by_ticker:
            mismatches.append((ticker, ['<not in the file>']))
            continue
        diff = rows_equivalent(by_ticker[ticker], rebuild(rec))
        if diff:
            mismatches.append((ticker, diff))
    for ticker, diff in mismatches[:max_report]:
        print(f'    {ticker}: {diff[:10]}')
    return mismatches


def fetch_published(con, run_date):
    """Raw ``(ticker, *typed, extra, blob)`` records published for *run_date*."""
    return con.execute(
        f'SELECT {_SELECT} FROM core.results r JOIN core.tickers t USING (ticker_id) '
        'LEFT JOIN core.edgar_blobs b ON b.sha = r.edgar_history_sha WHERE r.run_date = %s',
        (run_date,)).fetchall()


def rebuild(rec):
    """The row a published record stands for."""
    return join_row(rec[0], COLS, rec[1:1 + len(COLS)], rec[-2], rec[-1])


def _run_mismatches(con, run_date, data):
    meta = split_snapshot(data)[0]
    run = con.execute('SELECT status::text, source_sha256, risk_free_rate, risk_free_rate_source, meta '
                      'FROM core.runs WHERE run_date = %s', (run_date,)).fetchone()
    if run is None:
        return [('<run>', ['not published'])]
    out = []
    if run[0] != 'complete':
        out.append(('<run>', [f'status {run[0]}']))
    if run[1] != canonical_sha256(data):
        out.append(('<run>', ['source_sha256 differs from the file (rewritten since publishing?)']))
    if not values_equal(run[2], meta.get('risk_free_rate')) or run[3] != meta.get('risk_free_rate_source'):
        out.append(('<run>', ['risk_free_rate / source differ']))
    stored = decode({k: v for k, v in (run[4] or {}).items() if k != '_publish'})
    want = {k: v for k, v in meta.items() if k not in ('date', 'risk_free_rate', 'risk_free_rate_source')}
    want = decode(json.loads(json.dumps(encode(want), allow_nan=False)))   # as JSON would carry it
    if not values_equal(stored, want):
        out.append(('<run>', ['meta differs: ' + ', '.join(sorted(
            k for k in set(stored) | set(want) if not values_equal(stored.get(k), want.get(k))))[:200]]))
    return out


def check_published(con, path):
    """``(n_rows, mismatches, {})`` comparing a published date with its file."""
    run_date = snapshot_date_from_path(path)
    data = read_snapshot(path)
    by_ticker = {r['ticker']: r for r in split_snapshot(data)[1] if r.get('ticker')}
    mismatches = _run_mismatches(con, run_date, data)
    if mismatches and mismatches[0][1] == ['not published']:
        return len(by_ticker), mismatches, {}
    return len(by_ticker), mismatches + _compare(fetch_published(con, run_date), by_ticker), {}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--dsn', default=os.environ.get('TEST_DATABASE_URL'), required='TEST_DATABASE_URL' not in os.environ)
    ap.add_argument('--published', action='store_true',
                    help='compare dates already published instead of round-tripping (read-only)')
    ap.add_argument('files', nargs='+')
    a = ap.parse_args(argv)
    check = check_published if a.published else check_snapshot
    total_rows, total_bad, all_failures = 0, 0, collections.Counter()
    with connect(a.dsn) as con:
        for path in sorted(a.files):
            t0 = time.time()
            n, bad, failures = check(con, path)
            total_rows += n
            total_bad += len(bad)
            all_failures.update(failures)
            print(f'{snapshot_date_from_path(path)}: {n} rows, {len(bad)} mismatched, '
                  f'{sum(failures.values())} cast failures -> extra ({time.time() - t0:.1f}s)')
    print(f'\n{total_rows:,} rows checked, {total_bad} mismatched')
    for k, n in all_failures.most_common(20):
        print(f'  cast failure: {k} x{n}')
    return 1 if total_bad else 0


if __name__ == '__main__':
    sys.exit(main())
