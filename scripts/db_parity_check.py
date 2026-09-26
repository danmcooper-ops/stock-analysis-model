# scripts/db_parity_check.py
"""Verify a backfilled Supabase database against its snapshot files (P3).

This covers stability checks 1-2 of ``design/supabase-migration.md``, date by
date:

1. **Fidelity.** Every published row rebuilds to the file's row (codec
   contract), and the run's status, ``source_sha256``, risk-free rate and meta
   match the file (``db_fidelity_check.check_published``).
2. **Decision parity.** The rows rebuilt from the database and the file's rows
   are both re-scored with ``scoring.score_and_rate`` (today's code and
   parameters). Every ticker must get the same rating, raw rating, cap and
   composite score from both. This shows that nothing the scorers read was
   lost on the way in. Re-scoring the file alone is not the test: an old
   snapshot was scored by older code.
3. **Rating history.** ``core.rating_changes`` equals the change points
   computed from the files in date order: the first observation plus every
   transition, skipping NULL or empty ratings. That is the DuckDB store's
   ``rating_history()`` rule.

Read-only; needs a direct connection.

    python scripts/db_parity_check.py --results-dir output/archive \\
        --dsn postgresql://postgres:postgres@127.0.0.1:54322/postgres

Exit code 1 on any mismatch.
"""
import argparse
import copy
import logging
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.db.codec import values_equal  # noqa: E402
from data.db.connect import connect  # noqa: E402
from data.snapshot_store import list_snapshot_files, read_snapshot, split_snapshot  # noqa: E402
from scripts.db_fidelity_check import _compare, _run_mismatches, fetch_published, rebuild  # noqa: E402

DECISION_KEYS = ('rating', 'rating_raw', '_rating_cap', '_composite_score')


def _score(rows):
    from scripts.scoring import score_and_rate
    rows = copy.deepcopy(rows)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        score_and_rate(rows)
    return {r['ticker']: r for r in rows}


def decision_mismatches(file_rows, db_rows):
    """Tickers whose re-scored decision differs between the file and the database."""
    a, b = _score(file_rows), _score(db_rows)
    out = []
    for t in sorted(set(a) | set(b)):
        diff = [k for k in DECISION_KEYS if not values_equal((a.get(t) or {}).get(k), (b.get(t) or {}).get(k))]
        if diff:
            out.append((t, diff))
    return out


class HistoryFromFiles:
    """Rating change points accumulated from the files in date order."""

    def __init__(self):
        self.last = {}
        self.points = set()

    def add(self, run_date, rows):
        for r in rows:
            rating = r.get('rating')
            if rating is None or rating == '':
                continue
            if self.last.get(r['ticker']) != rating:
                self.points.add((r['ticker'], run_date, rating))
            self.last[r['ticker']] = rating


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--dsn', default=os.environ.get('TEST_DATABASE_URL'), required='TEST_DATABASE_URL' not in os.environ)
    ap.add_argument('--results-dir', required=True)
    ap.add_argument('--since')
    ap.add_argument('--until')
    ap.add_argument('--no-scoring', action='store_true', help='skip decision parity (the slow part)')
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger('scripts.scoring').setLevel(logging.ERROR)

    history = HistoryFromFiles()
    totals = {'dates': 0, 'rows': 0, 'fidelity': 0, 'decisions': 0}
    t_start = time.time()
    with connect(a.dsn) as con:
        for d, path in list_snapshot_files(a.results_dir):
            if (a.since and d < a.since) or (a.until and d > a.until):
                continue
            data = read_snapshot(path)
            rows = [r for r in split_snapshot(data)[1] if r.get('ticker')]
            by_ticker = {r['ticker']: r for r in rows}           # last one wins, as published
            history.add(d, list(by_ticker.values()))
            recs = fetch_published(con, d)
            fid = _run_mismatches(con, d, data)
            if not (fid and fid[0][1] == ['not published']):
                fid += _compare(recs, by_ticker, max_report=3)
            dec = []
            if not a.no_scoring and recs:
                dec = decision_mismatches(list(by_ticker.values()), [rebuild(r) for r in recs])
                for t, keys in dec[:3]:
                    print(f'    decision {t}: {keys}')
            totals['dates'] += 1
            totals['rows'] += len(by_ticker)
            totals['fidelity'] += len(fid)
            totals['decisions'] += len(dec)
            print(f'{d}: {len(by_ticker)} rows, {len(fid)} fidelity, {len(dec)} decision mismatches', flush=True)
        db_points = {(t, d.isoformat(), r) for t, d, r in con.execute(
            'SELECT t.ticker, rc.run_date, rc.rating FROM core.rating_changes rc JOIN core.tickers t USING (ticker_id) '
            'WHERE (%s::date IS NULL OR rc.run_date >= %s::date) AND (%s::date IS NULL OR rc.run_date <= %s::date)',
            (a.since, a.since, a.until, a.until)).fetchall()}
    want = {p for p in history.points if (not a.since or p[1] >= a.since) and (not a.until or p[1] <= a.until)}
    missing, extra = want - db_points, db_points - want
    if a.since:
        # before --since, the files are not read, so a first observation in the
        # window may be a continuation in the database; only extra points count
        missing = set()
    print(f"\n{totals['dates']} dates, {totals['rows']:,} rows: {totals['fidelity']} fidelity mismatches, "
          f"{totals['decisions']} decision mismatches; rating change points: {len(want):,} from files, "
          f'{len(db_points):,} in the database, {len(missing)} missing, {len(extra)} extra '
          f'({time.time() - t_start:.0f}s)')
    for p in sorted(missing)[:5]:
        print(f'  missing change point {p}')
    for p in sorted(extra)[:5]:
        print(f'  extra change point {p}')
    bad = totals['fidelity'] or totals['decisions'] or missing or extra
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
