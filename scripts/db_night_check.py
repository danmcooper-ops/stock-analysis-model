#!/usr/bin/env python3
"""The nightly database check and the cutover readiness gate (P6).

``record`` runs as ``run.sh`` step 07e, after the publish (06a):

1. **DB check.** The run is complete, and its row count and source SHA-256
   match the snapshot (``check_snapshot_store.check_database``).
2. **Rating-history parity.** The database's change points equal the JSON
   cache the report keeps (``output/rating_history.json``), as of the cache's
   last scanned day. See :func:`rating_history_parity`.
3. **Record.** The night's verdict goes to ``core.night_checks`` through
   ``pipeline.record_night_check``. It is green when 06a exited 0 and both
   checks passed.

``status`` counts the streak: consecutive trading days, newest first, each
with a green record. A trading day with no record (the database unreachable,
the run dead) breaks the streak, as a red one does. When it reaches
``--need`` (20, the plan's bar), the database is ready to become primary.
The flip is manual: set ``DB_PRIMARY=1`` for the cloud routine
(``scheduled-tasks/RECOVERY.md``).

Usage:
    python scripts/db_night_check.py record --date 2026-10-01 --publish-rc 0
    python scripts/db_night_check.py status
    python scripts/db_night_check.py status --until 2026-10-30 --need 20

Transport: the Data API (``SUPABASE_URL`` + ``SUPABASE_SERVICE_ROLE_KEY``),
or ``SUPABASE_DB_URL`` for a direct connection.

Exit code: 0 when the night is green (``record``) or the gate is met
(``status``), 1 when not, 2 when the database cannot be reached or the inputs
cannot be read.
"""
import argparse
import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NEED = 20
MAX_CACHE_LAG = 5         # trading days the rating-history cache may trail the run
_BEFORE = 'before'


# --- rating-history parity ---------------------------------------------------

def _by_ticker(rows):
    """``[[ticker, date, rating], ...]`` → ``{ticker: [[date, rating], ...]}``, by date."""
    out = {}
    for tk, d, rating in rows:
        out.setdefault(tk, []).append([str(d)[:10], rating])
    for seq in out.values():
        seq.sort()
    return out


def _state(seq, as_of, start):
    """``(rating, since)`` of the last change point on or before *as_of*;
    a *since* on or before *start* is reported as ``'before'``, because the
    two sources may begin at different days."""
    last = None
    for d, rating in seq or ():
        if d > as_of:
            break
        last = (rating, d)
    if last is None:
        return None
    return last[0], (_BEFORE if last[1] <= start else last[1])


def rating_history_parity(db_hist, cache_hist, as_of):
    """Compare two ``{ticker: [[date, rating], ...]}`` change-point histories
    as the report consumes them, the "BUY since ..." line: each ticker's
    rating and since-date as of *as_of*.

    The sources may begin on different days (the database holds every
    published run; the cache starts wherever it was first built, and never
    folds in a late-backfilled older day). So both are compared from the
    later of their first days, and a since-date on or before it counts as
    "before". A ticker known to only one source, and only from before that
    day, is left out and counted: that is what the cache deliberately
    ignores.

    Returns ``(mismatches, stats)``: mismatches are
    ``(ticker, db_state, cache_state)``.
    """
    starts = [min(seq[0][0] for seq in h.values() if seq) for h in (db_hist, cache_hist)
              if any(h.values())]
    if not starts:
        return [], {'tickers': 0, 'common_start': None, 'only_before_start': 0}
    start = max(starts)                       # the later of the two first days
    mismatches, only_before = [], 0
    tickers = set(db_hist) | set(cache_hist)
    for tk in sorted(tickers):
        a = _state(db_hist.get(tk), as_of, start)
        b = _state(cache_hist.get(tk), as_of, start)
        if a == b:
            continue
        if (a is None and b and b[1] == _BEFORE) or (b is None and a and a[1] == _BEFORE):
            only_before += 1
            continue
        mismatches.append((tk, a, b))
    return mismatches, {'tickers': len(tickers), 'common_start': start, 'only_before_start': only_before}


def load_cache(results_dir, name='rating_history.json'):
    """``(hist, last_scanned)`` of the report's rating-history cache."""
    with open(os.path.join(results_dir, name), encoding='utf-8') as f:
        c = json.load(f)
    if not isinstance(c, dict) or not isinstance(c.get('hist'), dict) or not c.get('last_scanned'):
        raise ValueError(f'{name} has no hist/last_scanned')
    return c['hist'], c['last_scanned']


def cache_lag(as_of, run_date):
    """Trading days after the cache's *as_of* up to *run_date* (1 is normal:
    the cache holds every day before tonight)."""
    end, start = dt.date.fromisoformat(run_date), dt.date.fromisoformat(as_of)
    n = 0
    for d in trading_days_back(end):
        if d <= start:
            break
        n += 1
    return n


def check_parity(transport, results_dir, run_date=None, max_lag=MAX_CACHE_LAG):
    """``(parity_ok, details)`` of the database against the cache.

    A cache more than *max_lag* trading days behind *run_date* is not
    evidence: comparing the database with the same old day every night
    would pass without checking anything new. It reads "not checked".
    """
    try:
        cache, as_of = load_cache(results_dir)
    except (OSError, ValueError) as e:
        return None, {'parity': f'not checked: {e}'}
    if run_date is not None:
        lag = cache_lag(as_of, run_date)
        if lag > max_lag:
            return None, {'parity': f'not checked: cache stale since {as_of} ({lag} trading days behind)',
                          'parity_as_of': as_of}
    before = (dt.date.fromisoformat(as_of) + dt.timedelta(days=1)).isoformat()
    db_rows = transport.call('rating_history', {'p_before': before}, idempotent=True) or []
    mismatches, stats = rating_history_parity(_by_ticker(db_rows), cache, as_of)
    details = {'parity_as_of': as_of, **stats, 'mismatches': len(mismatches),
               'examples': [list(m) for m in mismatches[:10]]}
    return not mismatches, details


# --- the streak ----------------------------------------------------------------

def trading_days_back(end, n_max=400):
    """Trading days on or before *end*, newest first (NYSE calendar)."""
    from scripts.market_open import market_status
    d = end
    for _ in range(n_max * 2):
        if market_status(d)[0]:
            yield d
        d -= dt.timedelta(days=1)


def is_green(rec):
    return bool(rec) and rec.get('publish_rc') == 0 and bool(rec.get('db_check_ok')) and rec.get('parity_ok') is True


def streak(records, until, need=NEED):
    """``(n, broken_by)``: consecutive green trading days ending at *until*
    (counted up to *need*), and the day and reason that ended the count."""
    by_date = {str(r['run_date'])[:10]: r for r in records}
    n = 0
    for d in trading_days_back(until):
        rec = by_date.get(d.isoformat())
        if not is_green(rec):
            why = 'no record' if rec is None else (
                f"06a exited {rec.get('publish_rc')}" if rec.get('publish_rc') != 0 else
                'db check failed' if not rec.get('db_check_ok') else
                'parity not checked' if rec.get('parity_ok') is None else 'parity mismatch')
            return n, (d.isoformat(), why)
        n += 1
        if n >= need:
            return n, None
    return n, None


def gate_line(n, need, broken_by):
    ready = n >= need
    tail = '' if ready or broken_by is None else f'; last break {broken_by[0]}: {broken_by[1]}'
    return (f"DB_CUTOVER_STREAK {n}/{need} "
            f"({'ready: set DB_PRIMARY=1 to make 06a blocking' if ready else 'not ready'}{tail})")


# --- CLI -------------------------------------------------------------------------

def _transport():
    from data.db.publish import transport_from_env
    return transport_from_env(('SUPABASE_DB_URL',))


def cmd_record(a, transport):
    from scripts.check_snapshot_store import check_database
    try:
        problems, info = check_database(a.results_dir, a.date, transport)
    except (OSError, ValueError) as e:
        problems, info = [f'could not read the {a.date} snapshot: {e}'], []
    parity_ok, details = check_parity(transport, a.results_dir, a.date)
    details.update({'db_problems': problems[:10], 'db_info': info[:5]})
    rec = transport.call('record_night_check', {
        'p_run_date': a.date, 'p_publish_rc': a.publish_rc, 'p_db_check_ok': not problems,
        'p_parity_ok': parity_ok, 'p_details': details})
    for p in problems:
        print(f'PROBLEM: {p}')
    for line in info:
        print(f'note: {line}')
    if parity_ok is None:
        print(f"PROBLEM: rating-history parity {details['parity']}")
    else:
        print(f"{'OK' if parity_ok else 'PROBLEM'}: rating-history parity as of {details['parity_as_of']}: "
              f"{details['tickers']} tickers from {details['common_start']}, {details['mismatches']} mismatch(es)"
              + (f", e.g. {details['examples'][:3]}" if details['mismatches'] else ''))
    print(f"night {a.date}: {'green' if rec and rec.get('green') else 'not green'} (06a rc={a.publish_rc})")
    return 0 if rec and rec.get('green') else 1


def cmd_status(a, transport):
    until = dt.date.fromisoformat(a.until) if a.until else dt.date.today()
    since = (until - dt.timedelta(days=int(a.need * 1.6) + 14)).isoformat()
    records = transport.call('night_checks', {'p_since': since}, idempotent=True) or []
    n, broken = streak(records, until, a.need)
    line = gate_line(n, a.need, broken)
    print(line)
    if a.status_file:
        with open(a.status_file, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    return 0 if n >= a.need else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('record', help="check tonight's publish and record the verdict")
    r.add_argument('--date', required=True, help='run date YYYY-MM-DD')
    r.add_argument('--publish-rc', type=int, required=True, help="step 06a's exit code")
    r.add_argument('--results-dir', default='output')
    s = sub.add_parser('status', help='count the green streak')
    s.add_argument('--until', help='last day to count (default today)')
    for p in (r, s):
        p.add_argument('--need', type=int, default=NEED)
        p.add_argument('--status-file', help='append the DB_CUTOVER_STREAK line here')
    a = ap.parse_args(argv)
    from data.db.publish import PublishError
    try:
        transport, closer, _ = _transport()
    except PublishError as e:
        print(f'db_night_check: {e}', file=sys.stderr)
        return 2
    try:
        if a.cmd == 'record':
            rc = cmd_record(a, transport)
            a.until = a.date
            cmd_status(a, transport)
            return rc
        return cmd_status(a, transport)
    except PublishError as e:
        print(f'db_night_check: {e}', file=sys.stderr)
        return 2
    finally:
        if closer is not None:
            closer.close()


if __name__ == '__main__':
    sys.exit(main())
