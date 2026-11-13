#!/usr/bin/env python3
"""Each stock's composite score over time, re-scored with today's model.

The report and the iOS app show the composite score, not the rating. Its
trend and "vs the prior run" change cannot come from the stored scores: the
scoring model changed six times between 2026-07-04 and 2026-10-23 (weights,
gate masking, the Monte Carlo penalty), so stored scores from different
dates are not on one scale. Instead every sampled snapshot is re-scored with
``default_params()`` — exactly what the weekly backtest's
``rescored_current`` view does (``backtest._evaluate_params_on_snapshots``):
the stored fields are kept and only the scoring layer is re-applied.

Display only. Nothing here writes a snapshot, the store or a rating, and the
backtest and scoring code are imported, never changed.

The cache, ``<results_dir>/score_history.json``::

    {"version": 1, "params_hash": "...", "built": "<run date>",
     "scores": {"<date>": {"<TICKER>": 64.2, ...}, ...}}

holds one snapshot per ISO week over the last ``WEEKS`` weeks (never before
``FLOOR``: older snapshots predate the gate set the current model scores),
plus the ``RECENT`` newest dates, so the prior run is always there for
``score_prev``. A night whose model hash matches the cache re-scores only the
dates it lacks (usually just tonight's); a new hash rebuilds the whole cache.

    python scripts/score_history.py --results-dir output --date 2026-11-12 \
        --archive-dir .claude/worktrees/snapshots-data
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import logging
import time
from datetime import date, timedelta

logger = logging.getLogger(__name__)

CACHE_NAME = 'score_history.json'
CACHE_VERSION = 1
WEEKS = 26
# 2026-07-04 (ddef53b74): the gate rebalance that set today's gate list and
# moved the thresholds from 60/43/29 to 57/39/25. Earlier snapshots lack the
# swapped-in gates' inputs, so a re-score of them is not today's model.
FLOOR = '2026-07-04'
RECENT = 7


def _iso(d):
    return d.isoformat() if hasattr(d, 'isoformat') else str(d)[:10]


def wanted_dates(all_dates, run_date, weeks=WEEKS, floor=FLOOR, recent=RECENT):
    """The snapshot dates the cache should hold for *run_date*.

    The last available date of each ISO week in the window (one sample per
    week), plus the *recent* newest dates on or before *run_date*.
    """
    run = _iso(run_date)
    start = max(floor, (date.fromisoformat(run) - timedelta(weeks=weeks)).isoformat())
    avail = sorted(d for d in set(all_dates) if start <= d <= run)
    weekly = {}
    for d in avail:
        y, w, _ = date.fromisoformat(d).isocalendar()
        weekly[(y, w)] = d          # ascending, so the week's last date wins
    return sorted(set(weekly.values()) | set(avail[-recent:]))


def load_cache(path):
    """The cache dict, or an empty one when absent, unreadable or another version."""
    try:
        with open(path, encoding='utf-8') as fh:
            c = json.load(fh)
        if isinstance(c, dict) and c.get('version') == CACHE_VERSION \
                and isinstance(c.get('scores'), dict):
            return c
        logger.warning("score history: %s is another version; rebuilding", path)
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning("score history: %s unreadable (%s); rebuilding", path, e)
    return {'version': CACHE_VERSION, 'params_hash': None, 'built': None, 'scores': {}}


def _store(results_dir):
    try:
        from data.snapshot_store import SnapshotStore
        return SnapshotStore.for_results_dir(results_dir, allow_db=False)
    except Exception as e:
        logger.warning("score history: snapshot store unavailable (%s)", e)
        return None


def available_dates(results_dir, archive_dir=None):
    """Every snapshot date the results dir, its store or the archive holds."""
    from data.snapshot_store import list_snapshot_files
    ds = {d for d, _ in list_snapshot_files(results_dir)}
    st = _store(results_dir)
    if st is not None:
        try:
            ds.update(_iso(d) for d in st.dates())
        finally:
            st.close()
    if archive_dir and os.path.isdir(archive_dir):
        ds.update(d for d, _ in list_snapshot_files(archive_dir))
    return ds


def _rows_for(results_dir, d, archive_dir=None):
    """Snapshot *d*'s rows: the results dir's file (via the backtest's loader,
    which prefers the Parquet export and the store), else the store, else the
    archive's file."""
    from data.snapshot_store import list_snapshot_files
    from scripts.backtest import load_corpus
    if d in dict(list_snapshot_files(results_dir)):
        snaps = load_corpus(results_dir, dates=[d])
        return (snaps[0].get('results') if snaps else None) or []
    st = _store(results_dir)
    if st is not None:
        try:
            if st.has_date(d):
                return st.rows(d)
        finally:
            st.close()
    if archive_dir and d in dict(list_snapshot_files(archive_dir)):
        snaps = load_corpus(archive_dir, dates=[d], use_store=False)
        return (snaps[0].get('results') if snaps else None) or []
    return []


def rescore_date(results_dir, d, params, archive_dir=None):
    """``{ticker: composite}`` for snapshot *d*, re-scored with *params*."""
    from scripts.scoring import compute_continuous_scores
    rows = [r for r in _rows_for(results_dir, d, archive_dir) if isinstance(r, dict)]
    if not rows:
        return None
    compute_continuous_scores(rows, params=params)
    out = {}
    for r in rows:
        tk, s = r.get('ticker'), r.get('_composite_score')
        if tk and isinstance(s, (int, float)) and s == s:   # drop NaN
            out[str(tk)] = round(float(s), 1)
    return out


def update(results_dir, run_date, cache_path=None, params_hash=None, params=None,
           archive_dir=None):
    """Bring the cache up to *run_date*; returns ``(cache, n_rescored)``.

    Re-scores only the wanted dates the cache lacks, or everything when the
    model hash changed. A date that fails to load is skipped and logged; the
    next night tries it again.
    """
    from scripts.param_set import default_params, scoring_params_hash
    cache_path = cache_path or os.path.join(results_dir, CACHE_NAME)
    params = params if params is not None else default_params()
    params_hash = params_hash or scoring_params_hash()
    cache = load_cache(cache_path)
    if cache.get('params_hash') != params_hash:
        if cache.get('params_hash'):
            logger.info("score history: model %s -> %s, rebuilding",
                        cache.get('params_hash'), params_hash)
        cache = {'version': CACHE_VERSION, 'params_hash': params_hash,
                 'built': None, 'scores': {}}
    want = wanted_dates(available_dates(results_dir, archive_dir), run_date)
    scores = {d: v for d, v in cache['scores'].items() if d in want}
    n = 0
    for d in want:
        if d in scores:
            continue
        try:
            got = rescore_date(results_dir, d, params, archive_dir)
        except Exception as e:
            logger.warning("score history: re-scoring %s failed (%s); skipped", d, e)
            continue
        if got:
            scores[d] = got
            n += 1
    cache['scores'] = dict(sorted(scores.items()))
    cache['built'] = _iso(run_date)
    tmp = cache_path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(cache, fh, separators=(',', ':'))
    os.replace(tmp, cache_path)
    return cache, n


def series(cache, run_date, current_hash=None):
    """What the render ships, from a loaded cache.

    Returns a dict:
      ``dates``: the weekly sample dates strictly before *run_date* (the
        render appends tonight's own score as the last point);
      ``hist``: ``{ticker: [score or None per date]}``, whole numbers;
      ``prev_date`` / ``prev``: the newest cached date before *run_date* and
        each ticker's re-scored composite then (one decimal);
      ``stale``: True when the cache was built by another model than
        *current_hash* (the render then falls back to stored scores).
    """
    run = _iso(run_date)
    scores = (cache or {}).get('scores') or {}
    before = sorted(d for d in scores if d < run)
    stale = bool(current_hash) and (cache or {}).get('params_hash') != current_hash
    out = {'dates': [], 'hist': {}, 'prev_date': None, 'prev': {}, 'stale': stale}
    if not before:
        return out
    weekly = {}
    for d in before:
        y, w, _ = date.fromisoformat(d).isocalendar()
        weekly[(y, w)] = d
    dates = sorted(weekly.values())
    tickers = set()
    for d in dates:
        tickers.update(scores[d])
    hist = {}
    for tk in tickers:
        vals = [scores[d].get(tk) for d in dates]
        hist[tk] = [None if v is None else int(round(v)) for v in vals]
    out['dates'] = dates
    out['hist'] = hist
    out['prev_date'] = before[-1]
    out['prev'] = dict(scores[before[-1]])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--results-dir', default='output')
    ap.add_argument('--date', required=True, help='run date, YYYY-MM-DD')
    ap.add_argument('--archive-dir', default=None,
                    help='the data/snapshots worktree, for dates neither the '
                         'results dir nor its store holds')
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    t0 = time.monotonic()
    cache, n = update(a.results_dir, a.date, archive_dir=a.archive_dir)
    print(f"score history: {len(cache['scores'])} dates held, {n} re-scored, "
          f"model {cache['params_hash']}, {time.monotonic() - t0:.0f}s")
    return 0


if __name__ == '__main__':
    sys.exit(main())
