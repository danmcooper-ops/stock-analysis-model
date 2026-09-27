#!/usr/bin/env python3
"""scripts/backtest_cloud.py

Helpers for the weekly backtest's cloud Routine
(scheduled-tasks/cloud-weekly-backtest/run.sh). The container starts empty
every week, so everything the backtest reads has to be staged from the
``data/snapshots`` branch, and the prices it measures against are downloaded
cold. These subcommands keep that logic out of shell heredocs:

  stage          Materialize the backtest corpus out of a blob-less, checkout-
                 less clone of data/snapshots: every snapshot dated on/after
                 --since (plus the edgar_history blobs they reference), the
                 persisted forward-return sidecars under returns/, and the
                 newest earlier backtest summary for the week-over-week check.
                 All in a few batched fetches, never one round trip per file.
  tickers        Print the tickers of every snapshot matured at --horizon
                 days, plus the benchmarks: the price-download list.
  check-prices   Fail (exit 1) when the downloaded prices cannot support a
                 measurement: the benchmark's parquet must be current, and
                 at least --min-share of the tickers must have a parquet.
  backfill-prices
                 Fill what Yahoo lacks from Tiingo: tickers with no price
                 file, a history that starts after their first snapshot
                 (Yahoo keeps only a stub of a delisted symbol), or one that
                 ends early. Writes the parquet from Tiingo's series, caches
                 every fetched series under price_backfill/ (persisted on
                 data/snapshots, so each is fetched once), and writes
                 <prices>/_backfill.json: the confirmed delistings that
                 backtest.py measures to their last close.
  compare        Week-over-week regression check of two backtest summaries.
                 Exit 1 with REGRESSION lines when the corpus shrank, a new
                 snapshot was skipped, a (date, horizon) went unmeasured or
                 fell below the return-coverage floor. NOTICE lines (no
                 failure) say when the scoring model changed or the
                 as-recorded headline pools more than one model.

Usage:
    python scripts/backtest_cloud.py stage --repo .cloud-backtest/snapshots-data --dest output
    python scripts/backtest_cloud.py tickers --results-dir output
    python scripts/backtest_cloud.py check-prices --tickers-file tickers.txt --prices-dir output/prices
    python scripts/backtest_cloud.py backfill-prices --results-dir output --prices-dir output/prices
    python scripts/backtest_cloud.py compare output/backtest_summary_2026-09-27.json
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.snapshot_store import (SnapshotStore, list_snapshot_files,  # noqa: E402
                                 read_snapshot, split_snapshot)
from scripts.stage_snapshot_blobs import _cat_blobs, _git, stage_blobs  # noqa: E402

SNAPSHOT_RE = re.compile(r'results_(\d{4}-\d{2}-\d{2})\.json(\.gz)?')
SUMMARY_RE = re.compile(r'backtest_summary_(\d{4}-\d{2}-\d{2})\.json')
RETURNS_DIR = 'returns'
BACKFILL_DIR = 'price_backfill'
PERSISTED_DIRS = (RETURNS_DIR, BACKFILL_DIR)
BENCHMARKS = ('SPY', 'QQQ', 'IWM', 'DIA')
FETCH_BATCH = 500


# ---------------------------------------------------------------------------
# stage
# ---------------------------------------------------------------------------

def _tree(repo, rev='HEAD', path=None):
    """``{path: oid}`` for the blobs at the top level (or under *path*)."""
    args = ['ls-tree', rev] + (['-r', '--', path + '/'] if path else [])
    out = _git(repo, *args, capture_output=True, text=True).stdout
    tree = {}
    for line in out.splitlines():
        meta, _, name = line.partition('\t')
        parts = meta.split()
        if len(parts) == 3 and parts[1] == 'blob':
            tree[name] = parts[2]
    return tree


def pick_snapshots(names, since=None, newest=None, today=None, matured_days=None):
    """The archive names to stage, oldest first, one per date.

    A date held in both forms resolves to the plain ``.json``, as
    ``list_snapshot_files`` does. *since* drops earlier dates; *matured_days*
    keeps only dates at least that old; *newest* then keeps the last N.
    """
    by_date = {}
    for nm in sorted(names):
        m = SNAPSHOT_RE.fullmatch(nm)
        if not m:
            continue
        d = m.group(1)
        if d not in by_date or not nm.endswith('.gz'):
            by_date[d] = nm
    dates = sorted(by_date)
    if since:
        dates = [d for d in dates if d >= since]
    if matured_days is not None:
        cutoff = ((today or date.today()) - timedelta(days=matured_days)).isoformat()
        dates = [d for d in dates if d <= cutoff]
    if newest:
        dates = dates[-newest:]
    return [by_date[d] for d in dates]


def fetch_objects(repo, oids, remote='origin'):
    """``{oid: bytes}`` for *oids*, fetched in batches out of a partial clone."""
    oids = sorted(set(oids))
    for i in range(0, len(oids), FETCH_BATCH):
        _git(repo, '-c', 'fetch.negotiationAlgorithm=noop', 'fetch', '-q',
             '--no-tags', '--no-write-fetch-head', '--recurse-submodules=no',
             '--filter=blob:none', remote, *oids[i:i + FETCH_BATCH])
    return _cat_blobs(repo, oids) if oids else {}


def _write_atomic(path, data):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = f'{path}.stage.{os.getpid()}'
    with open(tmp, 'wb') as fh:
        fh.write(data)
    os.replace(tmp, path)


def stage(repo, dest, since=None, newest=None, matured_days=None, remote='origin',
          today=None, log=print):
    """Stage snapshots (+ blobs), return sidecars and the prior summary."""
    top = _tree(repo)
    picked = pick_snapshots(top, since=since, newest=newest, today=today,
                            matured_days=matured_days)
    if not picked:
        raise OSError(f'no snapshots on the archive branch match since={since} '
                      f'newest={newest} matured_days={matured_days}')
    persisted = {d: _tree(repo, path=d) for d in PERSISTED_DIRS}
    returns = persisted[RETURNS_DIR]
    summaries = sorted(nm for nm in top if SUMMARY_RE.fullmatch(nm))
    prior = summaries[-1:]

    wanted = {nm: top[nm] for nm in picked}
    for tree in persisted.values():
        wanted.update(tree)
    wanted.update({nm: top[nm] for nm in prior})
    blobs = fetch_objects(repo, wanted.values(), remote=remote)
    for name, oid in wanted.items():
        _write_atomic(os.path.join(dest, name), blobs[oid])
    log(f'[stage] {len(picked)} snapshot(s) {picked[0][8:18]} .. {picked[-1][8:18]}, '
        f'{len(returns)} return sidecar(s), '
        f'{len(persisted[BACKFILL_DIR])} backfilled price series, prior summary: '
        f'{prior[0] if prior else "none"}')
    stage_blobs(repo, dest, [os.path.join(dest, nm) for nm in picked],
                remote=remote, log=log)
    return picked


# ---------------------------------------------------------------------------
# tickers / check-prices
# ---------------------------------------------------------------------------

def matured_snapshots(results_dir, horizon=30, today=None):
    """``[(date_str, path)]`` of the snapshots whose *horizon* has elapsed."""
    cutoff = (today or date.today()) - timedelta(days=horizon)
    return [(d, p) for d, p in list_snapshot_files(results_dir)
            if date.fromisoformat(d) <= cutoff]


def matured_ticker_dates(results_dir, horizon=30, today=None):
    """``{ticker: [date_str, ...]}`` over every matured snapshot, oldest first.

    Reads the ticker column from the local snapshot store when it holds the
    date (a few ms), and parses the file otherwise (seconds, blobs resolved).
    Never the database: the corpus is what was staged here.
    """
    dates = {}
    store = SnapshotStore.for_results_dir(results_dir, allow_db=False)
    try:
        for d, path in matured_snapshots(results_dir, horizon, today):
            rows = None
            if store is not None and store.has_date(d):
                rows = store.rows(d, columns=['ticker'])
            if rows is None:
                _, rows = split_snapshot(read_snapshot(path))
            for r in rows:
                if isinstance(r, dict) and r.get('ticker'):
                    dates.setdefault(r['ticker'], []).append(d)
    finally:
        if store is not None:
            store.close()
    return dates


def matured_tickers(results_dir, horizon=30, today=None):
    """Every ticker of every matured snapshot, plus the benchmarks."""
    return sorted(set(BENCHMARKS) | set(matured_ticker_dates(results_dir, horizon, today)))


def _bar_span(path):
    """``(first, last)`` bar dates of a price parquet, or None."""
    import pandas as pd
    idx = pd.to_datetime(pd.read_parquet(path, columns=[]).index)
    return (idx.min().date(), idx.max().date()) if len(idx) else None


def _last_bar(path):
    span = _bar_span(path)
    return span[1] if span else None


def check_prices(tickers, prices_dir, today=None, min_share=0.90,
                 benchmark='SPY', max_benchmark_age_days=5):
    """``(ok, lines)``: can these prices support this week's measurement?

    The benchmark must be current: every excess return subtracts it, and a
    stale SPY leaves the newest matured pairs unmeasured. And at least
    *min_share* of the tickers must have a parquet at all — below that, a
    throttled download night, not the model, would decide the sample.
    """
    today = today or date.today()
    lines, ok = [], True
    spy = os.path.join(prices_dir, f'{benchmark}.parquet')
    last = _last_bar(spy) if os.path.exists(spy) else None
    if last is None or (today - last).days > max_benchmark_age_days:
        ok = False
        lines.append(f'PROBLEM: {benchmark} prices end {last or "nowhere"}, more '
                     f'than {max_benchmark_age_days} days before {today}')
    else:
        lines.append(f'{benchmark} prices end {last}')
    have = [t for t in tickers if os.path.exists(os.path.join(prices_dir, f'{t}.parquet'))]
    share = len(have) / len(tickers) if tickers else 0.0
    line = (f'{len(have)} of {len(tickers)} ticker(s) have a price file '
            f'({share:.1%}, floor {min_share:.0%})')
    if share < min_share:
        ok = False
        line = 'PROBLEM: ' + line
    lines.append(line)
    return ok, lines


# ---------------------------------------------------------------------------
# backfill-prices
# ---------------------------------------------------------------------------

GAP_DAYS = 7                 # = backtest.MAX_SNAP_GAP_DAYS
CACHE_FRESH_DAYS = 7         # a still-trading series is refetched after this
UNKNOWN_RETRY_DAYS = 28      # a ticker Tiingo does not know is retried after this


def price_problems(ticker_dates, prices_dir, today=None, gap=GAP_DAYS):
    """Tickers whose price file cannot answer every matured snapshot.

    Returns ``{ticker: {'first': date, 'rows': n, 'reason': ...}}`` where the
    reason is 'missing' (no file), 'starts_late' (no bar within *gap* days of
    the first snapshot holding it — Yahoo's stub of a delisted symbol) or
    'ends_early' (the last bar is more than *gap* days old).
    """
    today = today or date.today()
    out = {}
    for t, ds in ticker_dates.items():
        if t in BENCHMARKS:
            continue
        first = date.fromisoformat(min(ds))
        path = os.path.join(prices_dir, f'{t}.parquet')
        span = None
        if os.path.exists(path):
            try:
                span = _bar_span(path)
            except Exception as e:           # a corrupt file is as good as none
                print(f'  [warn] unreadable price file {path}: {e}')
        if span is None:
            reason = 'missing'
        elif span[0] > first + timedelta(days=gap):
            reason = 'starts_late'
        elif span[1] < today - timedelta(days=gap):
            reason = 'ends_early'
        else:
            continue
        out[t] = {'first': first, 'rows': len(ds), 'reason': reason}
    return out


def _cache_path(cache_dir, t):
    return os.path.join(cache_dir, f'{t}.json')


def _load_cached(cache_dir, t):
    try:
        with open(_cache_path(cache_dir, t), encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _cache_usable(entry, today, gap=GAP_DAYS):
    """Whether a cached fetch still answers without a new request."""
    if not entry or 'fetched' not in entry:
        return False
    fetched = date.fromisoformat(entry['fetched'])
    if entry.get('status') == 'unknown':
        return (today - fetched).days < UNKNOWN_RETRY_DAYS
    closes = entry.get('closes') or {}
    if closes and date.fromisoformat(max(closes)) < fetched - timedelta(days=gap):
        return True                          # already delisted when fetched: final
    return (today - fetched).days < CACHE_FRESH_DAYS


def _series_from_entry(entry):
    import pandas as pd
    closes = (entry or {}).get('closes') or {}
    if not closes:
        return pd.Series(dtype=float)
    return pd.Series({pd.Timestamp(d): float(v) for d, v in closes.items()}).sort_index()


def backfill_prices(problems, prices_dir, cache_dir, client, since, today=None,
                    max_calls=40, gap=GAP_DAYS, log=print):
    """Resolve *problems* (see price_problems) from Tiingo; write the manifest.

    Fetches at most *max_calls* series (most-affected tickers first; the rest
    wait for next week, since each fetch is cached), and stops early on a rate
    limit. A Tiingo series replaces the price file when it covers more of the
    window than the file does. A series whose last bar is more than *gap*
    days before *today* is a confirmed delisting. The manifest also lists the
    tickers whose file was replaced, so a return sidecar frozen before this
    run knows to top them up.
    """
    import pandas as pd
    today = today or date.today()
    os.makedirs(cache_dir, exist_ok=True)
    calls = 0
    stats = {'problems': len(problems), 'fetched': 0, 'cached': 0,
             'unknown': 0, 'deferred': 0, 'written': 0, 'delisted': 0}
    backfilled, deferred = [], []
    order = sorted(problems.items(), key=lambda kv: (-kv[1]['rows'], kv[0]))
    for t, _info in order:
        entry = _load_cached(cache_dir, t)
        if _cache_usable(entry, today, gap):
            stats['cached'] += 1
        elif calls >= max_calls or getattr(client, 'rate_limited', False):
            deferred.append(t)
            if not entry:
                continue
        else:
            calls += 1
            series = client.fetch_closes(t, since)
            if series is None:               # failed: keep any older entry
                deferred.append(t)
                if not entry:
                    continue
            else:
                stats['fetched'] += 1
                entry = {'ticker': t, 'fetched': today.isoformat(),
                         'status': 'ok' if len(series) else 'unknown',
                         'source': 'tiingo',
                         'closes': {d.date().isoformat(): round(float(v), 6)
                                    for d, v in series.items()}}
                _write_atomic(_cache_path(cache_dir, t),
                              json.dumps(entry, sort_keys=True).encode())
        series = _series_from_entry(entry)
        if not len(series):
            stats['unknown'] += 1
            continue
        path = os.path.join(prices_dir, f'{t}.parquet')
        span = None
        if os.path.exists(path):
            try:
                span = _bar_span(path)
            except Exception:
                span = None
        s_first, s_last = series.index.min().date(), series.index.max().date()
        if span is None or s_first < span[0] or s_last > span[1]:
            df = pd.DataFrame({'Close': series.values},
                              index=pd.DatetimeIndex(series.index, name='Date'))
            df.to_parquet(path)
            backfilled.append(t)
            stats['written'] += 1
    stats['deferred'] = len(deferred)

    # The manifest is rebuilt from every cached series, not just this run's
    # problems: a delisting found in an earlier week still has to be measured.
    delisted = {}
    for fn in sorted(os.listdir(cache_dir)):
        if not fn.endswith('.json'):
            continue
        t = fn[:-5]
        path = os.path.join(prices_dir, f'{t}.parquet')
        if not os.path.exists(path):
            continue
        entry = _load_cached(cache_dir, t)
        closes = (entry or {}).get('closes') or {}
        if not closes:
            continue
        fetched = date.fromisoformat(entry['fetched'])
        last = max(closes)
        # Delisted: Tiingo's series stopped well before it was fetched, and
        # the price file (whichever source) ends there too.
        span = _bar_span(path)
        if (date.fromisoformat(last) < fetched - timedelta(days=gap)
                and span and span[1] < today - timedelta(days=gap)):
            delisted[t] = {'last_date': span[1].isoformat(),
                           'last_close': closes[last], 'source': 'tiingo'}
    stats['delisted'] = len(delisted)
    manifest = {'generated': today.isoformat(), 'delisted': delisted,
                'backfilled': sorted(backfilled), 'deferred': sorted(deferred)}
    _write_atomic(os.path.join(prices_dir, '_backfill.json'),
                  json.dumps(manifest, indent=1, sort_keys=True).encode())
    log(f'[backfill] {stats["problems"]} ticker(s) without usable prices: '
        f'{stats["fetched"]} fetched, {stats["cached"]} from cache, '
        f'{stats["deferred"]} deferred, {stats["unknown"]} unknown to Tiingo; '
        f'{stats["written"]} price file(s) written, {stats["delisted"]} confirmed delisting(s)')
    for t in sorted(delisted):
        log(f'  delisted {t}: last close {delisted[t]["last_close"]} on {delisted[t]["last_date"]}')
    return stats


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

def compare_summaries(cur, prior, min_coverage=None):
    """REGRESSION lines for *cur* against the *prior* weekly summary.

    Also flags current-only problems (unmeasured pairs, low coverage), so a
    first week with no prior summary is still checked.
    """
    out = []
    floor = min_coverage
    if floor is None:
        floor = (cur.get('provenance') or {}).get('min_return_coverage', 0.90)
    if prior:
        lost = sorted(set(prior.get('snapshots', [])) - set(cur.get('snapshots', [])))
        if lost:
            out.append(f'{len(lost)} snapshot(s) measured last week are not measured '
                       f'this week: {", ".join(lost)}')
        prior_skipped = {d for d, _ in prior.get('skipped_snapshots', [])}
        new_skips = [(d, why) for d, why in cur.get('skipped_snapshots', [])
                     if d not in prior_skipped]
        # A prior summary written before skipped_snapshots existed carries no
        # list to compare against; the first week just sets the baseline.
        if 'skipped_snapshots' in prior:
            for d, why in new_skips:
                out.append(f'newly skipped snapshot {d}: {why}')
    for u in cur.get('unmeasured', []):
        out.append(f'{u["run_date"]} +{u["horizon"]}d unmeasured: {u["reason"]}')
    for c in cur.get('coverage', []):
        cov = c.get('coverage')
        if cov is not None and cov < floor:
            out.append(f'{c["run_date"]} +{c["horizon"]}d priced {cov:.1%} of its '
                       f'tickers (floor {floor:.0%})')
    return out


def model_notices(cur, prior):
    """NOTICE lines about the scoring model — informational, never a failure.

    A weight or gate change is a reviewed decision, not a regression, but it
    changes what the as-recorded headline means: from the change on it pools
    two models. Say so, and point at the re-scored view.
    """
    out = []
    cur_hash = (cur.get('provenance') or {}).get('current_params_hash')
    prior_hash = ((prior or {}).get('provenance') or {}).get('current_params_hash')
    if cur_hash and prior_hash and cur_hash != prior_hash:
        out.append(f'scoring model changed since last week ({prior_hash} -> {cur_hash}); '
                   f'the re-scored view measures the new model over every snapshot')
    regimes = cur.get('regimes') or []
    if len(regimes) > 1:
        out.append('as-recorded headline pools ' + ', '.join(
            f'{r["params_hash"]} ({r["snapshots"]} snapshot(s) {r["first"]}..{r["last"]})'
            for r in regimes))
    return out


def _prior_summary_path(current_path):
    """The newest ``backtest_summary_*.json`` beside *current_path*, dated before it."""
    m = SUMMARY_RE.fullmatch(os.path.basename(current_path))
    cur_date = m.group(1) if m else '9999-99-99'
    cands = []
    for p in glob.glob(os.path.join(os.path.dirname(current_path) or '.',
                                    'backtest_summary_*.json')):
        mm = SUMMARY_RE.fullmatch(os.path.basename(p))
        if mm and mm.group(1) < cur_date:
            cands.append((mm.group(1), p))
    return max(cands)[1] if cands else None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('stage')
    p.add_argument('--repo', required=True, help='blob-less clone of data/snapshots')
    p.add_argument('--dest', default='output')
    p.add_argument('--since', default=None, help='stage snapshots dated >= this')
    p.add_argument('--newest', type=int, default=None, help='keep only the newest N')
    p.add_argument('--matured-days', type=int, default=None,
                   help='keep only snapshots at least this many days old')
    p.add_argument('--remote', default='origin')

    p = sub.add_parser('tickers')
    p.add_argument('--results-dir', default='output')
    p.add_argument('--horizon', type=int, default=30)

    p = sub.add_parser('check-prices')
    p.add_argument('--tickers-file', required=True,
                   help='the list `tickers` wrote (whitespace-separated)')
    p.add_argument('--prices-dir', default='output/prices')
    p.add_argument('--min-share', type=float, default=0.90)

    p = sub.add_parser('backfill-prices')
    p.add_argument('--results-dir', default='output')
    p.add_argument('--prices-dir', default='output/prices')
    p.add_argument('--cache-dir', default=None,
                   help=f'default: <results-dir>/{BACKFILL_DIR}')
    p.add_argument('--since', default='2026-07-06',
                   help='fetch closes from 10 days before this date')
    p.add_argument('--horizon', type=int, default=30)
    p.add_argument('--max-calls', type=int, default=40,
                   help='Tiingo requests per run (the rest wait a week)')

    p = sub.add_parser('compare')
    p.add_argument('current', help="this week's backtest_summary_<date>.json")
    p.add_argument('--prior', default=None,
                   help='default: the newest earlier summary beside it')

    args = ap.parse_args(argv)

    if args.cmd == 'stage':
        try:
            stage(args.repo, args.dest, since=args.since, newest=args.newest,
                  matured_days=args.matured_days, remote=args.remote)
        except (OSError, subprocess.CalledProcessError) as e:
            print(f'[stage] failed: {e}')
            return 1
        return 0

    if args.cmd == 'tickers':
        print(' '.join(matured_tickers(args.results_dir, args.horizon)))
        return 0

    if args.cmd == 'check-prices':
        with open(args.tickers_file, encoding='utf-8') as f:
            tickers = f.read().split()
        ok, lines = check_prices(tickers, args.prices_dir, min_share=args.min_share)
        print('\n'.join(lines))
        return 0 if ok else 1

    if args.cmd == 'backfill-prices':
        from data.tiingo_client import TiingoClient
        client = TiingoClient()
        if not client.available:
            print('[backfill] TIINGO_API_KEY unset — delisted names stay unmeasured')
            return 1
        since = (date.fromisoformat(args.since) - timedelta(days=10)).isoformat()
        problems = price_problems(matured_ticker_dates(args.results_dir, args.horizon),
                                  args.prices_dir)
        backfill_prices(problems, args.prices_dir,
                        args.cache_dir or os.path.join(args.results_dir, BACKFILL_DIR),
                        client, since, max_calls=args.max_calls)
        return 1 if client.rate_limited else 0

    # compare
    with open(args.current, encoding='utf-8') as f:
        cur = json.load(f)
    prior_path = args.prior or _prior_summary_path(args.current)
    prior = None
    if prior_path:
        with open(prior_path, encoding='utf-8') as f:
            prior = json.load(f)
    print(f'current: {args.current}\nprior:   {prior_path or "none"}')
    if prior:
        print(f'snapshots measured: {len(prior.get("snapshots", []))} -> '
              f'{len(cur.get("snapshots", []))}')
    for line in model_notices(cur, prior):
        print(f'NOTICE: {line}')
    problems = compare_summaries(cur, prior)
    for line in problems:
        print(f'REGRESSION: {line}')
    if not problems:
        print('no regressions')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
