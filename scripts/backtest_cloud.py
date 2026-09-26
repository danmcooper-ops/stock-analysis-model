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
  compare        Week-over-week regression check of two backtest summaries.
                 Exit 1 with REGRESSION lines when the corpus shrank, a new
                 snapshot was skipped, a (date, horizon) went unmeasured or
                 fell below the return-coverage floor.

Usage:
    python scripts/backtest_cloud.py stage --repo .cloud-backtest/snapshots-data --dest output
    python scripts/backtest_cloud.py tickers --results-dir output
    python scripts/backtest_cloud.py check-prices --tickers-file tickers.txt --prices-dir output/prices
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
    returns = _tree(repo, path=RETURNS_DIR)
    summaries = sorted(nm for nm in top if SUMMARY_RE.fullmatch(nm))
    prior = summaries[-1:]

    wanted = {nm: top[nm] for nm in picked}
    wanted.update(returns)
    wanted.update({nm: top[nm] for nm in prior})
    blobs = fetch_objects(repo, wanted.values(), remote=remote)
    for name, oid in wanted.items():
        _write_atomic(os.path.join(dest, name), blobs[oid])
    log(f'[stage] {len(picked)} snapshot(s) {picked[0][8:18]} .. {picked[-1][8:18]}, '
        f'{len(returns)} return sidecar(s), prior summary: '
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


def matured_tickers(results_dir, horizon=30, today=None):
    """Every ticker of every matured snapshot, plus the benchmarks.

    Reads the ticker column from the snapshot store when it holds the date
    (a few ms), and parses the file otherwise (seconds, blobs resolved).
    """
    tickers = set(BENCHMARKS)
    store = SnapshotStore.for_results_dir(results_dir)
    try:
        for d, path in matured_snapshots(results_dir, horizon, today):
            rows = None
            if store is not None and store.has_date(d):
                rows = store.rows(d, columns=['ticker'])
            if rows is None:
                _, rows = split_snapshot(read_snapshot(path))
            tickers.update(r['ticker'] for r in rows
                           if isinstance(r, dict) and r.get('ticker'))
    finally:
        if store is not None:
            store.close()
    return sorted(tickers)


def _last_bar(path):
    import pandas as pd
    idx = pd.to_datetime(pd.read_parquet(path, columns=[]).index)
    return idx.max().date() if len(idx) else None


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
    problems = compare_summaries(cur, prior)
    for line in problems:
        print(f'REGRESSION: {line}')
    if not problems:
        print('no regressions')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
