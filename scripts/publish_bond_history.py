#!/usr/bin/env python3
"""Bond price history for the iOS app's Charts › Bonds, as ``app/bond_px/``.

The bond model's repo (``~/Projects/bond-analysis-model``, bond repo PR #4)
builds three history tables in its ``output/history/``:

- ``marks_monthly.parquet``: N-PORT fund marks, clean price per 100 face,
  one per bond per month end. Observed, about 60 days behind.
- ``treasury_daily.parquet``: FedInvest end-of-day prices for Treasuries.
  Observed, daily. TIPS are real prices.
- ``estimates_daily.parquet``: the bond model's own ``clean_price_est`` per
  run. MODELLED, not traded: the app labels them estimates.

This writes the bonds in the night's ``bonds.json`` (the ones the app can
pick) into one JSON file per first two CUSIP characters, ~130 files rather
than ~9,000: Cloudflare Pages refuses more than 20,000 files and the site
already holds ~15,000. A file::

    {"v": 1, "generated": "2026-11-23",
     "bonds": {"46647PDX1": {"mark": [["2024-03-31", 101.23], ...],
                             "tsy": [...], "est": [...]}}}

Dates are ISO days, prices clean per 100 face to three decimals, each series
oldest first; a bond carries only the series it has. Display only.

Best-effort like the other optional app files: missing history or a bad
table writes nothing and removes the folder, so the app shows "no history"
rather than a stale one; the exit code is 0 either way unless the
destination cannot be written. Anything in the destination that this run did
not write (stale buckets, iCloud "XX 2.json" conflict copies) is removed.

    python scripts/publish_bond_history.py output/bonds.json docs/app/bond_px
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict

VERSION = 1
DEFAULT_HISTORY_DIR = os.path.expanduser('~/Projects/bond-analysis-model/output/history')


def history_dir() -> str:
    """The bond repo's history folder (``BOND_HISTORY_DIR`` overrides)."""
    return os.environ.get('BOND_HISTORY_DIR') or DEFAULT_HISTORY_DIR


def bucket(cusip: str) -> str:
    """The shard a CUSIP lives in: its first two characters, upper case."""
    return cusip[:2].upper()


def published_cusips(bonds_json: str) -> set[str]:
    """The CUSIPs in the night's bonds.json (its ``cusip`` column)."""
    with open(bonds_json, encoding='utf-8') as f:
        data = json.load(f)
    keys = [c.get('k') for c in data.get('columns') or []]
    if 'cusip' not in keys:
        return set()
    i = keys.index('cusip')
    return {r[i] for r in data.get('rows') or [] if len(r) > i and isinstance(r[i], str) and r[i]}


def _series(df, date_col: str, value_col: str, cusips: set[str]) -> dict[str, list]:
    """``{cusip: [[iso date, price], ...]}`` oldest first, one point a day."""
    import pandas as pd
    if df is None or df.empty or date_col not in df or value_col not in df:
        return {}
    d = df.loc[df['cusip'].isin(cusips), ['cusip', date_col, value_col]].dropna()
    d = d[(d[value_col] > 0) & (d[value_col] < 1000)]
    d = d.assign(day=pd.to_datetime(d[date_col]).dt.strftime('%Y-%m-%d'))
    d = d.drop_duplicates(['cusip', 'day'], keep='last').sort_values(['cusip', 'day'])
    out: dict[str, list] = defaultdict(list)
    for c, day, v in zip(d['cusip'], d['day'], d[value_col], strict=True):
        out[c].append([day, round(float(v), 3)])
    return out


def build(src: str, cusips: set[str]) -> dict[str, dict]:
    """``{cusip: {"mark": [...], "tsy": [...], "est": [...]}}`` from *src*."""
    import pandas as pd

    def read(name):
        path = os.path.join(src, name)
        return pd.read_parquet(path) if os.path.exists(path) else None

    marks = read('marks_monthly.parquet')
    mark_date = 'month' if marks is not None and 'month' in marks else 'report_date'
    series = {
        'mark': _series(marks, mark_date, 'clean_price_marked', cusips),
        'tsy': _series(read('treasury_daily.parquet'), 'price_date', 'end_of_day', cusips),
        'est': _series(read('estimates_daily.parquet'), 'run_date', 'clean_price_est', cusips),
    }
    bonds: dict[str, dict] = defaultdict(dict)
    for name, by_cusip in series.items():
        for c, pts in by_cusip.items():
            bonds[c][name] = pts
    return bonds


def write(bonds: dict[str, dict], dest: str, generated: str) -> int:
    """One file per bucket into *dest*; removes anything else there.
    Returns the number of files written."""
    by_bucket: dict[str, dict] = defaultdict(dict)
    for c, s in bonds.items():
        by_bucket[bucket(c)][c] = s
    os.makedirs(dest, exist_ok=True)
    written = set()
    for b, members in sorted(by_bucket.items()):
        name = f'{b}.json'
        tmp = os.path.join(dest, name + '.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump({'v': VERSION, 'generated': generated, 'bonds': members}, f, separators=(',', ':'))
        os.replace(tmp, os.path.join(dest, name))
        written.add(name)
    for name in os.listdir(dest):
        if name not in written:
            path = os.path.join(dest, name)
            shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
    return len(written)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('bonds_json', help="the night's bonds.json")
    ap.add_argument('dest', help='folder to write (e.g. docs/app/bond_px)')
    ap.add_argument('--history-dir', default=None, help='bond repo output/history (default: BOND_HISTORY_DIR or ~/Projects/bond-analysis-model/output/history)')
    args = ap.parse_args(argv)
    src = args.history_dir or history_dir()

    def drop(reason):
        print(f'[bond_px] {reason}; no bond history published')
        if os.path.isdir(args.dest):
            shutil.rmtree(args.dest)
        return 0

    if not os.path.exists(args.bonds_json):
        return drop(f'{args.bonds_json} missing')
    if not os.path.isdir(src):
        return drop(f'{src} missing')
    try:
        cusips = published_cusips(args.bonds_json)
        with open(args.bonds_json, encoding='utf-8') as f:
            generated = json.load(f).get('run_date') or ''
        bonds = build(src, cusips)
    except Exception as e:  # best-effort: a bad table must not stop the publish
        return drop(f'history unreadable ({type(e).__name__}: {e})')
    if not bonds:
        return drop('no history for the published bonds')
    n = write(bonds, args.dest, generated)
    points = sum(len(p) for s in bonds.values() for p in s.values())
    print(f'[bond_px] {len(bonds):,} bonds, {points:,} points in {n} files -> {args.dest}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
