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

    {"v": 2, "generated": "2026-11-23",
     "bonds": {"46647PDX1": {"mark": [["2024-03-31", 101.23], ["2024-04-30", 100.9, 1], ...],
                             "tsy": [["2026-11-02", 99.5], ...],
                             "est": [["2026-11-20", 101.9, 97], ...],
                             "flags": ["default"]}}}

Dates are ISO days, prices clean per 100 face to three decimals, each series
oldest first; a bond carries only the series it has. A mark's third element
``1`` means one fund marked it (23.5% of points; the app draws them
lighter); an estimate's is ``mark_age_days``, the age of the mark it was
modelled from. ``flags``, from the bond's latest mark: ``default``,
``arrears``, ``convertible``; ``real`` for a TIPS (real, unindexed prices,
not comparable with nominal ones). Display only.

Measured by the bond-history session on the backfill (2019-09..2026-07,
1.68M marks): marks are all ``nport_implied_clean`` (per 100); a mark under
5 or over 250 is dropped (7,774, mostly agency pools where value over
balance is not a price, and defaulted bonds, median 1 fund); convertibles
legitimately trade to 200+.

``meta.json`` beside the buckets carries ``validation``: Treasury fund marks
(3+ funds) against FedInvest's end-of-day price on the same date, recomputed
each night (median |difference| 0.016 points, p90 0.115, n 8,929 when
measured), so the app can say how far fund marks are from official prices.

The daily series (``tsy``, ``est``) keep every day of their last
``DAILY_DAYS`` (365) and one point a week (the week's last) before that: the
FedInvest backfill from 2019 is ~1,800 days per Treasury, and nearly every
Treasury shares the ``91`` file, which daily would approach Pages' 25 MiB
per-file limit. A file still over ``MAX_FILE_BYTES`` (20 MiB, under the
limit ``check_pages_limits.py`` enforces) is not written, with a warning,
so one bucket can never fail the deploy.

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

VERSION = 2
MARK_MIN, MARK_MAX = 5.0, 250.0
VALIDATION_MIN_FUNDS = 3
DAILY_DAYS = 365
MAX_FILE_BYTES = 20 * 2**20
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


def thin_weekly(points: list, daily_days: int = DAILY_DAYS) -> list:
    """Every point of the last *daily_days* before the series' end, and the
    last point of each ISO week before that. *points* are ``[iso day,
    price]`` oldest first."""
    from datetime import date, timedelta
    if not points:
        return points
    end = date.fromisoformat(points[-1][0])
    cut = (end - timedelta(days=daily_days)).isoformat()
    old = [p for p in points if p[0] < cut]
    recent = [p for p in points if p[0] >= cut]
    weekly: dict[tuple, list] = {}
    for p in old:
        weekly[date.fromisoformat(p[0]).isocalendar()[:2]] = p   # the week's last wins
    return list(weekly.values()) + recent


def _series(df, date_col: str, value_col: str, cusips: set[str], lo=1e-9, hi=1000.0,
            extra=None) -> dict[str, list]:
    """``{cusip: [[iso date, price(, extra)], ...]}`` oldest first, one point
    a day. *extra*, given a row's values, returns the optional third element
    (or None for none)."""
    import pandas as pd
    if df is None or df.empty or date_col not in df or value_col not in df:
        return {}
    d = df.loc[df['cusip'].isin(cusips)].dropna(subset=[date_col, value_col])
    d = d[(d[value_col] >= lo) & (d[value_col] <= hi)]
    d = d.assign(day=pd.to_datetime(d[date_col]).dt.strftime('%Y-%m-%d'))
    d = d.drop_duplicates(['cusip', 'day'], keep='last').sort_values(['cusip', 'day'])
    out: dict[str, list] = defaultdict(list)
    for row in d.itertuples(index=False):
        r = row._asdict()
        point = [r['day'], round(float(r[value_col]), 3)]
        x = extra(r) if extra else None
        if x is not None:
            point.append(x)
        out[r['cusip']].append(point)
    return out


def _single_fund(r):
    n = r.get('n_funds')
    return 1 if n is not None and n == n and int(n) <= 1 else None


def _mark_age(r):
    a = r.get('mark_age_days')
    return int(a) if a is not None and a == a else None


def _flags(marks, tsy, cusips: set[str]) -> dict[str, list]:
    """Per bond, from its latest mark: default / arrears / convertible; and
    ``real`` for a TIPS (FedInvest ``security_type``)."""
    out: dict[str, list] = defaultdict(list)
    if marks is not None and not marks.empty:
        date_col = 'month' if 'month' in marks else 'report_date'
        last = (marks[marks['cusip'].isin(cusips)].sort_values(date_col)
                .drop_duplicates('cusip', keep='last'))
        for col, flag in (('is_default', 'default'), ('in_arrears', 'arrears'), ('is_convertible', 'convertible')):
            if col in last:
                for c in last.loc[last[col].fillna(False).astype(bool), 'cusip']:
                    out[c].append(flag)
    if tsy is not None and not tsy.empty and 'security_type' in tsy:
        for c in set(tsy.loc[tsy['security_type'] == 'TIPS', 'cusip']) & cusips:
            out[c].append('real')
    return out


def validation(marks, tsy) -> dict | None:
    """Treasury fund marks (``VALIDATION_MIN_FUNDS``+ funds) against
    FedInvest's end-of-day on the same date: how far marks sit from official
    prices. None when the tables do not overlap."""
    import pandas as pd
    need = {'cusip', 'report_date', 'clean_price_marked'}
    if marks is None or tsy is None or not need <= set(marks) or 'end_of_day' not in tsy:
        return None
    m = marks[list(need | ({'n_funds'} & set(marks)))].copy()
    if 'n_funds' in m:
        m = m[m['n_funds'] >= VALIDATION_MIN_FUNDS]
    m['day'] = pd.to_datetime(m['report_date']).dt.normalize()
    t = tsy[['cusip', 'price_date', 'end_of_day']].copy()
    t['day'] = pd.to_datetime(t['price_date']).dt.normalize()
    j = m.merge(t, on=['cusip', 'day']).dropna(subset=['clean_price_marked', 'end_of_day'])
    j = j[j['end_of_day'] > 0]
    if len(j) < 30:
        return None
    diff = (j['clean_price_marked'] - j['end_of_day'])
    a = diff.abs()
    return {'n': int(len(j)), 'median_diff': round(float(diff.median()), 3),
            'median_abs': round(float(a.median()), 3), 'p90_abs': round(float(a.quantile(0.9)), 3),
            'min_funds': VALIDATION_MIN_FUNDS,
            'from': j['day'].min().strftime('%Y-%m-%d'), 'to': j['day'].max().strftime('%Y-%m-%d')}


def read_tables(src: str):
    """``(marks, treasury, estimates)`` from *src*, each None when absent."""
    import pandas as pd

    def read(name):
        path = os.path.join(src, name)
        return pd.read_parquet(path) if os.path.exists(path) else None

    return read('marks_monthly.parquet'), read('treasury_daily.parquet'), read('estimates_daily.parquet')


def build(src: str, cusips: set[str], tables=None) -> dict[str, dict]:
    """``{cusip: {"mark": [...], "tsy": [...], "est": [...], "flags": [...]}}``."""
    marks, treasury, estimates = tables or read_tables(src)
    mark_date = 'month' if marks is not None and 'month' in marks else 'report_date'
    tsy = _series(treasury, 'price_date', 'end_of_day', cusips)
    est = _series(estimates, 'run_date', 'clean_price_est', cusips, extra=_mark_age)
    series = {
        'mark': _series(marks, mark_date, 'clean_price_marked', cusips, lo=MARK_MIN, hi=MARK_MAX,
                        extra=_single_fund),
        'tsy': {c: thin_weekly(p) for c, p in tsy.items()},
        'est': {c: thin_weekly(p) for c, p in est.items()},
    }
    bonds: dict[str, dict] = defaultdict(dict)
    for name, by_cusip in series.items():
        for c, pts in by_cusip.items():
            bonds[c][name] = pts
    for c, flags in _flags(marks, treasury, cusips).items():
        if c in bonds:
            bonds[c]['flags'] = flags
    return bonds


def write(bonds: dict[str, dict], dest: str, generated: str, max_file_bytes: int = MAX_FILE_BYTES,
          meta: dict | None = None) -> int:
    """One file per bucket into *dest*, plus ``meta.json``; removes anything
    else there. Returns the number of bucket files written."""
    by_bucket: dict[str, dict] = defaultdict(dict)
    for c, s in bonds.items():
        by_bucket[bucket(c)][c] = s
    os.makedirs(dest, exist_ok=True)
    written = set()
    for b, members in sorted(by_bucket.items()):
        name = f'{b}.json'
        body = json.dumps({'v': VERSION, 'generated': generated, 'bonds': members}, separators=(',', ':'))
        if len(body.encode('utf-8')) > max_file_bytes:
            print(f'[bond_px] WARNING: {name} would be {len(body) / 2**20:.1f} MiB '
                  f'(cap {max_file_bytes / 2**20:.0f} MiB); not published')
            continue
        tmp = os.path.join(dest, name + '.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            f.write(body)
        os.replace(tmp, os.path.join(dest, name))
        written.add(name)
    n = len(written)
    tmp = os.path.join(dest, 'meta.json.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump({'v': VERSION, 'generated': generated, **(meta or {})}, f, separators=(',', ':'))
    os.replace(tmp, os.path.join(dest, 'meta.json'))
    written.add('meta.json')
    for name in os.listdir(dest):
        if name not in written:
            path = os.path.join(dest, name)
            shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
    return n


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
        tables = read_tables(src)
        bonds = build(src, cusips, tables)
        check = validation(tables[0], tables[1])
    except Exception as e:  # best-effort: a bad table must not stop the publish
        return drop(f'history unreadable ({type(e).__name__}: {e})')
    if not bonds:
        return drop('no history for the published bonds')
    n = write(bonds, args.dest, generated, meta={'validation': check} if check else None)
    if check:
        print(f"[bond_px] Treasury marks vs FedInvest: median |diff| {check['median_abs']}, "
              f"p90 {check['p90_abs']} (n {check['n']:,})")
    points = sum(len(p) for s in bonds.values() for k, p in s.items() if k != 'flags')
    print(f'[bond_px] {len(bonds):,} bonds, {points:,} points in {n} files -> {args.dest}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
