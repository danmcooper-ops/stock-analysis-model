# scripts/query_results.py
"""
Query historical analysis snapshots (output/results_YYYY-MM-DD.json).

Two modes:

1. Snapshot mode (default) — filter one day's results:
    python scripts/query_results.py                                # latest, top 50 by composite
    python scripts/query_results.py --date 2026-07-13 --rating BUY --sector Healthcare
    python scripts/query_results.py --where "composite>=57" --where "market_cap>1e10"
    python scripts/query_results.py --columns ticker,rating,mos --sort -mos --limit 20

2. History mode — track one ticker across every snapshot:
    python scripts/query_results.py --ticker AAPL --history
    python scripts/query_results.py --ticker AAPL --history --columns dcf_fv

Output goes to the console; add --csv PATH or --json PATH to also write a
file ('-' writes to stdout).

--where accepts  field OP value  with OP in  >= <= != == > <.  The value is
numeric when it parses as a float (1e10 works), else compared as a string
(country==Canada).  Rows missing a numerically-compared field are EXCLUDED
(NaN never satisfies a comparison).  Convenience aliases: composite/score →
_composite_score, composite_raw → _composite_score_raw, market_cap → mcap,
gates → _gates_passed (its "11/17" strings compare by the leading number).

3. SQL mode — query the snapshot store directly:
    python scripts/query_results.py --sql "SELECT date, count(*) FILTER (WHERE rating='BUY') AS buys FROM results GROUP BY date ORDER BY date"

History mode reads output/snapshots.duckdb (the DuckDB snapshot store, see
data/snapshot_store.py) when it holds every snapshot on disk — it carries all
scalar columns, so no column forces a slow path.  Without the store it falls
back to the older incremental parquet index under output/.query_index_v1/
(a slim column set; a column outside it triggers a full JSON scan).
--no-cache forces the full-scan path.  Build or refresh the store with:
    python scripts/ingest_snapshots.py --results-dir output
"""
import sys
import os
import re
import json
import argparse
import difflib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from data.snapshot_store import SnapshotStore, db_path_for
from data.snapshot_store import list_snapshot_files as _list_snapshot_files

# Slim per-snapshot column set persisted in the index cache. Bump the cache
# dir version suffix when this list changes so stale files rebuild cleanly.
INDEX_VERSION = 'v1'
INDEX_COLUMNS = [
    'date', 'ticker', 'rating', 'rating_raw', 'sector', 'industry',
    '_composite_score', '_composite_score_raw', 'price', 'mcap',
    'mos', 'roic', 'pe', '_gates_passed',
]

DEFAULT_COLUMNS = [
    'ticker', 'company_name', 'sector', 'rating', '_composite_score',
    'price', 'mos', 'roic', 'pe', 'mcap',
]

HISTORY_COLUMNS = [
    'date', 'rating', 'rating_raw', '_composite_score', 'price',
    'mos', 'roic', 'sector',
]

# Friendly names accepted anywhere a field is named (--where, --sort,
# --columns).
FIELD_ALIASES = {
    'composite': '_composite_score',
    'score': '_composite_score',
    'composite_raw': '_composite_score_raw',
    'market_cap': 'mcap',
    'gates': '_gates_passed',
}

_WHERE_RE = re.compile(r'^\s*([A-Za-z_]\w*)\s*(>=|<=|!=|==|>|<)\s*(.+?)\s*$')
_IDENT_RE = re.compile(r'^[A-Za-z_]\w*$')


# ---------------------------------------------------------------------------
# Snapshot discovery / loading
# ---------------------------------------------------------------------------

def list_snapshot_files(results_dir='output'):
    """[(date_str, path)] for canonical results_YYYY-MM-DD.json, sorted.

    Thin wrapper over data.snapshot_store.list_snapshot_files (the shared
    discovery helper): suffixed variants like results_*_replay.json are
    re-scored copies of earlier data and must never be queried as if they
    were live runs.
    """
    return _list_snapshot_files(results_dir)


def load_snapshot(path):
    """(meta, rows) from one snapshot. Older files may be a bare list."""
    with open(path, encoding='utf-8') as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        rows = data.get('results', [])
        meta = {k: v for k, v in data.items() if k != 'results'}
    else:
        rows, meta = data, {}
    return meta, rows


def pick_snapshot(files, date_str):
    """Resolve --date (or None → latest) to (date, path); helpful error."""
    if not files:
        sys.exit('No results_YYYY-MM-DD.json snapshots found in output/.')
    if date_str is None:
        return files[-1]
    for d, p in files:
        if d == date_str:
            return d, p
    dates = [d for d, _ in files]
    near = difflib.get_close_matches(date_str, dates, n=3, cutoff=0)
    sys.exit('No snapshot for %s. Nearest available: %s' % (date_str, ', '.join(near)))


# ---------------------------------------------------------------------------
# --where parsing / application
# ---------------------------------------------------------------------------

def resolve_field(name, available, context=''):
    """Map alias → canonical name; die with suggestions when unknown."""
    field = FIELD_ALIASES.get(name, name)
    if field in available:
        return field
    pool = sorted(set(list(available) + list(FIELD_ALIASES)))
    close = difflib.get_close_matches(name, pool, n=5, cutoff=0.4)
    sys.exit('Unknown field %r%s. Closest matches: %s'
             % (name, context, ', '.join(close) or '(none)'))


def parse_where(expr):
    m = _WHERE_RE.match(expr)
    if not m:
        sys.exit('Bad --where %r (expected: field OP value, OP in >= <= != == > <)' % expr)
    field, op, raw = m.groups()
    try:
        value = float(raw)
    except ValueError:
        value = raw
    return field, op, value


def _coerce_numeric(series):
    """to_numeric, additionally reading the leading number out of strings
    like _gates_passed's "11/17" so numeric filters work on them."""
    num = pd.to_numeric(series, errors='coerce')
    if len(num) and num.isna().all():
        extracted = series.astype(str).str.extract(r'^\s*(-?\d+\.?\d*)', expand=False)
        num = pd.to_numeric(extracted, errors='coerce')
    return num


def apply_wheres(df, wheres):
    for field, op, value in wheres:
        field = resolve_field(field, df.columns, context=' in --where')
        if isinstance(value, float):
            col = _coerce_numeric(df[field])
        else:
            col = df[field].astype(str)
        if op == '>=':
            mask = col >= value
        elif op == '<=':
            mask = col <= value
        elif op == '>':
            mask = col > value
        elif op == '<':
            mask = col < value
        elif op == '==':
            mask = col == value
        else:
            mask = col != value
        df = df[mask.fillna(False)]
    return df


# ---------------------------------------------------------------------------
# History-mode index cache
# ---------------------------------------------------------------------------

def _index_dir(results_dir):
    return os.path.join(results_dir, '.query_index_%s' % INDEX_VERSION)


def _index_paths(results_dir, date_str):
    base = os.path.join(_index_dir(results_dir), 'results_%s' % date_str)
    return base + '.parquet', base + '.csv.gz'


def _write_index(df, pq_path, csv_path):
    try:
        df.to_parquet(pq_path, index=False)
        return pq_path
    except Exception:
        # No usable parquet engine — degrade to csv.gz instead of dying.
        df.to_csv(csv_path, index=False, compression='gzip')
        return csv_path


def _read_index(pq_path, csv_path):
    if os.path.exists(pq_path):
        return pd.read_parquet(pq_path)
    return pd.read_csv(csv_path, compression='gzip')


def _index_fresh(src_path, pq_path, csv_path):
    dest = pq_path if os.path.exists(pq_path) else (csv_path if os.path.exists(csv_path) else None)
    return dest is not None and os.path.getmtime(dest) >= os.path.getmtime(src_path)


def build_index(results_dir='output', force=False):
    """Ensure every snapshot has a slim index file; return the union frame.

    Incremental: a snapshot is re-read only when its index file is missing
    or older than the source JSON.
    """
    files = list_snapshot_files(results_dir)
    if not files:
        sys.exit('No results_YYYY-MM-DD.json snapshots found in %s.' % results_dir)
    os.makedirs(_index_dir(results_dir), exist_ok=True)
    stale = [(d, p) for d, p in files
             if force or not _index_fresh(p, *_index_paths(results_dir, d))]
    if stale:
        print('Indexing %d of %d snapshots...' % (len(stale), len(files)), file=sys.stderr)
    frames = []
    for d, p in files:
        pq_path, csv_path = _index_paths(results_dir, d)
        if (d, p) in stale:
            _, rows = load_snapshot(p)
            df = pd.DataFrame(rows)
            df['date'] = d
            df = df.reindex(columns=INDEX_COLUMNS)
            _write_index(df, pq_path, csv_path)
            print('  indexed %s (%d rows)' % (d, len(df)), file=sys.stderr)
        else:
            df = _read_index(pq_path, csv_path)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def history_from_store(results_dir, ticker, columns):
    """One ticker's per-snapshot history from the DuckDB snapshot store.

    The store holds every scalar column of every ingested snapshot, so unlike
    the parquet index it never forces a full-scan fallback for an "off-index"
    column.  Returns a DataFrame (possibly empty) or None when the store is
    absent/unreadable, in which case the caller keeps its JSON path.
    """
    store = SnapshotStore.for_results_dir(results_dir)
    if store is None:
        return None
    try:
        with store:
            have = set(store.dates())
            if not have:
                return None
            # Only trust the store when it covers every snapshot on disk;
            # a partial store would silently truncate the ticker's history.
            on_disk = {d for d, _ in list_snapshot_files(results_dir)}
            if not on_disk.issubset(have):
                missing = len(on_disk - have)
                print('Snapshot store is missing %d snapshot(s) — reading JSON '
                      '(run scripts/ingest_snapshots.py to refresh).' % missing,
                      file=sys.stderr)
                return None
            known = set(store.column_types())
            wanted = [c for c in columns if c != 'date']
            sel = ['CAST(date AS VARCHAR) AS date']
            sel += [_sql_ident(c) for c in wanted if c in known]
            df = store.query(
                'SELECT %s FROM results WHERE ticker = ? ORDER BY date'
                % ', '.join(sel), [ticker])
            # A key that was None in every row of every snapshot has no
            # column. Fill it in as Python None (not a typed NA) so the
            # rendering matches the JSON path exactly.
            for c in wanted:
                if c not in known:
                    df[c] = None
    except Exception as e:
        print('Snapshot store read failed (%s) — reading JSON.' % e,
              file=sys.stderr)
        return None
    return df.reindex(columns=columns)


def _sql_ident(name):
    """Quote a column name for SQL (NULL when it isn't a plain identifier)."""
    if not _IDENT_RE.match(name):
        return 'NULL AS "%s"' % name.replace('"', '')
    return '"%s"' % name


def run_sql(results_dir, sql):
    """Run raw SQL against the snapshot store; returns a DataFrame.

    Exits with a message when the store is absent — there is nothing to query
    without it.
    """
    store = SnapshotStore.for_results_dir(results_dir)
    if store is None:
        sys.exit('No snapshot store at %s. Build one with:\n'
                 '  python scripts/ingest_snapshots.py --results-dir %s'
                 % (db_path_for(results_dir), results_dir))
    with store:
        try:
            return store.query(sql)
        except Exception as e:
            sys.exit('Query failed: %s' % e)


def history_full_scan(results_dir, ticker, columns):
    """Slow path: pull one ticker's row out of every snapshot JSON."""
    files = list_snapshot_files(results_dir)
    recs = []
    for d, p in files:
        _, rows = load_snapshot(p)
        for r in rows:
            if r.get('ticker') == ticker:
                rec = {'date': d}
                rec.update({c: r.get(c) for c in columns if c != 'date'})
                recs.append(rec)
                break
    return pd.DataFrame(recs, columns=columns)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def emit(df, args):
    if df.empty:
        print('No rows matched.')
    else:
        with pd.option_context('display.max_columns', None, 'display.width', None):
            print(df.to_string(index=False, na_rep='—',
                               float_format=lambda v: '%.2f' % v))
        print('\n%d row%s' % (len(df), '' if len(df) == 1 else 's'))
    for attr, writer in (('csv', lambda f: df.to_csv(f, index=False)),
                         ('json', lambda f: df.to_json(f, orient='records',
                                                       date_format='iso', indent=2))):
        dest = getattr(args, attr)
        if not dest:
            continue
        if dest == '-':
            writer(sys.stdout)
            print()
        else:
            writer(dest)
            print('Wrote %d rows to %s' % (len(df), dest))


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def cmd_snapshot(args):
    files = list_snapshot_files(args.results_dir)
    date_str, path = pick_snapshot(files, args.date)
    _, rows = load_snapshot(path)
    df = pd.DataFrame(rows)
    print('Snapshot %s — %d stocks' % (date_str, len(df)), file=sys.stderr)

    if args.rating:
        wanted = {r.upper() for r in args.rating}
        df = df[df['rating'].astype(str).str.upper().isin(wanted)]
    if args.sector:
        wanted = {s.lower() for s in args.sector}
        df = df[df['sector'].astype(str).str.lower().isin(wanted)]
    df = apply_wheres(df, [parse_where(w) for w in args.where])

    if args.columns:
        cols = [resolve_field(c.strip(), df.columns, ' in --columns')
                for c in args.columns.split(',') if c.strip()]
    else:
        cols = [c for c in DEFAULT_COLUMNS if c in df.columns]

    sort = args.sort or '-_composite_score'
    asc = not sort.startswith('-')
    sort_field = resolve_field(sort.lstrip('-'), df.columns, ' in --sort')
    df = df.sort_values(sort_field, ascending=asc, na_position='last',
                        key=lambda s: _coerce_numeric(s) if s.dtype == object else s)
    if sort_field not in cols:
        cols = cols + [sort_field]
    df = df[cols]
    if args.limit:
        df = df.head(args.limit)
    emit(df.reset_index(drop=True), args)


def cmd_history(args):
    ticker = args.ticker.upper()
    extra = [c.strip() for c in (args.columns or '').split(',') if c.strip()]
    extra = [FIELD_ALIASES.get(c, c) for c in extra]
    cols = HISTORY_COLUMNS + [c for c in extra if c not in HISTORY_COLUMNS]

    df = None if args.no_cache else history_from_store(
        args.results_dir, ticker, cols)
    if df is None:
        off_index = [c for c in cols if c not in INDEX_COLUMNS]
        if args.no_cache or off_index:
            if off_index and not args.no_cache:
                print('Column(s) %s not in the index — full scan (slower).'
                      % ', '.join(off_index), file=sys.stderr)
            df = history_full_scan(args.results_dir, ticker, cols)
        else:
            idx = build_index(args.results_dir, force=False)
            df = idx[idx['ticker'] == ticker][cols]

    if df.empty:
        sys.exit('Ticker %s not found in any snapshot.' % ticker)
    df = df.sort_values('date')
    emit(df.reset_index(drop=True), args)


def main():
    ap = argparse.ArgumentParser(
        description='Query historical results_YYYY-MM-DD.json snapshots.',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('--results-dir', default='output')
    ap.add_argument('--date', help='snapshot date YYYY-MM-DD (default: latest)')
    ap.add_argument('--rating', action='append', default=[],
                    help='filter by rating; repeatable (BUY, "LEAN BUY", ...)')
    ap.add_argument('--sector', action='append', default=[],
                    help='filter by sector; repeatable')
    ap.add_argument('--where', action='append', default=[],
                    help='metric filter, e.g. "composite>=57"; repeatable')
    ap.add_argument('--columns', help='comma-separated output columns')
    ap.add_argument('--sort', help='sort field; prefix - for descending '
                                   '(default: -composite)')
    ap.add_argument('--limit', type=int, default=50,
                    help='max rows (0 = all; default 50)')
    ap.add_argument('--ticker', help='ticker for --history mode')
    ap.add_argument('--history', action='store_true',
                    help='track --ticker across all snapshots')
    ap.add_argument('--sql', metavar='QUERY',
                    help='Run raw SQL against the snapshot store '
                         '(tables: results, runs) and print the result. '
                         'Ignores every other filter/selection flag.')
    ap.add_argument('--no-cache', action='store_true',
                    help='history mode: bypass the index cache (full scan)')
    ap.add_argument('--csv', help="also write CSV to PATH ('-' = stdout)")
    ap.add_argument('--json', help="also write JSON to PATH ('-' = stdout)")
    args = ap.parse_args()

    if args.sql:
        emit(run_sql(args.results_dir, args.sql), args)
    elif args.history:
        if not args.ticker:
            ap.error('--history requires --ticker')
        cmd_history(args)
    else:
        cmd_snapshot(args)


if __name__ == '__main__':
    main()
