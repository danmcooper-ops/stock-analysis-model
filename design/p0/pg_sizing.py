"""P0 spike: measure Postgres storage for the planned Supabase schema.

Loads real snapshots into a prototype of the design in
``design/supabase-migration.md``: typed columns for the keys the scorers and
readers read, everything else in an lz4-compressed ``extra jsonb``, and
``edgar_history`` stored once per distinct value in ``edgar_blobs``. Report-only
narrative keys (``DEFAULT_EXCLUDE_KEYS``) are left out, as the plan does.

Throwaway measurement code, not the production loader: it creates a schema
``p0`` in the target database and drops it first.

    python design/p0/pg_sizing.py --dsn postgresql://postgres@127.0.0.1:55432/postgres \
        output/results_2026-09-*.json.gz
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import psycopg  # noqa: E402

from data.snapshot_store import DEFAULT_EXCLUDE_KEYS, read_snapshot, snapshot_date_from_path  # noqa: E402
from scripts.report_html import _PREV_DRIVER_KEYS  # noqa: E402
from scripts.scoring import APPLICABILITY_FIELDS, GATES  # noqa: E402

# Scoring / reader inputs beyond the gate lists (see plan: column registry).
_OTHER_HOT = (
    'ticker', 'company_name', 'name', 'sector', 'industry', 'exchange', 'cik', 'price', 'mcap',
    'shares_out', 'data_source', 'source_group', 'currency',
    '_risk_free_rate', 'mc_cv', '_fv_effective', 'beneish_flag', 'altman_z_zone',
    'fx_fetch_failed', 'edgar_quality_score', 'avg_dollar_volume_3m', 'sbc', 'revenue',
    'fcf', 'fcf_edgar', 'int_cov_edgar', 'dcf_fv', '_dcf_fv_preblend', 'epv_growth_fv',
    'rim_fv', 'ddm_fv', 'sbc_pct_rev_xbrl', 'pfcf', 'pp_margin_advantage',
    'operating_margin', '_sector_median_opm', 'operating_income', 'enterprise_value',
    'op_margin_avg_10y', 'rev_down_years', 'rev_cagr_5y', 'fcf_neg_years_5y',
    'net_debt_slope_3y', 'roic', 'wacc', 'roic_trend_slope', 'div_fcf_ratio_3y',
    'momentum_12_1', 'short_pct_float', 'pp_multiple', 'trap_score',
    'trap_score_ex_momentum', '_gates_passed', '_gates_passed_num', 'pe', 'beta',
)
EDGAR_KEY = 'edgar_history'
# Filing-driven series that change only when a company files: stored with
# edgar_history in one content-addressed blob per distinct value.
FILING_KEYS = (EDGAR_KEY, 'roic_by_year', '_nopat_by_year', '_ic_by_year')


def hot_keys():
    keys = set(_PREV_DRIVER_KEYS) | set(APPLICABILITY_FIELDS) | set(_OTHER_HOT)
    for g in GATES:
        short = g.field
        keys.add(g.field)
        keys.update({f'_gate_{short}', f'_score_{short}', f'_gp_{short}'})
    return keys


def _kind(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return 'boolean'
    if isinstance(v, int):
        return 'bigint'
    if isinstance(v, float):
        return 'double precision'
    if isinstance(v, str):
        return 'text'
    return 'jsonb'


_RANK = {'boolean': 0, 'bigint': 1, 'double precision': 2, 'text': 3, 'jsonb': 4}


def infer_types(snapshots, hot):
    """Widest observed type per hot key; keys never seen non-null are dropped."""
    types = {}
    for _, _, rows in snapshots:
        for r in rows:
            for k in hot:
                t = _kind(r.get(k))
                if t and (k not in types or _RANK[t] > _RANK[types[k]]):
                    types[k] = t
    return {k: t for k, t in types.items() if k != 'ticker'}


def encode(v):
    """jsonb-safe codec: non-finite floats -> {"$nf": ...}, '$'-keyed dicts escaped."""
    if isinstance(v, float) and not math.isfinite(v):
        return {'$nf': 'NaN' if math.isnan(v) else ('Inf' if v > 0 else '-Inf')}
    if isinstance(v, dict):
        out = {str(k): encode(x) for k, x in v.items()}
        return {'$lit': out} if any(str(k).startswith('$') for k in v) else out
    if isinstance(v, (list, tuple)):
        return [encode(x) for x in v]
    if isinstance(v, str):
        return v.replace('\x00', '\\u0000')
    return v


def dumps(v):
    return json.dumps(encode(v), separators=(',', ':'), allow_nan=False)


def cast(v, t):
    """Value for a typed column, or raise ValueError so it goes to extra."""
    if v is None:
        return None
    if t == 'boolean':
        if isinstance(v, bool):
            return v
        raise ValueError
    if t == 'bigint':
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError
        return v
    if t == 'double precision':
        if isinstance(v, bool):
            raise ValueError
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str) and v in ('Infinity', '-Infinity', 'NaN'):
            return float(v)
        raise ValueError
    if t == 'text':
        return v if isinstance(v, str) else json.dumps(v)
    return dumps(v)


def ddl(types):
    cols = ',\n  '.join(f'"{k}" {t}' for k, t in sorted(types.items()))
    return f"""
DROP SCHEMA IF EXISTS p0 CASCADE;
CREATE SCHEMA p0;
CREATE TABLE p0.tickers (ticker_id int GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                         ticker text UNIQUE NOT NULL);
CREATE TABLE p0.edgar_blobs (sha text PRIMARY KEY, value jsonb COMPRESSION lz4 NOT NULL);
CREATE TABLE p0.results (
  run_date date NOT NULL,
  ticker_id int NOT NULL,
  {cols},
  extra jsonb COMPRESSION lz4,
  edgar_history_sha text,
  PRIMARY KEY (run_date, ticker_id)
) PARTITION BY RANGE (run_date);
CREATE TABLE p0.results_2026 PARTITION OF p0.results
  FOR VALUES FROM ('2026-01-01') TO ('2027-01-01');
CREATE INDEX ON p0.results (ticker_id, run_date DESC);
CREATE INDEX ON p0.results (run_date, rating);
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--dsn', required=True)
    ap.add_argument('--promote', choices=('hot', 'scalar'), default='hot',
                    help="typed columns: 'hot' = keys the scorers/readers read; "
                         "'scalar' = every key whose values are never dict/list")
    ap.add_argument('--dedup', choices=('edgar', 'filing'), default='edgar',
                    help="content-addressed blob holds edgar_history only, or all FILING_KEYS")
    ap.add_argument('files', nargs='+')
    a = ap.parse_args()

    t0 = time.time()
    snaps = []
    for p in sorted(a.files):
        d = read_snapshot(p)
        snaps.append((snapshot_date_from_path(p), p, d['results']))
    print(f'parsed {len(snaps)} snapshots in {time.time() - t0:.0f}s')

    all_keys = set().union(*(r.keys() for _, _, rows in snaps for r in rows))
    excluded = set(DEFAULT_EXCLUDE_KEYS)
    hot = hot_keys()
    if a.promote == 'scalar':
        hot |= all_keys - excluded - {EDGAR_KEY}
    blob_keys = FILING_KEYS if a.dedup == 'filing' else (EDGAR_KEY,)
    hot -= set(blob_keys)
    types = infer_types(snaps, hot)
    if a.promote == 'scalar':
        types = {k: t for k, t in types.items() if t != 'jsonb'}
    print(f'distinct row keys: {len(all_keys)}; typed columns: {len(types)}; '
          f'excluded (report-only): {len(excluded & all_keys)}')

    cols = sorted(types)
    stats = {'rows': 0, 'cast_fail': 0, 'typed_values': 0, 'blob_new': 0, 'blob_refs': 0,
             'extra_bytes': 0, 'json_bytes': 0}
    with psycopg.connect(a.dsn, autocommit=True) as con:
        con.execute(ddl(types))
        tick_ids = {}
        blobs_seen = set()
        load_s = []
        for run_date, _, rows in snaps:
            t1 = time.time()
            with con.transaction():
                new = sorted({r['ticker'] for r in rows} - tick_ids.keys())
                if new:
                    with con.cursor() as cur:
                        cur.executemany('INSERT INTO p0.tickers (ticker) VALUES (%s) ON CONFLICT DO NOTHING', [(t,) for t in new])
                    for tid, t in con.execute('SELECT ticker_id, ticker FROM p0.tickers WHERE ticker = ANY(%s)', (new,)):
                        tick_ids[t] = tid
                blob_rows = []
                out = []
                seen_today = set()
                for r in rows:
                    if r['ticker'] in seen_today:
                        continue
                    seen_today.add(r['ticker'])
                    stats['json_bytes'] += len(json.dumps(r, default=str))
                    rec = []
                    extra = {}
                    for k in cols:
                        v = r.get(k)
                        try:
                            rec.append(cast(v, types[k]))
                            if v is not None:
                                stats['typed_values'] += 1
                        except ValueError:
                            stats['cast_fail'] += 1
                            rec.append(None)
                            extra[k] = v
                    for k, v in r.items():
                        if k in types or k == 'ticker' or k in excluded or k in blob_keys:
                            continue
                        extra[k] = v
                    sha = None
                    eh = {k: r[k] for k in blob_keys if r.get(k) is not None}
                    if eh:
                        body = dumps(eh)
                        sha = hashlib.sha256(body.encode()).hexdigest()
                        stats['blob_refs'] += 1
                        if sha not in blobs_seen:
                            blobs_seen.add(sha)
                            blob_rows.append((sha, body))
                    ex = dumps(extra)
                    stats['extra_bytes'] += len(ex)
                    out.append((run_date, tick_ids[r['ticker']], *rec, ex, sha))
                stats['blob_new'] += len(blob_rows)
                with con.cursor() as cur:
                    with cur.copy('COPY p0.edgar_blobs (sha, value) FROM STDIN') as cp:
                        for b in blob_rows:
                            cp.write_row(b)
                    collist = ', '.join(['run_date', 'ticker_id', *[f'"{c}"' for c in cols], 'extra', 'edgar_history_sha'])
                    with cur.copy(f'COPY p0.results ({collist}) FROM STDIN') as cp:
                        for rec in out:
                            cp.write_row(rec)
                stats['rows'] += len(out)
            load_s.append(time.time() - t1)
            print(f'  {run_date}: {len(out)} rows, {len(blob_rows)} new edgar blobs, {load_s[-1]:.1f}s')
        con.execute('VACUUM ANALYZE p0.results_2026')
        con.execute('VACUUM ANALYZE p0.edgar_blobs')
        q = con.execute("""
            SELECT pg_table_size('p0.results_2026'), pg_indexes_size('p0.results_2026'),
                   pg_total_relation_size('p0.edgar_blobs'), pg_total_relation_size('p0.tickers')""").fetchone()
    res_tbl, res_idx, blob_total, tick_total = q
    n_days = len(snaps)
    rows_per_day = stats['rows'] / n_days
    per_row = (res_tbl + res_idx) / stats['rows']
    print()
    print(f"rows loaded            {stats['rows']:,} ({rows_per_day:,.0f}/day)")
    print(f"typed values           {stats['typed_values']:,}; cast failures -> extra: {stats['cast_fail']:,} "
          f"({stats['cast_fail'] / max(1, stats['typed_values']):.3%})")
    print(f"edgar blobs            {stats['blob_new']:,} distinct for {stats['blob_refs']:,} refs")
    print(f"results table          {res_tbl / 2**20:,.1f} MiB  + indexes {res_idx / 2**20:,.1f} MiB  = {per_row:,.0f} B/row")
    print(f"edgar_blobs            {blob_total / 2**20:,.1f} MiB   tickers {tick_total / 2**20:,.2f} MiB")
    print(f"raw row JSON (all keys) {stats['json_bytes'] / 2**20:,.1f} MiB;  extra payload {stats['extra_bytes'] / 2**20:,.1f} MiB (pre-TOAST)")
    print(f"mean load per day      {sum(load_s) / n_days:.1f}s")
    # Projection: results scale with rows; edgar blobs scale with the per-day
    # count of NEW distinct histories (the first day seeds the whole universe).
    for label, per_day in (('today', rows_per_day), ('8k tickers', 8000.0)):
        scale = per_day / rows_per_day
        res_year = per_row * per_day * 252
        # blob growth: observed new-blob rate over days 2..n, scaled
        blob_bytes_each = blob_total / max(1, stats['blob_new'])
        seed = rows_per_day * scale * blob_bytes_each
        new_per_day = max(0.0, (stats['blob_new'] - rows_per_day) / max(1, n_days - 1)) * scale
        blob_year = seed + new_per_day * 252 * blob_bytes_each
        print(f"projection {label:>10}: {per_day:,.0f} rows/day -> results {res_year / 2**30:,.2f} GiB/yr"
              f" + edgar {blob_year / 2**30:,.2f} GiB/yr = {(res_year + blob_year) / 2**30:,.2f} GiB/yr")


if __name__ == '__main__':
    main()
