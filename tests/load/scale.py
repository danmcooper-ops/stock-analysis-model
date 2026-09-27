#!/usr/bin/env python3
"""P5 scale harness for the Supabase database (design/supabase-migration.md).

Builds a synthetic history of N business days x T tickers from real snapshot
rows, then measures what the plan's scalability section asks for:

  seed      publish one real snapshot through the nightly path; its rows are
            the templates every synthetic row is cloned from
  generate  add synthetic tickers and days (bulk SQL, idempotent, append-only)
            plus their rating change points and latest pointers
  publish   publish the next business day as an 8k-row snapshot through the
            Data API, as run.sh step 06a does, and time it
  explain   EXPLAIN every reader query and check which partitions it touches
  bench     cold-cache latency of the reader queries (restart Postgres first)
  load      publish a day while reader threads run; check no reader ever sees
            a partial day and record their latency
  size      table sizes, bytes per row and the per-year projection
  anon      unauthenticated Data API calls must be refused

Every step prints one JSON document; ``all`` runs them in order and writes
the combined report to ``--out``.

Synthetic rows keep the template's width and value distributions: every
typed column, ``extra`` and the ``edgar_history`` reference are copied, the
headline numbers get a small deterministic wobble, and each ticker's rating
walks through runs of 20-80 days (about 2% of tickers change per day, as the
real ratings do: a median of 53 changes a night over ~2.5k tickers).

Usage (local stack from ``supabase start``):
    python tests/load/scale.py all --tickers 8000 --days 188 --out output/p5_half.json
    python tests/load/scale.py all --tickers 8000 --days 375 --out output/p5_full.json

The second call appends days 189-375 to the first, so the two reports are the
growth check (plan item 5). It needs roughly 4 KB of disk per row.
"""
import argparse
import datetime as dt
import json
import math
import os
import random
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

RATINGS = ('BUY', 'LEAN BUY', 'HOLD', 'PASS')
FIRST_DAY = dt.date(2027, 1, 4)           # a Monday; partitions exist through 2031
SYN_PREFIX = 'SYN'
WOBBLE = ('price', 'mos', '_composite_score', 'dcf_fv', 'mcap')
HISTORY_COLS = ('rating', 'mos', 'price', 'dcf_fv', '_composite_score')
TARGETS_MS = {'ticker_history_5y': 20, 'last_known_rows': 2000, 'rating_changes_since': 50, 'export': 60_000}
DEFAULT_DSN = 'postgresql://postgres:postgres@127.0.0.1:54322/postgres'
DEFAULT_API = 'http://127.0.0.1:54321'
DB_CONTAINER = 'supabase_db_stock-analysis-model'


# --- pure helpers (unit tested in tests/test_db_scale.py) --------------------

def business_days(start, n):
    """The first *n* weekdays from *start* (holidays ignored: the shape, not
    the calendar, is what matters here)."""
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def ticker_params(i):
    """Deterministic (seed, run length) of synthetic ticker *i*."""
    rnd = random.Random(i * 7919 + 17)
    return rnd.randrange(1_000_000), rnd.randint(20, 80)


def rating_for(seed, runlen, day_index):
    """The rating of a ticker on day *day_index*: runs of *runlen* days,
    phase-shifted by *seed* so tickers do not all change on the same day."""
    return RATINGS[(seed + (day_index + seed) // runlen) % len(RATINGS)]


def rating_sql(day_expr):
    """SQL twin of :func:`rating_for` over ``s.seed``/``s.runlen``."""
    arr = 'ARRAY[' + ','.join(f"'{r}'" for r in RATINGS) + ']'
    # mod(), not %: the statement also carries psycopg placeholders.
    return f'({arr})[mod(s.seed + ({day_expr} + s.seed) / s.runlen, {len(RATINGS)}) + 1]'


def percentile(values, p):
    """Nearest-rank percentile (p in 0..100) of a non-empty list."""
    xs = sorted(values)
    k = max(0, math.ceil(p / 100 * len(xs)) - 1)
    return xs[k]


def summarize(samples_ms):
    return {'n': len(samples_ms), 'first_ms': round(samples_ms[0], 2),
            'p50_ms': round(percentile(samples_ms, 50), 2), 'p95_ms': round(percentile(samples_ms, 95), 2),
            'max_ms': round(max(samples_ms), 2)}


def partitions_in_plan(plan):
    """Names of the ``results_*`` relations a JSON EXPLAIN plan scans, and the
    number of subplans removed by run-time pruning."""
    names, removed = set(), 0

    def walk(node):
        nonlocal removed
        rel = node.get('Relation Name', '')
        if rel.startswith('results_'):
            names.add(rel)
        removed += node.get('Subplans Removed', 0)
        for child in node.get('Plans', []):
            walk(child)
    walk(plan[0]['Plan'] if isinstance(plan, list) else plan)
    return names, removed


def node_types(plan):
    """Every node type in a JSON EXPLAIN plan."""
    out = set()

    def walk(node):
        out.add(node.get('Node Type'))
        for child in node.get('Plans', []):
            walk(child)
    walk(plan[0]['Plan'] if isinstance(plan, list) else plan)
    return out


def synthetic_snapshot(template, tickers, day_index, run_date):
    """A snapshot dict for *run_date*: template rows re-labelled as the
    synthetic tickers, with the rating the generator would give that day."""
    base = [r for r in template['results'] if isinstance(r, dict) and r.get('ticker')]
    rows = []
    for i in range(tickers):
        row = dict(base[i % len(base)])
        seed, runlen = ticker_params(i)
        row['ticker'] = f'{SYN_PREFIX}{i:05d}'
        if row.get('rating'):
            row['rating'] = rating_for(seed, runlen, day_index)
        rows.append(row)
    meta = {k: v for k, v in template.items() if k != 'results'}
    return {**meta, 'date': run_date.isoformat(), 'results': rows}


# --- database plumbing --------------------------------------------------------

def _connect(dsn):
    from data.db.connect import connect
    return connect(dsn, autocommit=True)


def _service_key():
    """The local stack's service_role JWT (the CLI's fixed demo secret)."""
    import base64
    import hashlib
    import hmac
    secret = os.environ.get('SUPABASE_JWT_SECRET', 'super-secret-jwt-token-with-at-least-32-characters-long')

    def b64(b):
        return base64.urlsafe_b64encode(b).rstrip(b'=').decode()
    head = b64(json.dumps({'alg': 'HS256', 'typ': 'JWT'}).encode())
    body = b64(json.dumps({'role': 'service_role', 'iss': 'supabase-demo',
                           'exp': int(time.time()) + 6 * 3600}).encode())
    sig = b64(hmac.new(secret.encode(), f'{head}.{body}'.encode(), hashlib.sha256).digest())
    return f'{head}.{body}.{sig}'


def _transport(a):
    from data.db.publish import DirectTransport, RestTransport
    if a.direct:
        return DirectTransport(_connect(a.dsn))
    return RestTransport(a.api, os.environ.get('SUPABASE_SERVICE_ROLE_KEY') or _service_key(),
                         timeout=(5, 900), retries=0)


def _template_date(con):
    row = con.execute("SELECT min(run_date) FROM core.runs WHERE status = 'complete' AND run_date < %s",
                      (FIRST_DAY,)).fetchone()
    if not row[0]:
        raise SystemExit('no template run: run the seed step first')
    return row[0]


def _syn_days(con):
    """Synthetic run dates present (the harness's own days only), oldest first."""
    return [r[0] for r in con.execute('SELECT r.run_date FROM core.runs r JOIN bench.days d USING (run_date) '
                                      'ORDER BY r.run_date').fetchall()]


def _latest(con):
    return _syn_days(con)[-1]


# --- steps ----------------------------------------------------------------------

def step_seed(a):
    """Publish the template snapshot through the real nightly path."""
    from data.db.publish import build_load, publish
    from data.snapshot_store import read_snapshot, snapshot_date_from_path
    con = _connect(a.dsn)
    have = con.execute('SELECT count(*) FROM core.runs WHERE run_date < %s', (FIRST_DAY,)).fetchone()[0]
    if have:
        return {'seed': 'already seeded', 'template_date': str(_template_date(con))}
    d = snapshot_date_from_path(a.template)
    data = read_snapshot(a.template)
    t0 = time.time()
    res = publish(build_load(data, d), _transport(a), min_row_ratio=0)
    return {'seed': d, 'rows': res.get('rows'), 'seconds': round(time.time() - t0, 1)}


def _columns(con):
    return [r[0] for r in con.execute(
        "SELECT attname FROM pg_attribute WHERE attrelid = 'core.results'::regclass "
        'AND attnum > 0 AND NOT attisdropped ORDER BY attnum').fetchall()]


def step_generate(a):
    """Add synthetic tickers and days up to --tickers x --days (append-only)."""
    con = _connect(a.dsn)
    tpl_date = _template_date(con)
    con.execute('CREATE SCHEMA IF NOT EXISTS bench')
    con.execute('CREATE TABLE IF NOT EXISTS bench.templates AS '
                'SELECT (row_number() OVER (ORDER BY ticker_id) - 1)::int AS tpl, r.* '
                'FROM core.results r WHERE r.run_date = %s' % _lit(tpl_date))
    n_tpl = con.execute('SELECT count(*) FROM bench.templates').fetchone()[0]
    con.execute('CREATE TABLE IF NOT EXISTS bench.syn (i int PRIMARY KEY, ticker_id bigint NOT NULL, '
                'tpl int NOT NULL, seed int NOT NULL, runlen int NOT NULL)')
    have = con.execute('SELECT count(*) FROM bench.syn').fetchone()[0]
    days = business_days(FIRST_DAY, a.days)
    if have < a.tickers:
        with con.transaction():
            for i in range(have, a.tickers):
                seed, runlen = ticker_params(i)
                tid = con.execute('INSERT INTO core.tickers (ticker, first_seen, last_seen) VALUES (%s, %s, %s) '
                                  'ON CONFLICT (ticker) DO UPDATE SET last_seen = EXCLUDED.last_seen '
                                  'RETURNING ticker_id', (f'{SYN_PREFIX}{i:05d}', days[0], days[-1])).fetchone()[0]
                con.execute('INSERT INTO bench.syn VALUES (%s, %s, %s, %s, %s)', (i, tid, i % n_tpl, seed, runlen))
    elif have > a.tickers:
        raise SystemExit(f'{have} synthetic tickers exist; the harness only appends (reset with --reset)')
    con.execute('CREATE TABLE IF NOT EXISTS bench.days (day_index int PRIMARY KEY, run_date date UNIQUE)')
    con.execute('INSERT INTO bench.days SELECT i, d FROM unnest(%s::date[]) WITH ORDINALITY u(d, i1), '
                'LATERAL (SELECT (i1 - 1)::int AS i) x ON CONFLICT DO NOTHING', (days,))
    cols = [c for c in _columns(con) if c not in ('run_date', 'ticker_id')]
    sel = []
    for c in cols:
        q = f'"{c}"'
        if c == 'rating':
            sel.append(f'CASE WHEN t.rating IS NULL OR t.rating = \'\' THEN t.rating ELSE {rating_sql("d.day_index")} END')
        elif c in WOBBLE:
            sel.append(f't.{q} * (1 + 0.01 * sin(d.day_index + s.seed))')
        else:
            sel.append(f't.{q}')
    insert = (f'INSERT INTO core.results (run_date, ticker_id, {", ".join(chr(34) + c + chr(34) for c in cols)}) '
              f'SELECT d.run_date, s.ticker_id, {", ".join(sel)} '
              'FROM bench.syn s JOIN bench.templates t ON t.tpl = s.tpl, bench.days d '
              'WHERE d.run_date = %s')
    meta = con.execute('SELECT meta, risk_free_rate FROM core.runs WHERE run_date = %s', (tpl_date,)).fetchone()
    present = set(_syn_days(con))
    todo = [d for d in days if d not in present]
    t0, done = time.time(), 0
    for d in todo:
        with con.transaction():
            con.execute('INSERT INTO core.runs (run_date, status, risk_free_rate, n_rows, source_sha256, meta, '
                        'completed_at) VALUES (%s, \'complete\', %s, %s, %s, %s, now())',
                        (d, meta[1], a.tickers, '0' * 64, json.dumps(meta[0])))
            con.execute(insert, (d,))
        done += 1
        if done % 25 == 0:
            print(f'  generated {done}/{len(todo)} days ({time.time() - t0:.0f}s)', file=sys.stderr, flush=True)
    gen_s = time.time() - t0
    t1 = time.time()
    with con.transaction():   # change points and latest pointers, from scratch for the synthetic tickers
        con.execute('DELETE FROM core.rating_changes WHERE ticker_id IN (SELECT ticker_id FROM bench.syn)')
        con.execute("""
            INSERT INTO core.rating_changes (ticker_id, run_date, rating, prev_rating)
            SELECT ticker_id, run_date, rating, prev FROM (
              SELECT r.ticker_id, r.run_date, r.rating,
                     lag(r.rating) OVER (PARTITION BY r.ticker_id ORDER BY r.run_date) AS prev
                FROM core.results r JOIN bench.syn s ON s.ticker_id = r.ticker_id
               WHERE r.rating IS NOT NULL AND r.rating <> '') x
             WHERE prev IS NULL OR prev <> rating""")
        con.execute("""
            INSERT INTO core.latest_results (ticker_id, run_date)
            SELECT r.ticker_id, max(r.run_date) FROM core.results r JOIN bench.syn s ON s.ticker_id = r.ticker_id
             GROUP BY r.ticker_id
            ON CONFLICT (ticker_id) DO UPDATE SET run_date = EXCLUDED.run_date""")
    derive_s = time.time() - t1
    t2 = time.time()
    con.execute('VACUUM (ANALYZE) core.results')
    con.execute('VACUUM (ANALYZE) core.rating_changes')
    return {'tickers': a.tickers, 'days_added': len(todo), 'days_total': len(_syn_days(con)),
            'generate_s': round(gen_s, 1), 'change_points_s': round(derive_s, 1),
            'vacuum_s': round(time.time() - t2, 1)}


def _lit(d):
    return "'" + d.isoformat() + "'"


def _prepare_publish(a, run_date=None):
    """Build the Load for *run_date* (default: the next business day)."""
    from data.db.publish import build_load
    from data.snapshot_store import read_snapshot
    con = _connect(a.dsn)
    try:
        if run_date is None:
            run_date = business_days(_syn_days(con)[-1] + dt.timedelta(days=1), 1)[0]
        day_index = sum(1 for _ in _days_between(FIRST_DAY, run_date))
        snap = synthetic_snapshot(read_snapshot(a.template), a.tickers, day_index, run_date)
        t0 = time.time()
        load = build_load(snap, run_date.isoformat())
        build_s = time.time() - t0
        con.execute('INSERT INTO bench.days VALUES (%s, %s) ON CONFLICT DO NOTHING', (day_index, run_date))
    finally:
        con.close()
    return run_date, load, build_s


def _publish_prepared(a, run_date, load, build_s, replace=False):
    from data.db.publish import publish
    res = publish(load, _transport(a), min_row_ratio=0, force=replace, reason='P5 replace test' if replace else None)
    return {'run_date': str(run_date), 'rows': res.get('rows'), 'replaced_rows': res.get('replaced_rows'),
            'rating_changes': res.get('rating_changes'), 'chunks': res.get('chunks'),
            'build_s': round(build_s, 1), 'staged_s': res.get('staged_s'), 'total_s': res.get('total_s'),
            'target_s': 300, 'ok': res.get('total_s', 1e9) + build_s < 300}


def step_publish(a, run_date=None, replace=False):
    """Publish the next business day (or *run_date*) as a synthetic snapshot,
    through the Data API unless --direct (the nightly path, run.sh 06a)."""
    return _publish_prepared(a, *_prepare_publish(a, run_date), replace=replace)


def _days_between(start, end):
    """Weekdays in [start, end)."""
    d = start
    while d < end:
        if d.weekday() < 5:
            yield d
        d += dt.timedelta(days=1)


# --- reader queries -------------------------------------------------------------

def _q_history(con, ticker, since, until):
    """pipeline.ticker_history: bounded on both sides, since an open-ended
    range cannot prune the (empty) partitions created for future years."""
    return con.execute('SELECT pipeline.ticker_history(%s, %s, %s)', (ticker, since, until)).fetchone()[0]


def _syn_tickers(con):
    """``[(ticker_id, ticker)]`` of the synthetic tickers."""
    return con.execute('SELECT s.ticker_id, t.ticker FROM bench.syn s JOIN core.tickers t USING (ticker_id) '
                       'ORDER BY s.i').fetchall()


def _q_changes(con, since):
    return con.execute('SELECT t.ticker, rc.run_date, rc.rating, rc.prev_rating FROM core.rating_changes rc '
                       'JOIN core.tickers t ON t.ticker_id = rc.ticker_id WHERE rc.run_date >= %s '
                       'ORDER BY rc.run_date, t.ticker', (since,)).fetchall()


LKR_COLS = ['rating', 'price', 'mos', 'dcf_fv', 'mcap', 'sector', 'company_name', '_composite_score',
            'roic', 'wacc', 'pe', 'beta_raw', 'fcf_yield', 'net_debt', 'div_yield', 'country',
            'industry', 'float_shares', 'ev_ebitda', 'altman_z']


def _q_lkr(con, before, cols=LKR_COLS):
    return con.execute('SELECT jsonb_array_length(pipeline.last_known_rows(%s, %s::text[], 7) -> \'rows\')',
                       (before, cols)).fetchone()[0]


def _export(dsn, run_date, tmpdir):
    """The export step: read the day's full rows back and write the Parquet file."""
    from data.db.parquet import export_snapshot
    from data.db.publish import DirectTransport
    from data.db.reader import DbStore
    con = _connect(dsn)
    try:
        store = DbStore(DirectTransport(con))
        rows = store.rows(run_date)
        meta = store.run_meta(run_date) or {}
        n, _ = export_snapshot({**{k: v for k, v in meta.items() if k != 'results'}, 'results': rows},
                               run_date, os.path.join(tmpdir, f'results_{run_date}.parquet'))
        return n
    finally:
        con.close()


def restart_db(a):
    """Cold cache: restart Postgres (empties shared_buffers) and try to drop
    the OS page cache. Returns whether the OS cache was dropped."""
    subprocess.run(['docker', 'restart', a.container], check=True, capture_output=True, timeout=180)
    for _ in range(120):
        try:
            _connect(a.dsn).close()
            break
        except Exception:
            time.sleep(1)
    try:
        subprocess.run(['sync'], check=False)
        with open('/proc/sys/vm/drop_caches', 'w', encoding='utf-8') as f:
            f.write('3\n')
        return True
    except OSError:
        return False


def step_bench(a):
    """Cold-cache latency of each reader query, targets from the plan."""
    import tempfile
    con = _connect(a.dsn)
    latest = _latest(con)
    syn = _syn_days(con)
    syn_t = _syn_tickers(con)
    rnd = random.Random(42)
    pick = rnd.sample(syn_t, min(a.samples, len(syn_t)))
    since5y = latest - dt.timedelta(days=5 * 365)
    con.close()
    os_dropped = restart_db(a) if not a.no_restart else None
    con = _connect(a.dsn)
    out = {'latest': str(latest), 'days': len(syn), 'os_cache_dropped': os_dropped}
    ms = []
    rows_back = 0
    for _, ticker in pick:
        t = time.perf_counter()
        rows_back = len(_q_history(con, ticker, since5y, latest))
        ms.append((time.perf_counter() - t) * 1000)
    out['ticker_history_5y'] = dict(summarize(ms), rows=rows_back)
    ms = []
    for tid, _ in pick[: max(5, a.samples // 5)]:
        t = time.perf_counter()
        con.execute('SELECT * FROM core.results WHERE ticker_id = %s AND run_date BETWEEN %s AND %s '
                    'ORDER BY run_date', (tid, since5y, latest)).fetchall()
        ms.append((time.perf_counter() - t) * 1000)
    out['ticker_history_5y_all_columns'] = summarize(ms)
    ms = []
    for k in range(min(a.samples, len(syn) - 1)):
        since = syn[max(0, len(syn) - 6 - (k % 20))]
        t = time.perf_counter()
        n = len(_q_changes(con, since))
        ms.append((time.perf_counter() - t) * 1000)
    out['rating_changes_since'] = dict(summarize(ms), rows_last=n)
    ms = []
    for k in range(5):
        before = syn[-1 - k] + dt.timedelta(days=1)
        t = time.perf_counter()
        n = _q_lkr(con, before)
        ms.append((time.perf_counter() - t) * 1000)
    out['last_known_rows'] = dict(summarize(ms), rows=n, columns=len(LKR_COLS))
    ms = []
    for k in range(3):
        before = syn[-1 - k] + dt.timedelta(days=1)
        t = time.perf_counter()
        _q_lkr(con, before, cols=None)
        ms.append((time.perf_counter() - t) * 1000)
    out['last_known_rows_all_columns'] = summarize(ms)
    t = time.perf_counter()
    n = len(con.execute('SELECT pipeline.rating_history(NULL)').fetchone()[0])
    out['rating_history_all'] = {'ms': round((time.perf_counter() - t) * 1000, 1), 'change_points': n}
    ms = []
    with tempfile.TemporaryDirectory() as tmp:
        for k in range(2):
            t = time.perf_counter()
            n = _export(a.dsn, syn[-1 - k].isoformat(), tmp)
            ms.append((time.perf_counter() - t) * 1000)
    out['export'] = dict(summarize(ms), rows=n)
    out['targets_ms'] = TARGETS_MS
    out['pass'] = {k: out[k]['p95_ms'] < v for k, v in TARGETS_MS.items()}
    return out


def step_explain(a):
    """Which partitions each reader query touches (plan item 1)."""
    con = _connect(a.dsn)
    latest = _latest(con)
    syn = _syn_days(con)
    tid = con.execute('SELECT ticker_id FROM bench.syn ORDER BY i LIMIT 1').fetchone()[0]
    parts = {r[0]: (r[1], r[2]) for r in con.execute(
        "SELECT c.relname, pg_get_expr(c.relpartbound, c.oid), c.reltuples FROM pg_inherits i "
        "JOIN pg_class c ON c.oid = i.inhrelid WHERE i.inhparent = 'core.results'::regclass").fetchall()}

    def year_parts(lo, hi):
        return {f'results_{y}' for y in range(lo.year, hi.year + 1) if f'results_{y}' in parts}

    def plan(sql, params=()):
        return con.execute('EXPLAIN (FORMAT JSON) ' + sql, params).fetchone()[0]

    since5y = latest - dt.timedelta(days=5 * 365)
    lkr_dates = syn[-7:]
    cases = {
        'ticker_history_5y': (plan('SELECT run_date, ' + ', '.join(f'"{c}"' for c in HISTORY_COLS)
                                   + ' FROM core.results WHERE ticker_id = %s AND run_date BETWEEN %s AND %s '
                                   'ORDER BY run_date', (tid, since5y, latest)),
                              year_parts(since5y, latest)),
        'ticker_history_open_ended': (plan('SELECT * FROM core.results WHERE ticker_id = %s AND run_date >= %s',
                                           (tid, since5y)), set(parts)),
        'read_rows_one_day': (plan('SELECT * FROM core.results WHERE run_date = %s', (latest,)),
                              year_parts(latest, latest)),
        'last_known_rows_pick': (plan('SELECT DISTINCT ON (ticker_id) ticker_id, run_date FROM core.results '
                                      'WHERE run_date = ANY (%s) AND rating IS NOT NULL '
                                      'ORDER BY ticker_id, run_date DESC', (lkr_dates,)),
                                 year_parts(lkr_dates[0], lkr_dates[-1])),
    }
    out, ok = {}, True
    for name, (p, expect) in cases.items():
        got, removed = partitions_in_plan(p)
        good = got <= expect and bool(got)
        ok &= good
        out[name] = {'scanned': sorted(got), 'expected_subset_of': sorted(expect), 'runtime_removed': removed,
                     'pruned': good, 'nodes': sorted(node_types(p))}
    # The history read is served from the covering index alone.
    out['ticker_history_index_only'] = 'Index Only Scan' in out['ticker_history_5y']['nodes']
    ok &= out['ticker_history_index_only']
    out['partitions'] = {k: v[0] for k, v in sorted(parts.items())}
    out['all_pruned'] = ok
    return out


def step_size(a):
    con = _connect(a.dsn)
    rows = con.execute('SELECT count(*) FROM core.results').fetchone()[0]
    per = {r[0]: {'rows_est': int(r[1]), 'total_bytes': r[2]} for r in con.execute(
        "SELECT c.relname, c.reltuples, pg_total_relation_size(c.oid) FROM pg_inherits i "
        "JOIN pg_class c ON c.oid = i.inhrelid WHERE i.inhparent = 'core.results'::regclass "
        'ORDER BY 1').fetchall()}
    results_bytes = sum(v['total_bytes'] for v in per.values())
    other = {t: con.execute('SELECT pg_total_relation_size(%s)', (t,)).fetchone()[0]
             for t in ('core.rating_changes', 'core.edgar_blobs', 'core.tickers', 'core.latest_results')}
    per_row = results_bytes / max(rows, 1)
    year_rows = 8000 * 252
    return {'results_rows': rows, 'results_bytes': results_bytes, 'bytes_per_row': round(per_row),
            'partitions': per, 'other_bytes': other,
            'database_bytes': con.execute('SELECT pg_database_size(current_database())').fetchone()[0],
            'projected_gib_per_year_8k': round(per_row * year_rows / 2**30, 2),
            'projected_gib_20m_rows': round(per_row * 20e6 / 2**30, 1)}


def step_anon(a):
    """Unauthenticated Data API calls are refused (plan item 3)."""
    import requests
    out = {}
    for name, method, path, body in (
            ('table core.results', 'GET', '/rest/v1/results?limit=1', None),
            ('rpc read_rows', 'POST', '/rest/v1/rpc/read_rows', {'p_run_date': '2027-01-04'}),
            ('rpc publish_run', 'POST', '/rest/v1/rpc/publish_run',
             {'p_load_id': '00000000-0000-0000-0000-000000000000', 'p_run_date': '2027-01-04', 'p_run': {}, 'p_expect': {}})):
        for who, headers in (('no key', {}), ('anon key', {'apikey': _anon_key(),
                                                           'Authorization': f'Bearer {_anon_key()}'})):
            r = requests.request(method, a.api + path, json=body, timeout=10,
                                 headers={**headers, 'Accept-Profile': 'pipeline', 'Content-Profile': 'pipeline'})
            out[f'{name} / {who}'] = r.status_code
    out['all_refused'] = all(v in (401, 403, 404, 406) for v in out.values())
    return out


def _anon_key():
    import base64
    import hashlib
    import hmac
    secret = os.environ.get('SUPABASE_JWT_SECRET', 'super-secret-jwt-token-with-at-least-32-characters-long')

    def b64(b):
        return base64.urlsafe_b64encode(b).rstrip(b'=').decode()
    head = b64(json.dumps({'alg': 'HS256', 'typ': 'JWT'}).encode())
    body = b64(json.dumps({'role': 'anon', 'iss': 'supabase-demo', 'exp': int(time.time()) + 3600}).encode())
    sig = b64(hmac.new(secret.encode(), f'{head}.{body}'.encode(), hashlib.sha256).digest())
    return f'{head}.{body}.{sig}'


def _reader_proc(dsn, k, tickers, syn5, new_day, stop, path):
    """One reader process: loop over the reader queries until *stop*, then
    write ``[[t_start, query, ms], ...]`` to *path*. A process, not a thread,
    so the publisher's JSON work cannot hold the readers up on the GIL."""
    c = _connect(dsn)
    rnd = random.Random(k)
    out, err = [], None
    try:
        while not stop.is_set():
            which = rnd.choice(('ticker_history_5y', 'ticker_history_5y', 'rating_changes_since', 'last_known_rows'))
            t0, t = time.time(), time.perf_counter()
            if which == 'ticker_history_5y':
                _q_history(c, rnd.choice(tickers), new_day - dt.timedelta(days=5 * 365), new_day)
            elif which == 'rating_changes_since':
                _q_changes(c, syn5)
            else:
                _q_lkr(c, new_day + dt.timedelta(days=1))
            out.append([t0, which, (time.perf_counter() - t) * 1000])
    except Exception as e:
        err = repr(e)
    finally:
        c.close()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'lat': out, 'error': err}, f)


def _watcher_proc(dsn, new_day, stop, path):
    """Poll the new day's row count, raw and through the read RPC."""
    c = _connect(dsn)
    raw, rpc, err = set(), set(), None
    try:
        while not stop.is_set():
            raw.add(c.execute('SELECT count(*) FROM core.results WHERE run_date = %s', (new_day,)).fetchone()[0])
            rpc.add(c.execute("SELECT jsonb_array_length(pipeline.read_rows(%s, ARRAY['rating']))",
                              (new_day,)).fetchone()[0])
            time.sleep(0.05)
    except Exception as e:
        err = repr(e)
    finally:
        c.close()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'raw': sorted(raw), 'rpc': sorted(rpc), 'error': err}, f)


def step_load(a):
    """Publish a new day, then republish it, while reader processes run
    (plan item 4). Reader latency is split into inside and outside the
    publish windows."""
    import multiprocessing as mp
    import tempfile
    con = _connect(a.dsn)
    syn = _syn_days(con)
    tickers = [t for _, t in _syn_tickers(con)]
    con.close()
    run_date, load, build_s = _prepare_publish(a)        # before the readers start
    ctx = mp.get_context('fork')
    stop = ctx.Event()
    with tempfile.TemporaryDirectory() as tmp:
        procs = [ctx.Process(target=_reader_proc, args=(a.dsn, k, tickers, syn[-5], run_date, stop,
                                                         os.path.join(tmp, f'r{k}.json')))
                 for k in range(a.readers)]
        procs.append(ctx.Process(target=_watcher_proc, args=(a.dsn, run_date, stop, os.path.join(tmp, 'w.json'))))
        for p in procs:
            p.start()
        time.sleep(5)
        windows = []
        t = time.time()
        first = _publish_prepared(a, run_date, load, build_s)
        windows.append((t, time.time()))
        t = time.time()
        second = _publish_prepared(a, run_date, load, build_s, replace=True)
        windows.append((t, time.time()))
        time.sleep(5)
        stop.set()
        for p in procs:
            p.join(120)
        readers = [json.load(open(os.path.join(tmp, f'r{k}.json'), encoding='utf-8')) for k in range(a.readers)]
        watch = json.load(open(os.path.join(tmp, 'w.json'), encoding='utf-8'))
    inside, outside = {}, {}
    for r in readers:
        for t0, which, ms in r['lat']:
            during = any(lo <= t0 <= hi for lo, hi in windows)
            (inside if during else outside).setdefault(which, []).append(ms)
    n = a.tickers
    errors = [r['error'] for r in readers if r['error']] + ([watch['error']] if watch['error'] else [])
    return {'run_date': str(run_date), 'publish': first, 'republish': second,
            'row_counts_seen': watch['raw'], 'read_rows_counts_seen': watch['rpc'],
            'atomic': set(watch['raw']) <= {0, n} and set(watch['rpc']) <= {0, n},
            'reader_latency_during_publish': {k: summarize(v) for k, v in inside.items()},
            'reader_latency_otherwise': {k: summarize(v) for k, v in outside.items()},
            'reader_errors': errors[:5], 'readers': a.readers}


GROWTH_METRICS = (
    ('publish (s)', ('publish', 'total_s')),
    ('ticker history 5y p95 (ms)', ('bench', 'ticker_history_5y', 'p95_ms')),
    ('ticker history rows', ('bench', 'ticker_history_5y', 'rows')),
    ('ticker history 5y, all columns p95 (ms)', ('bench', 'ticker_history_5y_all_columns', 'p95_ms')),
    ('rating changes since p95 (ms)', ('bench', 'rating_changes_since', 'p95_ms')),
    ('last_known_rows p95 (ms)', ('bench', 'last_known_rows', 'p95_ms')),
    ('last_known_rows, all columns p95 (ms)', ('bench', 'last_known_rows_all_columns', 'p95_ms')),
    ('rating_history(all) (ms)', ('bench', 'rating_history_all', 'ms')),
    ('export p95 (ms)', ('bench', 'export', 'p95_ms')),
    ('bytes per row', ('size', 'bytes_per_row')),
)


def _dig(d, path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def compare(paths):
    """Markdown table of the growth metrics across reports (plan item 5)."""
    reps = [json.load(open(p, encoding='utf-8')) for p in paths]
    rows = [_dig(r, ('size', 'results_rows')) for r in reps]
    head = '| metric | ' + ' | '.join(f'{n:,} rows' for n in rows) + ' | growth |'
    lines = [head, '|' + '---|' * (len(reps) + 2)]
    for label, path in GROWTH_METRICS:
        vals = [_dig(r, path) for r in reps]
        g = (f'{vals[-1] / vals[0]:.2f}x' if all(isinstance(v, (int, float)) for v in vals) and vals[0]
             else '')
        lines.append(f'| {label} | ' + ' | '.join('' if v is None else f'{v:,}' for v in vals) + f' | {g} |')
    lines.append('| (rows) | ' + ' | '.join(f'{n:,}' for n in rows) + f' | {rows[-1] / rows[0]:.2f}x |')
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('step', choices=['seed', 'generate', 'publish', 'explain', 'bench', 'load', 'size', 'anon', 'all',
                                     'compare'])
    ap.add_argument('--dsn', default=os.environ.get('SCALE_DSN', DEFAULT_DSN))
    ap.add_argument('--api', default=os.environ.get('SCALE_API', DEFAULT_API))
    ap.add_argument('--container', default=DB_CONTAINER, help='Postgres container restarted for a cold cache')
    ap.add_argument('--template', default='output/results_2026-09-25.json.gz')
    ap.add_argument('--tickers', type=int, default=8000)
    ap.add_argument('--days', type=int, default=188)
    ap.add_argument('--samples', type=int, default=100)
    ap.add_argument('--readers', type=int, default=4)
    ap.add_argument('--direct', action='store_true', help='publish over a direct connection, not the Data API')
    ap.add_argument('--no-restart', action='store_true', help='bench without restarting Postgres (warm cache)')
    ap.add_argument('--out', help='write the JSON report here')
    ap.add_argument('reports', nargs='*', help='compare: the JSON reports, smallest first')
    a = ap.parse_args(argv)
    if a.step == 'compare':
        print(compare(a.reports))
        return 0
    steps = {'seed': step_seed, 'generate': step_generate, 'publish': step_publish, 'explain': step_explain,
             'bench': step_bench, 'load': step_load, 'size': step_size, 'anon': step_anon}
    order = ['seed', 'generate', 'publish', 'explain', 'size', 'bench', 'load', 'anon'] if a.step == 'all' else [a.step]
    report = {'tickers': a.tickers, 'days': a.days}
    for s in order:
        t = time.time()
        print(f'== {s}', file=sys.stderr, flush=True)
        report[s] = steps[s](a)
        report[s]['step_seconds'] = round(time.time() - t, 1)
        print(json.dumps(report[s], indent=2, default=str), flush=True)
    if a.out:
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
    return 0


if __name__ == '__main__':
    sys.exit(main())
