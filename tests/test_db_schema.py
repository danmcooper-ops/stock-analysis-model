"""The migrated Supabase schema, checked against a live Postgres.

Runs only when ``TEST_DATABASE_URL`` points at a database with the migrations
in ``supabase/migrations`` applied (CI's ``db`` job: ``supabase db start``).
Everything is read-only or rolled back.
"""
import json
import math
import os

import pytest

pytestmark = pytest.mark.pg

DSN = os.environ.get('TEST_DATABASE_URL')
if not DSN:
    pytest.skip('TEST_DATABASE_URL not set', allow_module_level=True)

psycopg = pytest.importorskip('psycopg')

from data.db.codec import dumps, blob_sha, join_row, rows_equivalent, split_row  # noqa: E402
from data.db.columns import COLUMNS  # noqa: E402
from data.db.connect import connect  # noqa: E402

CORE_TABLES = {'tickers', 'runs', 'edgar_blobs', 'results', 'rating_changes', 'latest_results', 'night_checks',
               'screen_skip', 'snapshot_objects'}
PARTITIONS = {f'results_{y}' for y in range(2026, 2032)}


@pytest.fixture(scope='module')
def con():
    with connect(DSN) as c:
        yield c


def _tables(con, kinds=('r', 'p')):
    return {r[0]: r[1] for r in con.execute(
        "SELECT c.relname, c.oid FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname = 'core' AND c.relkind = ANY(%s)", (list(kinds),))}


def test_core_tables_exist(con):
    assert set(_tables(con)) == CORE_TABLES | PARTITIONS


def test_results_columns_match_registry(con):
    got = con.execute(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_schema = 'core' AND table_name = 'results' ORDER BY ordinal_position").fetchall()
    expected = [('run_date', 'date'), ('ticker_id', 'bigint'), *COLUMNS.items(),
                ('extra', 'jsonb'), ('edgar_history_sha', 'text')]
    assert got == expected


def test_yearly_partitions_and_no_default(con):
    rows = con.execute(
        "SELECT c.relname, pg_get_expr(c.relpartbound, c.oid) FROM pg_inherits i "
        "JOIN pg_class c ON c.oid = i.inhrelid WHERE i.inhparent = 'core.results'::regclass").fetchall()
    assert {r[0] for r in rows} == PARTITIONS
    assert not [r for r in rows if r[1] == 'DEFAULT']


def test_rls_enabled_everywhere(con):
    off = con.execute(
        "SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname = 'core' AND c.relkind IN ('r', 'p') AND NOT c.relrowsecurity").fetchall()
    assert off == []


@pytest.mark.parametrize('role', ['anon', 'authenticated'])
def test_public_roles_have_no_access(con, role):
    for schema in ('core', 'internal', 'pipeline'):
        assert not con.execute('SELECT has_schema_privilege(%s, %s, %s)', (role, schema, 'USAGE')).fetchone()[0]
    for name, oid in _tables(con).items():
        for priv in ('SELECT', 'INSERT', 'UPDATE', 'DELETE'):
            assert not con.execute('SELECT has_table_privilege(%s, %s, %s)', (role, oid, priv)).fetchone()[0], \
                f'{role} has {priv} on core.{name}'


def test_pipeline_schema_is_service_role_only(con):
    assert con.execute("SELECT has_schema_privilege('service_role', 'pipeline', 'USAGE')").fetchone()[0]
    assert con.execute("SELECT has_schema_privilege('pipeline_reader', 'core', 'USAGE')").fetchone()[0]


def test_public_roles_cannot_call_pipeline_functions(con):
    with con.transaction(force_rollback=True):
        con.execute('CREATE FUNCTION pipeline.p1_probe() RETURNS int LANGUAGE sql AS $$ SELECT 1 $$')
        for role in ('anon', 'authenticated'):
            with con.transaction(force_rollback=True):
                con.execute(f'SET LOCAL ROLE {role}')
                with pytest.raises(psycopg.errors.InsufficientPrivilege):
                    con.execute('SELECT pipeline.p1_probe()')


def test_every_private_function_revokes_public_execute(con):
    """Guard for P2+: new functions get EXECUTE for PUBLIC unless revoked."""
    open_fns = con.execute(
        "SELECT p.oid::regprocedure::text FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace "
        "WHERE n.nspname IN ('core', 'internal', 'pipeline') AND ("
        "  has_function_privilege('anon', p.oid, 'EXECUTE') OR "
        "  has_function_privilege('authenticated', p.oid, 'EXECUTE'))").fetchall()
    assert open_fns == []


def test_jsonb_columns_use_lz4(con):
    rows = con.execute(
        "SELECT c.relname, a.attname, a.attcompression FROM pg_attribute a "
        "JOIN pg_class c ON c.oid = a.attrelid JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname = 'core' AND a.atttypid = 'jsonb'::regtype AND a.attnum > 0 AND NOT a.attisdropped").fetchall()
    assert rows
    assert [r for r in rows if r[2] != 'l'] == []


def _sample_rows():
    num = next(k for k, t in COLUMNS.items() if t == 'double precision')
    txt = next(k for k, t in COLUMNS.items() if t == 'text')
    big = next((k for k, t in COLUMNS.items() if t == 'bigint'), None)
    rows = [
        # 17 significant digits: needs extra_float_digits > 0 to read back exactly
        {'ticker': 'AAA', num: 0.03932028370017462, txt: 'café', 'edgar_history': {'revenue_history': [1.0, None]},
         'p1_list': [1, math.inf, 'a\x00b'], 'p1_tag': {'$nf': 'NaN'}},
        {'ticker': 'BBB', num: -math.inf, txt: 'a\x00b', 'p1_new_key': 'Infinity', 'p1_nan': math.nan},
        {'ticker': 'CCC', num: 'Infinity', **({big: 2 ** 40} if big else {}),
         'edgar_history': {'revenue_history': [1.0, None]}},
        {'ticker': 'DDD', num: math.nan, 'p1_float': 85.37032414826739},
    ]
    return rows   # sorted by ticker, the order the test reads them back in


def test_codec_round_trip_through_postgres(con):
    rows = _sample_rows()
    cols = list(COLUMNS)
    with con.transaction(force_rollback=True):
        con.execute("INSERT INTO core.runs (run_date, status) VALUES ('2031-12-31', 'loading')")
        ids = {}
        for r in rows:
            ids[r['ticker']] = con.execute(
                "INSERT INTO core.tickers (ticker, first_seen, last_seen) VALUES (%s, '2031-12-31', '2031-12-31') "
                'RETURNING ticker_id', (r['ticker'],)).fetchone()[0]
        split = [split_row(r, COLUMNS) for r in rows]
        with con.cursor() as cur:
            for sr in split:
                if sr.blob:
                    body = dumps(sr.blob)
                    cur.execute('INSERT INTO core.edgar_blobs (sha, value) VALUES (%s, %s) ON CONFLICT DO NOTHING',
                                (blob_sha(body), body))
            names = ', '.join(['run_date', 'ticker_id', *[f'"{c}"' for c in cols], 'extra', 'edgar_history_sha'])
            with cur.copy(f'COPY core.results ({names}) FROM STDIN') as cp:
                for sr in split:
                    cp.write_row(['2031-12-31', ids[sr.ticker], *sr.typed,
                                  dumps(sr.extra) if sr.extra else None,
                                  blob_sha(dumps(sr.blob)) if sr.blob else None])
        select = ', '.join(['t.ticker', *[f'r."{c}"' for c in cols], 'r.extra', 'b.value'])
        got = con.execute(
            f'SELECT {select} FROM core.results r JOIN core.tickers t USING (ticker_id) '
            'LEFT JOIN core.edgar_blobs b ON b.sha = r.edgar_history_sha '
            "WHERE r.run_date = '2031-12-31' ORDER BY t.ticker").fetchall()
    assert len(got) == len(rows)
    for src, rec in zip(rows, got, strict=True):
        rebuilt = join_row(rec[0], cols, rec[1:1 + len(cols)], rec[-2], rec[-1])
        assert rows_equivalent(src, rebuilt) == [], json.dumps(src, default=str)


def test_service_role_outlives_the_api_statement_timeout(con):
    """publish_run is one transaction and takes ~13 s at 8k tickers; the Data
    API's authenticator default of 8 s would cancel it (P5)."""
    cfg = con.execute("SELECT rolconfig FROM pg_roles WHERE rolname = 'service_role'").fetchone()[0] or []
    assert 'statement_timeout=10min' in cfg
    anon = con.execute("SELECT rolconfig FROM pg_roles WHERE rolname = 'anon'").fetchone()[0] or []
    assert 'statement_timeout=3s' in anon                  # the public roles keep their short limits
