"""Storage manifest and screen-skip RPCs against a live database (P4b).

Runs only with ``TEST_DATABASE_URL``. The manifest test uses a November 2031
run and cleans up; the screen-skip test saves and restores the table.
"""
import os

import pytest

pytestmark = pytest.mark.pg

DSN = os.environ.get('TEST_DATABASE_URL')
if not DSN:
    pytest.skip('TEST_DATABASE_URL not set', allow_module_level=True)

psycopg = pytest.importorskip('psycopg')

from data.db.connect import connect  # noqa: E402
from data.db.publish import DirectTransport, build_load, publish  # noqa: E402

D = '2031-11-20'


@pytest.fixture
def con():
    c = connect(DSN, autocommit=True)
    yield c
    with c.transaction():
        c.execute('DELETE FROM core.snapshot_objects WHERE run_date = %s', (D,))
        c.execute('DELETE FROM core.rating_changes WHERE run_date = %s', (D,))
        c.execute("DELETE FROM core.latest_results WHERE ticker_id IN "
                  "(SELECT ticker_id FROM core.tickers WHERE ticker LIKE 'ZZO%')")
        c.execute('DELETE FROM core.results WHERE run_date = %s', (D,))
        c.execute('DELETE FROM core.runs WHERE run_date = %s', (D,))
        c.execute("DELETE FROM core.tickers WHERE ticker LIKE 'ZZO%'")
    c.close()


def test_record_snapshot_objects_upserts(con):
    t = DirectTransport(con)
    publish(build_load({'date': D, 'results': [{'ticker': 'ZZOA', 'rating': 'BUY'}]}, D), t, min_row_ratio=0)
    args = {'p_run_date': D, 'p_json_path': f'json/results_{D}.json.gz', 'p_json_sha256': 'a' * 64,
            'p_json_bytes': 123, 'p_parquet_path': f'parquet/results_{D}.parquet', 'p_parquet_sha256': 'b' * 64,
            'p_n_rows': 1}
    assert t.call('record_snapshot_objects', args)['json_path'] == args['p_json_path']
    t.call('record_snapshot_objects', dict(args, p_json_bytes=456))            # re-upload: replaced
    assert con.execute('SELECT json_bytes, n_rows FROM core.snapshot_objects WHERE run_date = %s',
                       (D,)).fetchone() == (456, 1)


def test_record_snapshot_objects_needs_a_published_run(con):
    with pytest.raises(Exception, match='foreign key|snapshot_objects'):
        DirectTransport(con).call('record_snapshot_objects', {
            'p_run_date': '2031-11-21', 'p_json_path': 'x', 'p_json_sha256': 'a' * 64, 'p_json_bytes': 1,
            'p_parquet_path': None, 'p_parquet_sha256': None, 'p_n_rows': 1})


def test_screen_skip_round_trip(con):
    t = DirectTransport(con)
    saved = t.call('screen_skip_load', {})
    try:
        entries = {'ZZOA': {'kind': 'mcap', 'mcap': 1234567.5, 'date': '2031-11-19'},
                   'ZZOB': {'kind': 'dead', 'date': '2031-11-18'}}
        assert t.call('screen_skip_replace', {'p_entries': entries}) == 2
        assert t.call('screen_skip_load', {}) == entries
        assert t.call('screen_skip_replace', {'p_entries': {}}) == 0
        assert t.call('screen_skip_load', {}) == {}
        with pytest.raises(Exception, match='kind'):
            t.call('screen_skip_replace', {'p_entries': {'ZZOC': {'kind': 'bogus', 'date': '2031-11-18'}}})
    finally:
        t.call('screen_skip_replace', {'p_entries': saved})


@pytest.mark.parametrize('fn', ['pipeline.record_snapshot_objects(date, text, text, bigint, text, text, bigint)',
                                'pipeline.screen_skip_load()', 'pipeline.screen_skip_replace(jsonb)'])
def test_only_the_pipeline_roles_may_call(con, fn):
    for role, allowed in (('anon', False), ('authenticated', False), ('service_role', True)):
        got = con.execute('SELECT has_function_privilege(%s, %s, %s)', (role, fn, 'EXECUTE')).fetchone()[0]
        assert got is allowed, (role, fn)
