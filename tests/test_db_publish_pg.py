"""The publish RPCs against a live, migrated Postgres (P2 stability checks 3-6).

Runs only with ``TEST_DATABASE_URL`` (CI's ``db`` job). Test data uses dates in
November 2031 and tickers starting ``ZZT`` and is deleted afterwards, so the
suite is safe on a database that also holds real runs.
"""
import math
import os
import threading
import time
import uuid

import pytest

pytestmark = pytest.mark.pg

DSN = os.environ.get('TEST_DATABASE_URL')
if not DSN:
    pytest.skip('TEST_DATABASE_URL not set', allow_module_level=True)

psycopg = pytest.importorskip('psycopg')

from data.db import publish as pub  # noqa: E402
from data.db.codec import join_row, rows_equivalent  # noqa: E402
from data.db.columns import COLUMNS  # noqa: E402
from data.db.connect import connect  # noqa: E402

D1, D2, D3 = '2031-11-03', '2031-11-04', '2031-11-05'
NUM = next(k for k, t in COLUMNS.items() if t == 'double precision' and k != 'mos')
COLS = list(COLUMNS)


def _cleanup(con):
    with con.transaction():
        con.execute("DELETE FROM core.rating_changes WHERE run_date >= '2031-11-01' AND run_date < '2031-12-01'")
        con.execute("DELETE FROM core.latest_results WHERE ticker_id IN "
                    "(SELECT ticker_id FROM core.tickers WHERE ticker LIKE 'ZZT%')")
        con.execute("DELETE FROM core.results WHERE run_date >= '2031-11-01' AND run_date < '2031-12-01'")
        con.execute("DELETE FROM core.runs WHERE run_date >= '2031-11-01' AND run_date < '2031-12-01'")
        con.execute("DELETE FROM core.tickers WHERE ticker LIKE 'ZZT%'")
        con.execute("DELETE FROM core.edgar_blobs b WHERE value ? 'zzt_marker' "
                    'AND NOT EXISTS (SELECT 1 FROM core.results r WHERE r.edgar_history_sha = b.sha)')
        con.execute("DELETE FROM internal.load_rows WHERE ticker LIKE 'ZZT%'")


@pytest.fixture
def con():
    c = connect(DSN, autocommit=True)
    _cleanup(c)
    yield c
    _cleanup(c)
    c.close()


def _row(ticker, rating, **kw):
    return {'ticker': ticker, 'rating': rating, 'mos': 0.03932028370017462, NUM: math.nan,
            'edgar_history': {'zzt_marker': ticker, 'rev': [1.5, None]}, **kw}


def _publish(con, run_date, rows, **kw):
    kw.setdefault('min_row_ratio', 0)        # tests control their own baselines
    data = {'date': run_date, 'risk_free_rate': 0.04, 'results': rows}
    return pub.publish(pub.build_load(data, run_date), pub.DirectTransport(con), chunk_bytes=2000, **kw)


def _changes(con):
    return con.execute(
        "SELECT t.ticker, rc.run_date::text, rc.rating, rc.prev_rating FROM core.rating_changes rc "
        "JOIN core.tickers t USING (ticker_id) WHERE t.ticker LIKE 'ZZT%' ORDER BY 1, 2").fetchall()


def _full_recompute(con):
    """What rating_changes must equal: the DuckDB store's rule over complete runs."""
    return con.execute("""
        SELECT ticker, run_date::text, rating, prev FROM (
          SELECT t.ticker, r.run_date, r.rating,
                 lag(r.rating) OVER (PARTITION BY r.ticker_id ORDER BY r.run_date) AS prev
            FROM core.results r JOIN core.tickers t USING (ticker_id)
            JOIN core.runs ru ON ru.run_date = r.run_date AND ru.status = 'complete'
           WHERE t.ticker LIKE 'ZZT%' AND r.rating IS NOT NULL AND r.rating <> '') s
        WHERE prev IS NULL OR prev <> rating ORDER BY 1, 2""").fetchall()


def _latest(con):
    return dict(con.execute(
        "SELECT t.ticker, l.run_date::text FROM core.latest_results l JOIN core.tickers t USING (ticker_id) "
        "WHERE t.ticker LIKE 'ZZT%'").fetchall())


def _day_digest(con, run_date):
    return con.execute(
        "SELECT md5(string_agg(r::text, '|' ORDER BY r.ticker_id)) FROM core.results r WHERE run_date = %s",
        (run_date,)).fetchone()[0]


def test_publish_round_trips_rows(con):
    rows = [_row('ZZTA', 'BUY', p2_list=[1, math.inf, 'a\x00b'], p2_tag={'$nf': 'NaN'}),
            _row('ZZTB', 'PASS', **{NUM: -math.inf}), {'ticker': 'ZZTC', 'rating': 'HOLD', 'mos': 'Infinity'}]
    result = _publish(con, D1, rows)
    assert result['rows'] == 3 and result['new_tickers'] == 3 and result['rating_changes'] == 3
    select = ', '.join(['t.ticker', *(f'r."{c}"' for c in COLS), 'r.extra', 'b.value'])
    got = con.execute(
        f'SELECT {select} FROM core.results r JOIN core.tickers t USING (ticker_id) '
        'LEFT JOIN core.edgar_blobs b ON b.sha = r.edgar_history_sha WHERE r.run_date = %s ORDER BY t.ticker',
        (D1,)).fetchall()
    for src, rec in zip(rows, got, strict=True):
        assert rows_equivalent(src, join_row(rec[0], COLS, rec[1:1 + len(COLS)], rec[-2], rec[-1])) == []
    status, sha, meta = con.execute(
        "SELECT status::text, source_sha256, meta FROM core.runs WHERE run_date = %s", (D1,)).fetchone()
    assert status == 'complete' and len(sha) == 64
    assert meta['_publish']['force'] is False
    assert _latest(con) == {'ZZTA': D1, 'ZZTB': D1, 'ZZTC': D1}
    assert con.execute('SELECT count(*) FROM internal.load_rows WHERE ticker LIKE %s', ('ZZT%',)).fetchone()[0] == 0


def test_republishing_a_day_is_idempotent(con):
    rows = [_row('ZZTA', 'BUY'), _row('ZZTB', 'PASS')]
    _publish(con, D1, rows)
    before, changes = _day_digest(con, D1), _changes(con)
    result = _publish(con, D1, rows)
    assert result['replaced_rows'] == 2 and result['new_tickers'] == 0 and result['new_blobs'] == 0
    assert _day_digest(con, D1) == before
    assert _changes(con) == changes


def test_out_of_order_publish_and_repair_match_a_full_recompute(con):
    """Stability check 4."""
    _publish(con, D1, [_row('ZZTA', 'BUY'), _row('ZZTB', 'PASS')])
    _publish(con, D3, [_row('ZZTA', 'HOLD'), _row('ZZTB', 'PASS'), _row('ZZTC', 'BUY')])
    assert _changes(con) == _full_recompute(con)
    _publish(con, D2, [_row('ZZTA', 'HOLD'), _row('ZZTB', 'BUY')])           # the missed day, late
    assert _changes(con) == _full_recompute(con)
    _publish(con, D2, [_row('ZZTA', 'BUY'), _row('ZZTB', 'BUY')])            # repaired again
    assert _changes(con) == _full_recompute(con)
    assert ('ZZTA', D3, 'HOLD', 'BUY') in _changes(con)
    assert _latest(con) == {'ZZTA': D3, 'ZZTB': D3, 'ZZTC': D3}              # never moved back


def test_ticker_dropped_on_republish_falls_back(con):
    _publish(con, D1, [_row('ZZTA', 'BUY'), _row('ZZTB', 'PASS')])
    _publish(con, D2, [_row('ZZTA', 'BUY'), _row('ZZTB', 'HOLD')])
    _publish(con, D2, [_row('ZZTA', 'BUY')])                                  # ZZTB gone from D2
    assert _latest(con) == {'ZZTA': D2, 'ZZTB': D1}
    assert _changes(con) == _full_recompute(con)
    _publish(con, D3, [_row('ZZTC', 'BUY')])
    _publish(con, D3, [_row('ZZTD', 'BUY')])                                  # ZZTC has no other day
    assert 'ZZTC' not in _latest(con)


def test_runs_that_are_not_complete_are_skipped_by_the_probes(con):
    """The anchor and fallback probes exclude non-complete runs by date
    rather than by joining core.runs (P5); a failed day must still be
    invisible to both."""
    _publish(con, D1, [_row('ZZTA', 'BUY'), _row('ZZTB', 'PASS')])
    _publish(con, D2, [_row('ZZTA', 'HOLD'), _row('ZZTB', 'PASS')])
    with con.transaction():                                 # retire D2 as an admin would
        con.execute("UPDATE core.runs SET status = 'failed' WHERE run_date = %s", (D2,))
        con.execute('DELETE FROM core.rating_changes WHERE run_date = %s', (D2,))
    _publish(con, D3, [_row('ZZTA', 'BUY'), _row('ZZTB', 'PASS')])
    assert _changes(con) == _full_recompute(con)
    assert ('ZZTA', D3, 'BUY', 'HOLD') not in _changes(con)         # D2's HOLD is not the anchor
    _publish(con, D3, [_row('ZZTA', 'BUY')])                            # ZZTB leaves D3
    assert _latest(con)['ZZTB'] == D1                                   # not the failed D2
    assert _changes(con) == _full_recompute(con)


def test_publish_outcome_states(con):
    """After a lost response the client asks publish_outcome (P5)."""
    t = pub.DirectTransport(con)

    def outcome(load_id, day=D1):
        return t.call('publish_outcome', {'p_load_id': load_id, 'p_run_date': day})['state']
    res = _publish(con, D1, [_row('ZZTA', 'BUY')])
    assert outcome(res['load_id']) == 'published'
    assert outcome(str(uuid.uuid4())) == 'unknown'                      # another load wrote D1
    load = pub.build_load({'date': D2, 'results': [_row('ZZTA', 'HOLD')]}, D2)
    staged = str(uuid.uuid4())
    for i, (rows, blobs) in enumerate(load.chunks(2000)):                # staged, never published
        t.call('stage_chunk', {'p_load_id': staged, 'p_chunk_no': i, 'p_rows': rows, 'p_blobs': blobs})
    assert outcome(staged, D2) == 'failed'
    other = connect(DSN, autocommit=True)
    try:
        with other.transaction():                                       # a publish_run in flight
            other.execute("SELECT pg_advisory_xact_lock(hashtext('core.publish_run'))")
            assert outcome(staged, D2) == 'running'
    finally:
        other.close()
    assert outcome(staged, D2) == 'failed'
    con.execute('DELETE FROM internal.load_rows WHERE load_id = %s', (staged,))
    con.execute('DELETE FROM internal.load_blobs WHERE load_id = %s', (staged,))
    con.execute('DELETE FROM internal.load_chunks WHERE load_id = %s', (staged,))


def test_ticker_history_rpc(con):
    t = pub.DirectTransport(con)
    _publish(con, D1, [_row('ZZTA', 'BUY', price=10.5), _row('ZZTB', 'PASS')])
    _publish(con, D2, [_row('ZZTA', 'HOLD', price=11.0)])
    _publish(con, D3, [_row('ZZTA', 'HOLD', price=12.0)])
    con.execute("UPDATE core.runs SET status = 'failed' WHERE run_date = %s", (D2,))

    def hist(tk, lo=D1, hi=D3):
        return t.call('ticker_history', {'p_ticker': tk, 'p_from': lo, 'p_to': hi})
    got = hist('ZZTA')
    assert [(d, r, p) for d, r, _, p, _, _ in got] == [(D1, 'BUY', 10.5), (D3, 'HOLD', 12.0)]   # D2 failed
    assert got[0][2] == 0.03932028370017462                                  # mos, full precision
    assert hist('ZZTA', D2, D2) == [] and hist('ZZTNONE') == []


def test_row_drop_is_refused_unless_forced_and_audited(con):
    """Stability check 5."""
    _publish(con, D1, [_row(f'ZZT{i:02d}', 'BUY') for i in range(10)])
    small = [_row('ZZT00', 'BUY'), _row('ZZT01', 'BUY')]
    with pytest.raises(pub.PublishError, match='previous complete run'):
        _publish(con, D2, small, min_row_ratio=0.7)
    assert con.execute('SELECT count(*) FROM core.runs WHERE run_date = %s', (D2,)).fetchone()[0] == 0
    _publish(con, D2, small, min_row_ratio=0.7, force=True, reason='universe shrank on purpose')
    meta = con.execute('SELECT meta FROM core.runs WHERE run_date = %s', (D2,)).fetchone()[0]
    assert meta['_publish']['force'] is True
    assert meta['_publish']['reason'] == 'universe shrank on purpose'
    assert meta['_publish']['warnings']


def _stage(con, rows, blobs=(), load_id=None, chunk_no=0):
    load_id = load_id or str(uuid.uuid4())
    pub.DirectTransport(con).call('stage_chunk', {'p_load_id': load_id, 'p_chunk_no': chunk_no,
                                                 'p_rows': list(rows), 'p_blobs': list(blobs)})
    return load_id


def _publish_run(con, load_id, n_rows, n_chunks=1):
    return pub.DirectTransport(con).call('publish_run', {
        'p_load_id': load_id, 'p_run_date': D1, 'p_run': {},
        'p_expect': {'n_rows': n_rows, 'n_chunks': n_chunks, 'min_row_ratio': 0}})


@pytest.mark.parametrize('rows,blobs,n_rows,message', [
    ([{'ticker': 'ZZTA', 'cols': {'rating': 'BUY'}}], [], 2, 'rows staged'),
    ([{'ticker': 'ZZTA', 'cols': {'no_such_column': 1}}], [], 1, 'not in core.results'),
    ([{'ticker': 'ZZTA', 'cols': {}, 'edgar_history_sha': 'a' * 64}], [], 1, 'never staged'),
])
def test_structural_errors_are_never_published(con, rows, blobs, n_rows, message):
    load_id = _stage(con, rows, blobs)
    with pytest.raises(pub.PublishError, match=message):
        _publish_run(con, load_id, n_rows)
    assert con.execute('SELECT count(*) FROM core.runs WHERE run_date = %s', (D1,)).fetchone()[0] == 0
    con.execute('DELETE FROM internal.load_chunks WHERE load_id = %s', (load_id,))


def test_stage_chunk_is_idempotent(con):
    rows = [{'ticker': 'ZZTA', 'cols': {'rating': 'BUY'}}]
    load_id = _stage(con, rows)
    again = pub.DirectTransport(con).call('stage_chunk', {'p_load_id': load_id, 'p_chunk_no': 0,
                                                         'p_rows': rows, 'p_blobs': []})
    assert again == {'staged': False, 'rows': 1}
    with pytest.raises(pub.PublishError, match='re-sent'):
        _stage(con, rows + [{'ticker': 'ZZTB', 'cols': {}}], load_id=load_id)
    with pytest.raises(pub.PublishError):                         # same ticker in another chunk
        _stage(con, rows, load_id=load_id, chunk_no=1)
    assert _publish_run(con, load_id, 1)['rows'] == 1


def test_abandoned_loads_are_purged(con):
    old = _stage(con, [{'ticker': 'ZZTA', 'cols': {}}])
    con.execute("UPDATE internal.load_chunks SET staged_at = now() - interval '2 days' WHERE load_id = %s", (old,))
    new = _stage(con, [{'ticker': 'ZZTB', 'cols': {}}])
    assert con.execute('SELECT count(*) FROM internal.load_rows WHERE load_id = %s', (old,)).fetchone()[0] == 0
    con.execute('DELETE FROM internal.load_rows WHERE load_id = %s', (new,))
    con.execute('DELETE FROM internal.load_chunks WHERE load_id = %s', (new,))


def test_concurrent_publishers_serialize(con):
    """Stability check 3: the second publisher waits instead of interleaving."""
    holder = connect(DSN, autocommit=True)
    holder.execute("SELECT pg_advisory_lock(hashtext('core.publish_run'))")
    errors, done = [], threading.Event()

    def run():
        try:
            with connect(DSN, autocommit=True) as c:
                _publish(c, D1, [_row('ZZTA', 'BUY')])
        except Exception as e:           # surfaced by the assertion below
            errors.append(e)
        finally:
            done.set()

    th = threading.Thread(target=run)
    th.start()
    try:
        assert not done.wait(2.0), 'publish did not wait for the running publisher'
    finally:
        holder.execute("SELECT pg_advisory_unlock(hashtext('core.publish_run'))")
        holder.close()
    th.join(30)
    assert errors == [] and done.is_set()
    assert _latest(con) == {'ZZTA': D1}


def test_killed_publish_leaves_the_previous_day(con):
    """Stability check 3: a publisher killed mid-transaction changes nothing."""
    _publish(con, D1, [_row('ZZTA', 'BUY'), _row('ZZTB', 'PASS')])
    before, changes, latest = _day_digest(con, D1), _changes(con), _latest(con)

    victim = connect(DSN, autocommit=True)
    pid = victim.info.backend_pid
    blocker = connect(DSN)
    blocker.execute('LOCK TABLE core.latest_results IN ACCESS EXCLUSIVE MODE')   # publish stalls late
    errors = []

    def run():
        try:
            _publish(victim, D1, [_row('ZZTA', 'HOLD')])
        except Exception as e:
            errors.append(e)

    th = threading.Thread(target=run)
    th.start()
    deadline = time.time() + 10
    while time.time() < deadline:        # wait until it is blocked inside publish_run
        waiting = con.execute("SELECT wait_event_type FROM pg_stat_activity WHERE pid = %s", (pid,)).fetchone()
        if waiting and waiting[0] == 'Lock':
            break
        time.sleep(0.05)
    con.execute('SELECT pg_terminate_backend(%s)', (pid,))
    th.join(10)
    blocker.rollback()
    blocker.close()
    assert errors, 'publish should have been killed'
    assert _day_digest(con, D1) == before
    assert _changes(con) == changes and _latest(con) == latest
    con.execute("DELETE FROM internal.load_chunks WHERE load_id NOT IN (SELECT DISTINCT load_id FROM internal.load_rows)")


@pytest.mark.parametrize('role', ['anon', 'authenticated'])
def test_public_roles_cannot_call_the_rpcs(con, role):
    with con.transaction(force_rollback=True):
        con.execute(f'SET LOCAL ROLE {role}')
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            con.execute("SELECT pipeline.stage_chunk(gen_random_uuid(), 0, '[]'::jsonb)")


def test_service_role_can_call_the_rpcs(con):
    for fn in ('pipeline.stage_chunk(uuid, integer, jsonb, jsonb)', 'pipeline.publish_run(uuid, date, jsonb, jsonb)'):
        assert con.execute("SELECT has_function_privilege('service_role', %s, 'EXECUTE')", (fn,)).fetchone()[0]


def test_list_runs_reports_published_dates(con):
    _publish(con, D1, [_row('ZZTA', 'BUY')])
    runs = {r['run_date']: r for r in pub.DirectTransport(con).call('list_runs', {})}
    assert runs[D1]['status'] == 'complete' and runs[D1]['n_rows'] == 1
    assert len(runs[D1]['source_sha256']) == 64
