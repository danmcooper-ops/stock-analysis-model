"""Publish payload, chunking, gates and transports (data/db/publish.py), offline."""
import json
import math
import socket
import time

import pytest

from data.db import publish as pub
from data.db.codec import decode, values_equal
from data.db.columns import COLUMNS
from data.snapshot_store import write_snapshot_file

NUM = next(k for k, t in COLUMNS.items() if t == 'double precision' and k != 'mos')


def _snapshot(rows, **meta):
    return {'date': '2031-11-03', 'risk_free_rate': 0.042, 'risk_free_rate_source': 'fred', 'count': len(rows),
            'provenance': {'timings': {'phase1': 1.5}}, **meta, 'results': rows}


def test_canonical_sha256_matches_the_plain_file(tmp_path):
    data = _snapshot([{'ticker': 'A', 'rating': 'BUY', NUM: 0.1}])
    path = tmp_path / 'results_2031-11-03.json'
    write_snapshot_file(str(path), data)
    import hashlib
    assert pub.canonical_sha256(data) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_load_payload():
    rows = [
        {'ticker': 'A', 'rating': 'BUY', 'mos': math.nan, NUM: math.inf, 'p2_new': [1, math.nan],
         'edgar_history': {'rev': [1.0]}, 'news_headlines': ['report only']},
        {'ticker': 'B', 'rating': 'PASS', 'mos': 0.03932028370017462, 'edgar_history': {'rev': [1.0]}},
        {'ticker': 'B', 'rating': 'HOLD'},                  # duplicate: last one wins
    ]
    load = pub.build_load(_snapshot(rows), '2031-11-03')
    by = {r['ticker']: r for r in load.rows}
    assert by['A']['cols']['mos'] == 'NaN' and by['A']['cols'][NUM] == 'Infinity'
    assert values_equal(decode(by['A']['extra']), {'p2_new': [1, math.nan]})
    assert 'news_headlines' not in json.dumps(by['A'])
    assert by['B']['cols'] == {'rating': 'HOLD'} and by['B']['edgar_history_sha'] is None
    assert load.stats['rows'] == 2 and load.stats['duplicate_rows'] == 1
    assert load.stats['blobs'] == 1                       # A's history; B's was replaced
    assert load.run['risk_free_rate'] == 0.042
    assert 'results' not in load.run['meta'] and 'date' not in load.run['meta']
    assert load.run['meta']['provenance'] == {'timings': {'phase1': 1.5}}
    json.dumps(load.rows, allow_nan=False)                # strict JSON on the wire


def test_cast_failures_are_counted_and_kept():
    load = pub.build_load(_snapshot([{'ticker': 'A', 'rating': 5}]), '2031-11-03')
    assert load.stats['cast_failures'] == 1
    assert load.stats['cast_failures_by_key'] == {'rating': 1}
    assert load.rows[0]['extra'] == {'rating': 5}


def test_chunks_split_by_size_and_send_each_blob_once():
    hist = {'rev': list(range(50))}
    rows = [{'ticker': f'T{i}', 'rating': 'BUY', NUM: i / 7, 'edgar_history': hist if i % 2 else {'x': i}}
            for i in range(40)]
    load = pub.build_load(_snapshot(rows), '2031-11-03')
    chunks = load.chunks(max_bytes=2000)
    assert len(chunks) > 3
    assert [r['ticker'] for c in chunks for r in c[0]] == [r['ticker'] for r in load.rows]
    sent = [b['sha'] for c in chunks for b in c[1]]
    assert len(sent) == len(set(sent)) == load.stats['blobs']
    for rows_, blobs in chunks:                       # a blob rides with its first row
        shas = {r['edgar_history_sha'] for r in rows_}
        assert {b['sha'] for b in blobs} <= shas


class FakeTransport:
    def __init__(self):
        self.calls = []

    def call(self, fn, args, idempotent=False):
        self.calls.append((fn, args, idempotent))
        return {'rows': args.get('p_expect', {}).get('n_rows')} if fn == 'publish_run' else {'staged': True}


def test_publish_stages_then_publishes():
    load = pub.build_load(_snapshot([{'ticker': f'T{i}', 'rating': 'BUY'} for i in range(30)]), '2031-11-03')
    t = FakeTransport()
    result = pub.publish(load, t, chunk_bytes=300, pipeline_version='abc123')
    stages = [c for c in t.calls if c[0] == 'stage_chunk']
    assert stages and all(c[2] for c in stages)                      # staging is retried
    assert [c[1]['p_chunk_no'] for c in stages] == list(range(len(stages)))
    fn, args, idem = t.calls[-1]
    assert fn == 'publish_run' and not idem                            # publishing is not
    assert args['p_expect']['n_chunks'] == len(stages) and args['p_expect']['n_rows'] == 30
    assert args['p_expect']['min_row_ratio'] == pub.MIN_ROW_RATIO
    assert args['p_run']['pipeline_version'] == 'abc123'
    assert {c[1]['p_load_id'] for c in t.calls} == {result['load_id']}


def test_publish_gates():
    bad = pub.build_load(_snapshot([{'ticker': f'T{i}', 'rating': 1} for i in range(3)]), '2031-11-03')
    with pytest.raises(pub.PublishError, match='do not fit'):
        pub.publish(bad, FakeTransport())
    with pytest.raises(pub.PublishError, match='reason'):
        pub.publish(bad, FakeTransport(), force=True)
    t = FakeTransport()
    pub.publish(bad, t, force=True, reason='test')
    assert t.calls[-1][1]['p_expect']['force'] is True
    assert t.calls[-1][1]['p_expect']['reason'] == 'test'


class FakeResponse:
    def __init__(self, status, body=None):
        self.status_code = status
        self._body = body if body is not None else {}
        self.text = json.dumps(self._body)

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.posts = []

    def post(self, url, data=None, headers=None, timeout=None):
        self.posts.append((url, json.loads(data), headers, timeout))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _rest(responses):
    session = FakeSession(responses)
    return pub.RestTransport('https://x.supabase.co/', 'svc-key', session=session), session


def test_rest_request_shape(monkeypatch):
    t, s = _rest([FakeResponse(200, {'ok': 1})])
    assert t.call('stage_chunk', {'p_rows': [{'v': 1.5}]}) == {'ok': 1}
    url, body, headers, timeout = s.posts[0]
    assert url == 'https://x.supabase.co/rest/v1/rpc/stage_chunk'
    assert body == {'p_rows': [{'v': 1.5}]}
    assert headers['Content-Profile'] == 'pipeline' and headers['Authorization'] == 'Bearer svc-key'
    assert headers['apikey'] == 'svc-key' and timeout == (5, 300)


def test_rest_retries_only_idempotent_calls(monkeypatch):
    import requests
    monkeypatch.setattr(pub.time, 'sleep', lambda s: None)
    t, s = _rest([FakeResponse(503), requests.ConnectionError('reset'), FakeResponse(200, {'staged': True})])
    assert t.call('stage_chunk', {}, idempotent=True) == {'staged': True}
    assert len(s.posts) == 3
    t, s = _rest([FakeResponse(503, {'message': 'busy'})])
    with pytest.raises(pub.PublishError, match='HTTP 503'):
        t.call('publish_run', {})
    assert len(s.posts) == 1
    t, s = _rest([FakeResponse(400, {'message': 'publish_run: 3 rows staged'})])
    with pytest.raises(pub.PublishError, match='3 rows staged'):
        t.call('stage_chunk', {}, idempotent=True)
    assert len(s.posts) == 1                                  # a 4xx is never retried


def test_rest_marks_lost_responses():
    import requests
    for resp in (FakeResponse(504, {'message': 'The upstream server is timing out'}),
                 requests.ReadTimeout('read timed out')):
        t, _ = _rest([resp])
        with pytest.raises(pub.PublishOutcomeUnknown):
            t.call('publish_run', {})
    t, _ = _rest([FakeResponse(500, {'code': '57014', 'message': 'canceling statement due to statement timeout'})])
    with pytest.raises(pub.PublishError) as e:               # Postgres answered: a definite failure
        t.call('publish_run', {})
    assert not isinstance(e.value, pub.PublishOutcomeUnknown)


class LostResponseTransport(FakeTransport):
    """publish_run's response is lost; publish_outcome answers from *states*."""

    def __init__(self, states):
        super().__init__()
        self.states = list(states)

    def call(self, fn, args, idempotent=False):
        self.calls.append((fn, args, idempotent))
        if fn == 'publish_run':
            raise pub.PublishOutcomeUnknown('publish_run: HTTP 504: upstream timing out')
        if fn == 'publish_outcome':
            st = self.states.pop(0)
            if isinstance(st, Exception):
                raise st
            return st
        return {'staged': True}


def _lost(states, **kw):
    load = pub.build_load(_snapshot([{'ticker': f'T{i}', 'rating': 'BUY'} for i in range(5)]), '2031-11-03')
    t = LostResponseTransport(states)
    return t, pub.publish(load, t, outcome_poll_s=0, **kw)


def test_a_lost_response_that_committed_is_a_success(monkeypatch):
    monkeypatch.setattr(pub.time, 'sleep', lambda s: None)
    t, result = _lost([{'state': 'running'}, pub.PublishError('publish_outcome: HTTP 502'),
                       {'state': 'published', 'rows': 5, 'publish': {'warnings': ['w']}}])
    assert result['rows'] == 5 and result['warnings'] == ['w'] and 'HTTP 504' in result['confirmed_after_lost_response']
    asks = [c for c in t.calls if c[0] == 'publish_outcome']
    assert len(asks) == 3 and all(c[2] for c in asks)                   # asking is idempotent
    assert asks[0][1] == {'p_load_id': result['load_id'], 'p_run_date': '2031-11-03'}


@pytest.mark.parametrize('state', ['failed', 'unknown'])
def test_a_lost_response_that_rolled_back_fails(monkeypatch, state):
    monkeypatch.setattr(pub.time, 'sleep', lambda s: None)
    with pytest.raises(pub.PublishError, match=f'publish_run {state} after a lost response'):
        _lost([{'state': 'running'}, {'state': state}])


def test_a_lost_response_gives_up_at_the_deadline(monkeypatch):
    monkeypatch.setattr(pub.time, 'sleep', lambda s: None)
    clock = iter(range(1000))
    monkeypatch.setattr(pub.time, 'time', lambda: next(clock))
    with pytest.raises(pub.PublishError, match='still running after 3s'):
        _lost([{'state': 'running'}] * 50, outcome_wait_s=3)


def test_connect_times_out_on_a_silent_server():
    """Plan R10: an unresponsive database costs seconds, not minutes."""
    pytest.importorskip('psycopg')
    import psycopg
    from data.db.connect import connect
    srv = socket.socket()
    srv.bind(('127.0.0.1', 0))
    srv.listen(1)                        # accepts TCP, never speaks the protocol
    port = srv.getsockname()[1]
    t0 = time.time()
    try:
        with pytest.raises(psycopg.OperationalError):
            connect(f'postgresql://u:p@127.0.0.1:{port}/db', connect_timeout=2)
    finally:
        srv.close()
    assert time.time() - t0 < 10
