"""Parquet exports, Storage uploads and the database-backed screen-skip cache (P4b), offline."""
import gzip
import hashlib
import json
import math

import pytest

from data.db import parquet as pqx
from data.db import reader
from data.db import storage as st
from data.db.codec import rows_equivalent, values_equal
from data.db.columns import COLUMNS
from data.snapshot_store import DEFAULT_PROJECTIONS

NUM = next(k for k, t in COLUMNS.items() if t == 'double precision' and k != 'mos')
BIG = next(k for k, t in COLUMNS.items() if t == 'bigint')


def _snapshot():
    rows = [
        {'ticker': 'BBB', 'rating': 'BUY', 'mos': 0.03932028370017462, NUM: math.nan, BIG: 2 ** 40,
         'p_list': [1, math.inf, 'a\x00b'], 'p_tag': {'$nf': 'x'}, 'pe': 'Infinity',
         'news_headlines': ['report only'],
         'edgar_history': {'years_available': 9, 'total_debt_history': {'2025': 0.0}, 'capex_history': {'2025': 1}}},
        {'ticker': 'AAA', 'rating': None, NUM: -math.inf},
        {'ticker': 'BBB', 'rating': 'HOLD', 'mos': 0.5, 'edgar_history': {'years_available': 10}},   # last one wins
    ]
    return {'date': '2031-11-03', 'risk_free_rate': 0.0425, 'provenance': {'t': math.nan}, 'results': rows}


def test_parquet_round_trip(tmp_path):
    data = _snapshot()
    path = str(tmp_path / 'results_2031-11-03.parquet')
    n, sha = pqx.export_snapshot(data, '2031-11-03', path)
    assert n == 2 and len(sha) == 64
    with open(path, 'rb') as f:
        assert hashlib.sha256(f.read()).hexdigest() == sha
    snap = pqx.read_snapshot_parquet(path)
    assert snap['date'] == '2031-11-03' and snap['risk_free_rate'] == 0.0425
    assert math.isnan(snap['provenance']['t'])
    rows = {r['ticker']: r for r in snap['results']}
    assert [r['ticker'] for r in snap['results']] == ['AAA', 'BBB']
    assert rows['AAA'][NUM] == -math.inf and 'rating' not in rows['AAA']
    assert rows['BBB']['rating'] == 'HOLD' and rows['BBB']['edgar_history'] == {'years_available': 10}


def test_parquet_rows_match_the_source_under_the_codec_contract(tmp_path):
    data = _snapshot()
    data['results'] = data['results'][:2]
    path = str(tmp_path / 'x.parquet')
    pqx.export_snapshot(data, '2031-11-03', path)
    got = {r['ticker']: r for r in pqx.read_snapshot_parquet(path)['results']}
    for src in data['results']:
        want = dict(src)
        if 'edgar_history' in want:
            want['edgar_history'] = {k: v for k, v in want['edgar_history'].items()
                                     if k in DEFAULT_PROJECTIONS['edgar_history']}
        assert rows_equivalent(want, got[src['ticker']]) == []
    assert 'capex_history' not in got['BBB']['edgar_history']
    assert got['BBB']['edgar_history']['total_debt_history'] == {'2025': 0.0}


def test_a_foreign_parquet_is_rejected(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = str(tmp_path / 'other.parquet')
    pq.write_table(pa.table({'a': [1]}), path)
    with pytest.raises(ValueError, match='is not a'):
        pqx.read_snapshot_parquet(path)


def test_export_dir_skips_existing(tmp_path):
    from data.snapshot_store import write_snapshot_file
    write_snapshot_file(str(tmp_path / 'results_2031-11-03.json'), _snapshot())
    assert pqx.export_dir(str(tmp_path), str(tmp_path / 'pq')) == [('2031-11-03', 2)]
    assert pqx.export_dir(str(tmp_path), str(tmp_path / 'pq')) == []
    assert pqx.export_dir(str(tmp_path), str(tmp_path / 'pq'), replace=True) == [('2031-11-03', 2)]


def test_snapshot_gzip_is_deterministic_and_canonical():
    from data.db.publish import canonical_sha256
    data = {'date': '2031-11-03', 'results': [{'ticker': 'A', 'v': 1.5}]}
    a, b = st.snapshot_gzip(data), st.snapshot_gzip(data)
    assert a == b
    assert hashlib.sha256(gzip.decompress(a)).hexdigest() == canonical_sha256(data)


class Resp:
    def __init__(self, status, content=b'', text=''):
        self.status_code, self.content, self.text = status, content, text or content.decode('latin-1')


class FakeSession:
    def __init__(self, bucket_exists=False, corrupt=False):
        self.objects, self.buckets, self.calls, self.corrupt = {}, set(), [], corrupt
        if bucket_exists:
            self.buckets.add('snapshots')

    def request(self, method, url, headers=None, timeout=None, data=None, json=None):
        path = url.split('/storage/v1', 1)[1]
        self.calls.append((method, path, headers))
        if method == 'GET' and path.startswith('/bucket/'):
            return Resp(200 if path.split('/')[-1] in self.buckets else 400, text='not found')
        if method == 'POST' and path == '/bucket':
            self.buckets.add(json['id'])
            return Resp(200)
        if method == 'POST' and path.startswith('/object/'):
            assert headers['x-upsert'] == 'true'
            self.objects[path[len('/object/'):]] = data + (b'!' if self.corrupt else b'')
            return Resp(200)
        if method == 'GET' and path.startswith('/object/'):
            key = path[len('/object/'):]
            return Resp(200, self.objects[key]) if key in self.objects else Resp(404, text='missing')
        return Resp(500)


class RecordingTransport:
    def __init__(self):
        self.calls = []

    def call(self, fn, args, idempotent=False):
        self.calls.append((fn, args))
        return {'run_date': args.get('p_run_date')}


def test_upload_run_uploads_verifies_and_records(tmp_path):
    session = FakeSession()
    client = st.StorageClient('https://x.supabase.co', 'svc', session=session)
    t = RecordingTransport()
    data = _snapshot()
    manifest = st.upload_run(data, '2031-11-03', client, t, str(tmp_path / 'pq'), n_rows=2)
    assert 'snapshots' in session.buckets
    assert set(session.objects) == {'snapshots/json/results_2031-11-03.json.gz',
                                    'snapshots/parquet/results_2031-11-03.parquet'}
    assert session.objects['snapshots/json/results_2031-11-03.json.gz'] == st.snapshot_gzip(data)
    assert all(h['Authorization'] == 'Bearer svc' for _, _, h in session.calls)
    fn, args = t.calls[-1]
    assert fn == 'record_snapshot_objects' and args['p_n_rows'] == 2
    assert args['p_json_sha256'] == hashlib.sha256(st.snapshot_gzip(data)).hexdigest()
    assert manifest['json_path'] == 'json/results_2031-11-03.json.gz'


def test_upload_run_refuses_a_corrupted_copy(tmp_path):
    client = st.StorageClient('https://x.supabase.co', 'svc', session=FakeSession(bucket_exists=True, corrupt=True))
    t = RecordingTransport()
    with pytest.raises(st.StorageError, match='does not match'):
        st.upload_run(_snapshot(), '2031-11-03', client, t, str(tmp_path))
    assert t.calls == []                           # nothing recorded for a bad upload


# --- screen skip cache with the database backend ---------------------------

class SkipTransport:
    def __init__(self, remote=None, fail=False):
        self.remote, self.fail, self.saved = dict(remote or {}), fail, None

    def call(self, fn, args, idempotent=False):
        if self.fail:
            raise RuntimeError('database down')
        if fn == 'screen_skip_load':
            return self.remote
        if fn == 'screen_skip_replace':
            self.saved = json.loads(json.dumps(args['p_entries']))
            return len(self.saved)
        raise AssertionError(fn)


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setenv('SNAPSHOT_STORE_BACKEND', 'postgres')
    reader.reset_backend_state()
    yield monkeypatch
    reader.reset_backend_state()


def _cache(tmp_path, transport, monkeypatch, file_entries=None):
    from datetime import date

    from data import screen_skip_cache as ssc
    path = tmp_path / 'screen_skip.json'
    if file_entries is not None:
        path.write_text(json.dumps(file_entries), encoding='utf-8')
    import data.db.publish as pub
    monkeypatch.setattr(pub, 'transport_from_env', lambda *a: (transport, None, 'x'))
    return ssc.ScreenSkipCache(str(path), today=date(2031, 11, 5))


def test_screen_skip_merges_the_database_newest_wins(tmp_path, backend):
    t = SkipTransport({'AAA': {'kind': 'mcap', 'mcap': 1e6, 'date': '2031-11-04'},
                       'BBB': {'kind': 'dead', 'date': '2031-11-01'}})
    cache = _cache(tmp_path, t, backend, {'BBB': {'kind': 'dead', 'date': '2031-11-03'},
                                          'CCC': {'kind': 'dead', 'date': '2031-11-02'}})
    assert len(cache) == 3
    assert cache._entries['BBB']['date'] == '2031-11-03'     # the file's is newer
    assert cache.skip_reason('AAA', 300e6)
    cache.record_dead('DDD')
    cache.forget('CCC')
    cache.save()
    assert set(t.saved) == {'AAA', 'BBB', 'DDD'}
    assert set(json.loads((tmp_path / 'screen_skip.json').read_text(encoding='utf-8'))) == {'AAA', 'BBB', 'DDD'}


def test_screen_skip_survives_a_database_failure(tmp_path, backend, caplog):
    t = SkipTransport(fail=True)
    cache = _cache(tmp_path, t, backend, {'CCC': {'kind': 'dead', 'date': '2031-11-02'}})
    assert len(cache) == 1 and 'database load failed' in caplog.text
    cache.record_dead('DDD')
    cache.save()                                        # file still written, no second attempt
    assert set(json.loads((tmp_path / 'screen_skip.json').read_text(encoding='utf-8'))) == {'CCC', 'DDD'}


class FlakySaveTransport(SkipTransport):
    def __init__(self):
        super().__init__()
        self.saves = 0

    def call(self, fn, args, idempotent=False):
        if fn == 'screen_skip_replace':
            self.saves += 1
            raise RuntimeError('gateway timeout')
        return super().call(fn, args, idempotent)


def test_screen_skip_stops_saving_to_the_database_after_a_failure(tmp_path, backend, caplog):
    t = FlakySaveTransport()
    cache = _cache(tmp_path, t, backend)
    for tk in ('DDD', 'EEE', 'FFF'):                    # three flushes, as every 500 tickers
        cache.record_dead(tk)
        cache.save()
    assert t.saves == 1 and caplog.text.count('database save failed') == 1
    assert set(json.loads((tmp_path / 'screen_skip.json').read_text(encoding='utf-8'))) == {'DDD', 'EEE', 'FFF'}


def test_screen_skip_without_the_backend_never_calls_the_database(tmp_path, monkeypatch):
    monkeypatch.delenv('SNAPSHOT_STORE_BACKEND', raising=False)
    t = SkipTransport(fail=True)
    cache = _cache(tmp_path, t, monkeypatch)
    cache.record_dead('DDD')
    cache.save()
    assert values_equal(len(cache), 1)
