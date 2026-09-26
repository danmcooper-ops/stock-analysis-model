"""Database-backed snapshot store (data/db/reader.py), offline with fake transports."""
import hashlib
import json
import math

import pytest

from data.db import publish as pub
from data.db import reader
from data.db.columns import COLUMNS
from data.snapshot_store import SnapshotStore, sync_snapshot_file, write_snapshot_file

NUM = next(k for k, t in COLUMNS.items() if t == 'double precision' and k != 'mos')
BIG = next(k for k, t in COLUMNS.items() if t == 'bigint')


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for v in ('SNAPSHOT_STORE_BACKEND', 'SUPABASE_URL', 'SUPABASE_SERVICE_ROLE_KEY', 'SUPABASE_READER_URL',
              'SUPABASE_DB_URL', 'DB_DEFER_PUBLISH'):
        monkeypatch.delenv(v, raising=False)
    reader.reset_backend_state()
    yield
    reader.reset_backend_state()


class FakeTransport:
    def __init__(self, runs=(), rows=(), lkr=None, history=(), meta=None):
        self.runs, self.rows_, self.lkr, self.history, self.meta = list(runs), list(rows), lkr, list(history), meta
        self.calls = []

    def call(self, fn, args, idempotent=False):
        self.calls.append((fn, args))
        return {'list_runs': self.runs, 'read_rows': self.rows_, 'last_known_rows': self.lkr,
                'rating_history': self.history, 'run_meta': self.meta}[fn]


RUNS = [{'run_date': '2031-11-03', 'status': 'complete', 'source_sha256': 'a' * 64, 'n_rows': 2},
        {'run_date': '2031-11-04', 'status': 'complete', 'source_sha256': 'b' * 64, 'n_rows': 2},
        {'run_date': '2031-11-05', 'status': 'loading', 'source_sha256': None, 'n_rows': None}]


def test_not_selected_means_no_database(tmp_path):
    assert reader.open_db_store(str(tmp_path)) is None
    assert SnapshotStore.for_results_dir(str(tmp_path)) is None          # no DuckDB file either


def test_selected_without_configuration_fails_once(monkeypatch, caplog):
    monkeypatch.setenv('SNAPSHOT_STORE_BACKEND', 'postgres')
    calls = []
    real = pub.transport_from_env
    monkeypatch.setattr(pub, 'transport_from_env', lambda *a: calls.append(1) or real(*a))
    assert reader.open_db_store() is None
    assert reader.open_db_store() is None
    assert len(calls) == 1                     # the failure is remembered for the process (R10)
    assert 'unavailable' in caplog.text


def test_a_remote_database_is_refused_under_pytest(monkeypatch):
    monkeypatch.setenv('SUPABASE_URL', 'https://prod.supabase.co')
    monkeypatch.setenv('SUPABASE_SERVICE_ROLE_KEY', 'k')
    with pytest.raises(pub.PublishError, match='non-local'):
        pub.transport_from_env()
    monkeypatch.delenv('SUPABASE_URL')
    monkeypatch.setenv('SUPABASE_DB_URL', 'postgresql://u:p@db.example.com:5432/postgres')
    with pytest.raises(pub.PublishError, match='non-local'):
        pub.transport_from_env()


def test_dates_are_complete_runs_only():
    s = reader.DbStore(FakeTransport(RUNS))
    assert s.dates() == ['2031-11-03', '2031-11-04']
    assert s.dates(before='2031-11-04') == ['2031-11-03']
    assert s.has_date('2031-11-04') and not s.has_date('2031-11-05')
    assert s.latest_date(before='2031-11-05') == '2031-11-04'


def test_a_rewritten_file_is_not_served(tmp_path):
    body = b'{"results": []}'
    (tmp_path / 'results_2031-11-03.json').write_bytes(body)
    runs = [dict(RUNS[0], source_sha256=hashlib.sha256(body).hexdigest()), RUNS[1]]
    (tmp_path / 'results_2031-11-04.json').write_bytes(b'{"results": [1]}')     # differs from 'b' * 64
    s = reader.DbStore(FakeTransport(runs), results_dir=str(tmp_path))
    assert s.dates() == ['2031-11-03']
    assert not s.has_date('2031-11-04')


def test_rows_cast_project_and_decode():
    recs = [{'ticker': 'A', 'cols': {'rating': 'BUY', NUM: 'NaN', 'mos': 0.03932028370017462, BIG: 7},
             'extra': {'p_list': [1, {'$nf': 'Inf'}]}, 'blob': None},
            {'ticker': 'B', 'cols': {NUM: '-Infinity'}, 'extra': None, 'blob': {'edgar_history': {'rev': [1.5]}}}]
    t = FakeTransport(RUNS, rows=recs)
    s = reader.DbStore(t)
    rows = s.rows('2031-11-03', ['rating', NUM, 'p_list', 'no_such', 'edgar_history'])
    fn, args = t.calls[-1]
    assert fn == 'read_rows' and args['p_columns'] == ['rating', NUM]
    assert args['p_with_extra'] is True and args['p_with_blob'] is True
    a, b = rows
    assert list(a) == ['ticker', 'rating', NUM, 'p_list', 'no_such', 'edgar_history']
    assert math.isnan(a[NUM]) and a['p_list'] == [1, math.inf] and a['no_such'] is None
    assert b[NUM] == -math.inf and b['edgar_history'] == {'rev': [1.5]} and b['rating'] is None
    full = s.rows('2031-11-03')
    assert t.calls[-1][1]['p_columns'] is None
    assert full[0]['mos'] == 0.03932028370017462 and full[0][BIG] == 7 and isinstance(full[0][BIG], int)


def test_typed_only_requests_skip_extra_and_blob():
    t = FakeTransport(RUNS, rows=[])
    reader.DbStore(t).rows('2031-11-03', ['ticker', 'shares_out', 'mcap'])
    assert t.calls[-1][1] == {'p_run_date': '2031-11-03', 'p_columns': ['shares_out', 'mcap'],
                              'p_with_extra': False, 'p_with_blob': False}


def test_last_known_rows_shape():
    lkr = {'primary_date': '2031-11-04', 'rows': [
        {'date': '2031-11-04', 'ticker': 'A', 'cols': {'rating': 'BUY'}, 'extra': None},
        {'date': '2031-11-03', 'ticker': 'B', 'cols': {'rating': 'HOLD', 'mos': 0.5}, 'extra': None}]}
    s = reader.DbStore(FakeTransport(RUNS, lkr=lkr))
    primary, out, n_fallback = s.last_known_rows('2031-11-05', ['rating', 'mos'], max_lookback=2)
    assert primary == '2031-11-04' and n_fallback == 1
    assert out == {'A': {'rating': 'BUY', 'mos': None}, 'B': {'rating': 'HOLD', 'mos': 0.5}}
    with pytest.raises(NotImplementedError):
        s.last_known_rows('2031-11-05', ['edgar_history'])


def test_rating_history_and_run_meta():
    s = reader.DbStore(FakeTransport(RUNS, history=[['A', '2031-11-03', 'BUY'], ['A', '2031-11-04', 'HOLD']],
                                     meta={'status': 'complete', 'risk_free_rate': 0.042,
                                           'risk_free_rate_source': 'fred', 'n_rows': 2,
                                           'meta': {'count': 2, 'provenance': {'x': {'$nf': 'NaN'}}}}))
    assert s.rating_history(before='2031-11-05') == {'A': [['2031-11-03', 'BUY'], ['2031-11-04', 'HOLD']]}
    with pytest.raises(NotImplementedError):
        s.rating_history(column='rating_raw')
    meta = s.run_meta('2031-11-03')
    assert meta['date'] == '2031-11-03' and meta['risk_free_rate'] == 0.042 and meta['count'] == 2
    assert math.isnan(meta['provenance']['x'])
    with pytest.raises(NotImplementedError):
        s.query('SELECT 1')


def test_sync_republishes_unless_deferred(monkeypatch, tmp_path):
    path = str(tmp_path / 'results_2031-11-03.json')
    data = {'date': '2031-11-03', 'results': [{'ticker': 'A', 'rating': 'BUY'}]}
    write_snapshot_file(path, data)
    published = []
    monkeypatch.setattr(pub, 'publish_file', lambda p, d=None, **k: published.append(p) or {'rows': 1})
    sync_snapshot_file(path, data=data)
    assert published == []                                     # backend not selected
    monkeypatch.setenv('SNAPSHOT_STORE_BACKEND', 'postgres')
    monkeypatch.setenv('DB_DEFER_PUBLISH', '1')
    sync_snapshot_file(path, data=data)
    assert published == []                                     # nightly run: step 06a publishes
    monkeypatch.delenv('DB_DEFER_PUBLISH')
    assert sync_snapshot_file(path, data=data) is True
    assert published == [path]


def test_sync_survives_a_publish_failure(monkeypatch, tmp_path, caplog):
    path = str(tmp_path / 'results_2031-11-03.json')
    data = {'date': '2031-11-03', 'results': [{'ticker': 'A', 'rating': 'BUY'}]}
    write_snapshot_file(path, data)
    monkeypatch.setenv('SNAPSHOT_STORE_BACKEND', 'postgres')

    def boom(*a, **k):
        raise pub.PublishError('down')

    monkeypatch.setattr(pub, 'publish_file', boom)
    assert sync_snapshot_file(path, data=data) is True         # the DuckDB mirror still ran
    assert 'database publish failed' in caplog.text


class _Store:
    def __init__(self, dates, authoritative):
        self._dates, self.authoritative = dates, authoritative

    def __enter__(self):
        return self

    def __exit__(self, *e):
        pass

    def dates(self, before=None):
        return [d for d in self._dates if before is None or d < before]

    def rating_history(self, before=None):
        return {'A': [['2031-11-01', 'BUY']]}


@pytest.mark.parametrize('authoritative,expected', [(True, {'A': [['2031-11-01', 'BUY']]}), (False, None)])
def test_rating_history_accepts_a_superset_only_from_the_database(monkeypatch, authoritative, expected):
    """Plan item R3: the cloud stages ten files; the database holds them all."""
    from scripts import report_html
    store = _Store(['2031-10-30', '2031-10-31', '2031-11-01', '2031-11-02'], authoritative)
    monkeypatch.setattr(report_html, '_open_snapshot_store', lambda out_dir: store)
    assert report_html._rating_history_from_store('x', '2031-11-03', ['2031-11-01', '2031-11-02']) == expected


def test_check_database(tmp_path):
    from scripts.check_snapshot_store import check_database
    data = {'date': '2031-11-03', 'results': [{'ticker': 'A'}, {'ticker': 'B'}, {'ticker': 'B'}]}
    write_snapshot_file(str(tmp_path / 'results_2031-11-03.json'), data)
    good = {'run_date': '2031-11-03', 'status': 'complete', 'n_rows': 2, 'source_sha256': pub.canonical_sha256(data)}
    assert check_database(str(tmp_path), '2031-11-03', FakeTransport([good]))[0] == []
    bad = dict(good, n_rows=3, source_sha256='0' * 64)
    problems, _ = check_database(str(tmp_path), '2031-11-03', FakeTransport([bad]))
    assert len(problems) == 2 and 'changed since it was published' in problems[1]
    problems, _ = check_database(str(tmp_path), '2031-11-03', FakeTransport([]))
    assert problems == ['the database has no run for 2031-11-03']
    json.dumps(problems)
