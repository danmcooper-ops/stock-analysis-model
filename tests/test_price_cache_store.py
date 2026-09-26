# tests/test_price_cache_store.py
"""The price-parquet cache in Supabase Storage.

The cloud container is stateless, so output/prices/ dies with every run and
step 03 re-fetches the whole universe. This carries it across runs. What the
tests pin is mostly about *not* doing harm: an existing local file is never
overwritten by an older stored one, a half-empty local directory cannot
clobber a good cache, a failed object never leaves a truncated parquet (the
freshness check reads a parquet's index, so a truncated file could read as
current), and a missing credential is a clean no-op rather than a failure.
"""
import json
import os

import pytest

from data.price_cache_store import (MIN_KEEP_FRACTION, PriceCacheError,
                                    PriceCacheStore)


class FakeResponse:
    def __init__(self, status, body=None, content=b''):
        self.status_code = status
        self._body = body
        self.content = content
        self.text = json.dumps(body) if body is not None else ''

    def json(self):
        return self._body


class FakeSession:
    """Serves a scripted bucket; records every call."""

    def __init__(self, objects=None, fail=None):
        self.objects = dict(objects or {})     # name -> bytes
        self.fail = set(fail or ())            # names whose GET/POST errors
        self.calls = []
        self.headers_seen = []

    def request(self, method, url, headers=None, timeout=None, **kw):
        self.calls.append((method, url, kw))
        self.headers_seen.append(headers or {})
        if method == 'POST' and '/object/list/' in url:
            rows = [{'name': n, 'metadata': {'size': len(b)}}
                    for n, b in sorted(self.objects.items())]
            offset = (kw.get('json') or {}).get('offset', 0)
            limit = (kw.get('json') or {}).get('limit', 1000)
            return FakeResponse(200, rows[offset:offset + limit])
        name = url.rsplit('/', 1)[-1]
        if name in self.fail:
            return FakeResponse(404, {'error': 'nope'})
        if method == 'GET':
            return FakeResponse(200, content=self.objects[name])
        if method == 'POST':
            self.objects[name] = kw.get('data', b'')
            return FakeResponse(200, {'Key': name})
        raise AssertionError(method)


def _store(session, **kw):
    return PriceCacheStore('https://x.supabase.co/', 'svc-key', session=session, **kw)


def test_from_env_is_none_without_credentials(monkeypatch):
    # No Supabase secrets is the normal developer/smoke case, not an error.
    monkeypatch.delenv('SUPABASE_URL', raising=False)
    monkeypatch.delenv('SUPABASE_SERVICE_ROLE_KEY', raising=False)
    assert PriceCacheStore.from_env() is None
    monkeypatch.setenv('SUPABASE_URL', 'https://x.supabase.co')
    assert PriceCacheStore.from_env() is None, 'a URL without a key is not usable'
    monkeypatch.setenv('SUPABASE_SERVICE_ROLE_KEY', 'k')
    assert isinstance(PriceCacheStore.from_env(), PriceCacheStore)


def test_request_shape_carries_the_service_key():
    s = FakeSession({'AAPL.parquet': b'x'})
    assert _store(s).list_objects() == {'AAPL.parquet': 1}
    method, url, kw = s.calls[0]
    assert method == 'POST'
    assert url == 'https://x.supabase.co/storage/v1/object/list/price-cache'
    assert kw['json']['prefix'] == 'prices'


def test_listing_pages_past_the_limit(monkeypatch):
    monkeypatch.setattr('data.price_cache_store.LIST_PAGE', 2)
    s = FakeSession({f'T{i}.parquet': b'ab' for i in range(5)})
    assert len(_store(s).list_objects()) == 5
    # 2 + 2 + 1: the short page ends it.
    assert sum(1 for c in s.calls if '/object/list/' in c[1]) == 3


def test_restore_never_overwrites_a_local_file(tmp_path):
    # A resumed run may hold something fresher than the bucket.
    (tmp_path / 'AAPL.parquet').write_bytes(b'local-and-newer')
    s = FakeSession({'AAPL.parquet': b'older', 'MSFT.parquet': b'new'})
    counts = _store(s).restore(str(tmp_path))
    assert counts == {'stored': 2, 'present': 1, 'restored': 1, 'failed': 0, 'bytes': 3}
    assert (tmp_path / 'AAPL.parquet').read_bytes() == b'local-and-newer'
    assert (tmp_path / 'MSFT.parquet').read_bytes() == b'new'


def test_one_bad_object_does_not_end_the_restore(tmp_path):
    s = FakeSession({'AAA.parquet': b'a', 'BAD.parquet': b'b', 'CCC.parquet': b'c'},
                    fail={'BAD.parquet'})
    counts = _store(s).restore(str(tmp_path))
    assert counts['restored'] == 2 and counts['failed'] == 1
    # A partial cache is worth having, and nothing truncated is left behind.
    assert sorted(os.listdir(tmp_path)) == ['AAA.parquet', 'CCC.parquet']


def test_failed_download_leaves_no_temp_or_partial_file(tmp_path):
    s = FakeSession({'AAA.parquet': b'a'}, fail={'AAA.parquet'})
    counts = _store(s).restore(str(tmp_path))
    assert counts['failed'] == 1
    assert os.listdir(tmp_path) == []


def test_save_uploads_only_changed_or_new(tmp_path):
    (tmp_path / 'SAME.parquet').write_bytes(b'12345')
    (tmp_path / 'GREW.parquet').write_bytes(b'123456')
    (tmp_path / 'NEW.parquet').write_bytes(b'1')
    s = FakeSession({'SAME.parquet': b'12345', 'GREW.parquet': b'12345'})
    counts = _store(s).save(str(tmp_path), force=True)
    assert counts['uploaded'] == 2 and counts['unchanged'] == 1
    assert s.objects['GREW.parquet'] == b'123456'
    assert s.objects['NEW.parquet'] == b'1'
    uploaded = [c[1].rsplit('/', 1)[-1] for c in s.calls if c[0] == 'POST' and 'list' not in c[1]]
    assert 'SAME.parquet' not in uploaded


def test_save_all_ignores_the_size_comparison(tmp_path):
    (tmp_path / 'SAME.parquet').write_bytes(b'12345')
    s = FakeSession({'SAME.parquet': b'12345'})
    counts = _store(s).save(str(tmp_path), force=True, all_files=True)
    assert counts['uploaded'] == 1 and counts['unchanged'] == 0


def test_a_half_failed_run_cannot_clobber_a_good_cache(tmp_path):
    # 2 local against 10 stored: the run lost most of the universe.
    for i in range(2):
        (tmp_path / f'T{i}.parquet').write_bytes(b'x')
    s = FakeSession({f'S{i}.parquet': b'x' for i in range(10)})
    with pytest.raises(PriceCacheError, match='refusing to save'):
        _store(s).save(str(tmp_path))
    assert len(s.objects) == 10, 'nothing was uploaded'
    # The same call goes through when the shrink is deliberate.
    assert _store(s).save(str(tmp_path), force=True)['uploaded'] == 2


def test_the_floor_admits_a_normal_night(tmp_path):
    stored = {f'T{i}.parquet': b'x' for i in range(100)}
    for i in range(int(100 * MIN_KEEP_FRACTION)):
        (tmp_path / f'T{i}.parquet').write_bytes(b'x')
    # Exactly at the floor is allowed; the guard is for a collapse, not drift.
    assert _store(FakeSession(stored)).save(str(tmp_path))['local'] == 80


def test_save_refuses_an_empty_directory(tmp_path):
    s = FakeSession({'AAA.parquet': b'a'})
    with pytest.raises(PriceCacheError, match='no parquets'):
        _store(s).save(str(tmp_path))
    assert s.objects == {'AAA.parquet': b'a'}


def test_http_error_is_reported_not_swallowed():
    class _Boom(FakeSession):
        def request(self, *a, **k):
            return FakeResponse(500, {'error': 'server'})
    with pytest.raises(PriceCacheError, match='HTTP 500'):
        _store(_Boom(), retries=0).list_objects()


def test_upload_upserts_and_keeps_the_auth_headers(tmp_path):
    """Without x-upsert the second night's save would 409 on every object."""
    (tmp_path / 'AAA.parquet').write_bytes(b'new-bytes')
    s = FakeSession({'AAA.parquet': b'old'})
    _store(s).save(str(tmp_path), force=True)
    put = [h for h, c in zip(s.headers_seen, s.calls, strict=True)
           if c[0] == 'POST' and '/object/list/' not in c[1]]
    assert len(put) == 1
    assert put[0]['x-upsert'] == 'true'
    assert put[0]['Content-Type'] == 'application/octet-stream'
    # The extra headers must add to the service key, never replace it.
    assert put[0]['apikey'] == 'svc-key'
    assert put[0]['Authorization'] == 'Bearer svc-key'
    assert s.objects['AAA.parquet'] == b'new-bytes'


class TestCli:
    """scripts/price_cache.py — what run.sh's steps 02b and 05e2 actually call."""

    @staticmethod
    def _cli():
        from scripts import price_cache
        return price_cache

    def test_no_credentials_is_a_clean_no_op(self, monkeypatch, capsys):
        # run.sh calls this unconditionally; a dev box without Supabase must
        # not fail the step.
        monkeypatch.delenv('SUPABASE_URL', raising=False)
        monkeypatch.delenv('SUPABASE_SERVICE_ROLE_KEY', raising=False)
        assert self._cli().main(['restore']) == 0
        assert 'skipping the price cache' in capsys.readouterr().out

    def test_restore_reports_what_it_fetched(self, monkeypatch, tmp_path, capsys):
        cli = self._cli()
        s = FakeSession({'AAA.parquet': b'a' * 100, 'BBB.parquet': b'b' * 100})
        monkeypatch.setattr(cli.PriceCacheStore, 'from_env',
                            classmethod(lambda c, **k: _store(s)))
        assert cli.main(['restore', '--prices-dir', str(tmp_path)]) == 0
        assert 'restored 2/2 parquets' in capsys.readouterr().out
        assert sorted(os.listdir(tmp_path)) == ['AAA.parquet', 'BBB.parquet']

    def test_save_refusal_exits_nonzero(self, monkeypatch, tmp_path, capsys):
        cli = self._cli()
        (tmp_path / 'ONE.parquet').write_bytes(b'x')
        s = FakeSession({f'S{i}.parquet': b'x' for i in range(20)})
        monkeypatch.setattr(cli.PriceCacheStore, 'from_env',
                            classmethod(lambda c, **k: _store(s)))
        assert cli.main(['save', '--prices-dir', str(tmp_path)]) == 1
        assert 'refusing to save' in capsys.readouterr().err
        assert len(s.objects) == 20

    def test_save_missing_directory_exits_nonzero(self, monkeypatch, tmp_path):
        cli = self._cli()
        monkeypatch.setattr(cli.PriceCacheStore, 'from_env',
                            classmethod(lambda c, **k: _store(FakeSession())))
        assert cli.main(['save', '--prices-dir', str(tmp_path / 'nope')]) == 1
