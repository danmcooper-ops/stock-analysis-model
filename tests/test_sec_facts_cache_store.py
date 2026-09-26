# tests/test_sec_facts_cache_store.py
"""Carrying the companyfacts cache between runs, and the guard that makes it sound.

Unlike the price parquets, a companyfacts blob carries no internal date, and
the freshness backstop read file mtime — which a restore resets. Transporting
the cache without fixing that would have said "fetched today" about a blob of
any age and silently disabled the only protection against indefinitely-stale
fundamentals.

The backstop now also reads the sweep watermark, which survives the trip and
measures whether filing-driven eviction kept up. These tests pin that guard,
and pin that evictions reach the bucket instead of returning on the next
restore.
"""
import gzip
import json
import os
from datetime import date, timedelta

import pytest

from data.sec_facts_cache import SECFactsCache
from data.sec_facts_cache_store import SecFactsCacheStore, StorageError
from data.supabase_storage import StorageError as _StorageError
from tests.test_price_cache_store import FakeResponse, FakeSession


def _blob(payload=None):
    import io
    buf = io.BytesIO()
    with gzip.open(buf, 'wt', encoding='utf-8') as f:
        json.dump(payload or {'facts': {}}, f)
    return buf.getvalue()


def _state(days_ago):
    through = (date.today() - timedelta(days=days_ago)).isoformat()
    return json.dumps({'last_index_sweep': through}).encode()


class DeletingFakeSession(FakeSession):
    """FakeSession plus the DELETE the eviction path needs."""

    def request(self, method, url, headers=None, timeout=None, **kw):
        if method == 'DELETE':
            self.calls.append((method, url, kw))
            self.headers_seen.append(headers or {})
            for p in (kw.get('json') or {}).get('prefixes', []):
                self.objects.pop(os.path.basename(p), None)
            return FakeResponse(200, [])
        return super().request(method, url, headers=headers, timeout=timeout, **kw)


def _store(session):
    return SecFactsCacheStore('https://x.supabase.co/', 'svc-key', session=session)


class TestSweepLagGuard:
    """The half of the age backstop that survives being restored."""

    def test_a_current_watermark_keeps_the_cache_usable(self, tmp_path):
        cache = SECFactsCache(cache_dir=str(tmp_path))
        cache.put('0000320193', {'facts': {'x': 1}})
        cache.record_sweep(date.today() - timedelta(days=1))
        assert cache.sweep_is_lagging() is False
        assert cache.get('0000320193') == {'facts': {'x': 1}}

    def test_a_lagging_sweep_makes_every_entry_read_as_missing(self, tmp_path):
        # The blob was written just now, so mtime says "fresh" — exactly the
        # state a restore produces. Only the watermark reveals the truth.
        cache = SECFactsCache(cache_dir=str(tmp_path), max_age_days=30)
        cache.put('0000320193', {'facts': {'x': 1}})
        cache.record_sweep(date.today() - timedelta(days=45))
        assert cache.sweep_is_lagging() is True
        assert cache.get('0000320193') is None
        assert cache.age_days('0000320193') < 1, 'mtime alone would have said fresh'

    def test_no_watermark_falls_back_to_mtime(self, tmp_path):
        # A dev box that never swept: mtime is honest there, so the guard
        # must not lock the cache out.
        cache = SECFactsCache(cache_dir=str(tmp_path), max_age_days=30)
        cache.put('0000320193', {'facts': {'x': 1}})
        assert cache.sweep_lag_days() is None
        assert cache.sweep_is_lagging() is False
        assert cache.get('0000320193') == {'facts': {'x': 1}}

    def test_recording_a_sweep_clears_the_memo(self, tmp_path):
        cache = SECFactsCache(cache_dir=str(tmp_path), max_age_days=30)
        cache.put('0000320193', {'facts': {'x': 1}})
        cache.record_sweep(date.today() - timedelta(days=45))
        assert cache.get('0000320193') is None
        # The sweep catches up mid-run; the cache must become usable again
        # without a restart.
        cache.record_sweep(date.today())
        assert cache.get('0000320193') == {'facts': {'x': 1}}

    def test_a_disabled_backstop_never_lags(self, tmp_path):
        cache = SECFactsCache(cache_dir=str(tmp_path), max_age_days=0)
        cache.put('0000320193', {'facts': {'x': 1}})
        cache.record_sweep(date.today() - timedelta(days=9999))
        assert cache.sweep_is_lagging() is False
        assert cache.get('0000320193') == {'facts': {'x': 1}}


class TestRoundTrip:
    def test_restore_brings_the_watermark_with_the_blobs(self, tmp_path):
        s = DeletingFakeSession({'0000320193.json.gz': _blob(),
                                 '_state.json': _state(1)})
        counts = _store(s).restore(str(tmp_path))
        assert counts['restored'] == 2
        # Without _state.json the restored blobs would be vouched for by
        # nothing, and the guard would fall back to a reset mtime.
        assert os.path.exists(tmp_path / '_state.json')
        assert SECFactsCache(cache_dir=str(tmp_path)).sweep_lag_days() == 1

    def test_restore_keeps_a_local_entry(self, tmp_path):
        (tmp_path / '0000320193.json.gz').write_bytes(_blob({'local': True}))
        s = DeletingFakeSession({'0000320193.json.gz': _blob({'stored': True})})
        counts = _store(s).restore(str(tmp_path))
        assert counts['restored'] == 0 and counts['present'] == 1
        assert SECFactsCache(cache_dir=str(tmp_path)).get('320193') == {'local': True}

    def test_save_always_uploads_the_watermark(self, tmp_path):
        # Byte-identical to what is stored, but it still goes: a watermark
        # left behind its blobs is what the guard reads as a lagging sweep.
        (tmp_path / '0000320193.json.gz').write_bytes(_blob())
        (tmp_path / '_state.json').write_bytes(_state(1))
        s = DeletingFakeSession({'0000320193.json.gz': _blob(),
                                 '_state.json': _state(1)})
        counts = _store(s).save(str(tmp_path))
        assert counts['uploaded'] == 1 and counts['unchanged'] == 1
        sent = [c[1].rsplit('/', 1)[-1] for c in s.calls
                if c[0] == 'POST' and '/object/list/' not in c[1]]
        assert sent == ['_state.json']

    def test_evictions_reach_the_bucket(self, tmp_path):
        """An evicted blob left stored would return on the next restore.

        The sweep would not evict it again — the watermark has already moved
        past the day its filer filed — so it would be served indefinitely.
        """
        (tmp_path / '0000320193.json.gz').write_bytes(_blob())
        (tmp_path / '_state.json').write_bytes(_state(1))
        s = DeletingFakeSession({'0000320193.json.gz': _blob(),
                                 '0000789019.json.gz': _blob(),   # evicted locally
                                 '_state.json': _state(2)})
        counts = _store(s).save(str(tmp_path))
        assert counts['deleted'] == 1
        assert '0000789019.json.gz' not in s.objects

    def test_a_collapsed_cache_cannot_delete_a_good_one(self, tmp_path):
        (tmp_path / '_state.json').write_bytes(_state(1))
        s = DeletingFakeSession({f'{i:010d}.json.gz': _blob() for i in range(20)})
        with pytest.raises(StorageError, match='refusing to save'):
            _store(s).save(str(tmp_path))
        assert len(s.objects) == 20, 'nothing was uploaded or deleted'

    def test_save_refuses_an_empty_cache_dir(self, tmp_path):
        s = DeletingFakeSession({'0000320193.json.gz': _blob()})
        with pytest.raises(StorageError, match='no companyfacts blobs'):
            _store(s).save(str(tmp_path))
        assert len(s.objects) == 1

    def test_the_error_type_is_the_shared_one(self):
        assert StorageError is _StorageError


class TestCli:
    @staticmethod
    def _cli():
        from scripts import sec_cache
        return sec_cache

    def test_no_credentials_is_a_clean_no_op(self, monkeypatch, capsys):
        monkeypatch.delenv('SUPABASE_URL', raising=False)
        monkeypatch.delenv('SUPABASE_SERVICE_ROLE_KEY', raising=False)
        assert self._cli().main(['restore']) == 0
        assert 'skipping the sec cache' in capsys.readouterr().out

    def test_restore_reports_a_lagging_sweep(self, monkeypatch, tmp_path, capsys):
        # The failure mode worth explaining: the restore "works" but every
        # entry then reads as missing, and without this line that is a mystery.
        cli = self._cli()
        s = DeletingFakeSession({'0000320193.json.gz': _blob(),
                                 '_state.json': _state(60)})
        monkeypatch.setattr(cli.SecFactsCacheStore, 'from_env',
                            classmethod(lambda c, **k: _store(s)))
        assert cli.main(['restore', '--cache-dir', str(tmp_path)]) == 0
        out = capsys.readouterr().out
        assert 'restored 2/2' in out
        assert 'filing sweep is 60d behind' in out

    def test_restore_reports_a_healthy_sweep(self, monkeypatch, tmp_path, capsys):
        cli = self._cli()
        s = DeletingFakeSession({'0000320193.json.gz': _blob(),
                                 '_state.json': _state(2)})
        monkeypatch.setattr(cli.SecFactsCacheStore, 'from_env',
                            classmethod(lambda c, **k: _store(s)))
        assert cli.main(['restore', '--cache-dir', str(tmp_path)]) == 0
        assert 'the cache is usable' in capsys.readouterr().out

    def test_save_refusal_exits_nonzero(self, monkeypatch, tmp_path, capsys):
        cli = self._cli()
        (tmp_path / '_state.json').write_bytes(_state(1))
        s = DeletingFakeSession({f'{i:010d}.json.gz': _blob() for i in range(20)})
        monkeypatch.setattr(cli.SecFactsCacheStore, 'from_env',
                            classmethod(lambda c, **k: _store(s)))
        assert cli.main(['save', '--cache-dir', str(tmp_path)]) == 1
        assert 'refusing to save' in capsys.readouterr().err


def test_the_first_save_creates_the_bucket(tmp_path):
    """Night one has no bucket; making it beats failing every upload."""
    created = []

    class _NoBucket(DeletingFakeSession):
        def request(self, method, url, headers=None, timeout=None, **kw):
            if url.endswith('/bucket'):            # the create
                created.append((kw.get('json') or {}))
                return FakeResponse(200, {'name': 'sec-facts-cache'})
            if '/bucket/' in url:                  # the existence probe
                return FakeResponse(404, {'error': 'not found'})
            return super().request(method, url, headers=headers, timeout=timeout, **kw)

    (tmp_path / '0000320193.json.gz').write_bytes(_blob())
    (tmp_path / '_state.json').write_bytes(_state(1))
    s = _NoBucket({})
    counts = _store(s).save(str(tmp_path))
    assert counts['uploaded'] == 2
    assert created and created[0].get('public') is False, 'the cache bucket is private'
