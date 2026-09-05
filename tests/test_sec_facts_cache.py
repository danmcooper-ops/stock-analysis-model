# tests/test_sec_facts_cache.py
"""Persistent companyfacts cache and its filing-index freshness sweep.

The cache exists so a nightly run stops re-downloading the whole XBRL corpus
(~3.9 MB per ticker uncompressed). The risk it introduces is serving stale
fundamentals, so most of these tests are about when an entry must NOT be
served: after its filer files, past the age backstop, or when a sweep could
not read the index it needed.
"""

import gzip
import json
import os
import time
import urllib.error
from datetime import date

import pytest

from data.sec_facts_cache import (DEFAULT_MAX_AGE_DAYS, SECFactsCache,
                                  _norm_cik)
from data.sec_xbrl_client import ABSENT, SECXBRLClient

AAPL_CIK = '0000320193'
FACTS = {'cik': 320193, 'entityName': 'Apple Inc.',
         'facts': {'us-gaap': {'Revenues': {'units': {'USD': []}}}}}


@pytest.fixture
def cache(tmp_path):
    return SECFactsCache(cache_dir=str(tmp_path / 'sec_facts'))


# --- storage ---------------------------------------------------------------

class TestStorage:

    def test_roundtrip_is_gzipped_on_disk(self, cache):
        path = cache.put(AAPL_CIK, FACTS)
        assert path.endswith('.json.gz')
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            assert json.load(f) == FACTS
        assert cache.get(AAPL_CIK) == FACTS
        assert cache.stats()['hits'] == 1 and cache.stats()['writes'] == 1

    def test_padded_and_bare_ciks_are_the_same_entry(self, cache):
        """The ticker map holds zero-padded CIKs, the daily index bare ones."""
        cache.put('320193', FACTS)
        assert cache.get(AAPL_CIK) == FACTS
        assert cache.invalidate(['320193']) == 1
        assert cache.get(AAPL_CIK) is None

    @pytest.mark.parametrize('bad', ['', '  ', 'abc', None])
    def test_unusable_cik_is_ignored_not_raised(self, cache, bad):
        assert _norm_cik(bad) is None
        assert cache.path_for(bad) is None
        assert cache.put(bad, FACTS) is None
        assert cache.get(bad) is None

    def test_miss_on_absent_entry(self, cache):
        assert cache.get(AAPL_CIK) is None
        assert cache.stats()['misses'] == 1

    def test_corrupt_entry_reads_as_a_miss(self, cache):
        path = cache.put(AAPL_CIK, FACTS)
        with open(path, 'wb') as f:            # truncated gzip
            f.write(b'\x1f\x8b not really gzip')
        assert cache.get(AAPL_CIK) is None
        assert cache.put(AAPL_CIK, FACTS) and cache.get(AAPL_CIK) == FACTS

    def test_none_is_never_written(self, cache):
        assert cache.put(AAPL_CIK, None) is None
        assert cache.entry_count() == 0

    def test_no_temp_files_survive_a_write(self, cache):
        cache.put(AAPL_CIK, FACTS)
        assert [f for f in os.listdir(cache.cache_dir) if '.tmp.' in f] == []

    def test_age_backstop_expires_an_entry(self, tmp_path):
        c = SECFactsCache(cache_dir=str(tmp_path / 'f'), max_age_days=7)
        path = c.put(AAPL_CIK, FACTS)
        assert c.get(AAPL_CIK) == FACTS
        old = time.time() - 8 * 86400
        os.utime(path, (old, old))
        assert c.get(AAPL_CIK) is None
        assert c.age_days(AAPL_CIK) == pytest.approx(8, abs=0.1)

    def test_max_age_comes_from_the_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv('SEC_FACTS_CACHE_MAX_AGE_DAYS', '3')
        assert SECFactsCache(cache_dir=str(tmp_path)).max_age_days == 3
        monkeypatch.setenv('SEC_FACTS_CACHE_MAX_AGE_DAYS', 'not-a-number')
        assert SECFactsCache(cache_dir=str(tmp_path)).max_age_days == DEFAULT_MAX_AGE_DAYS

    def test_invalidate_counts_only_what_existed(self, cache):
        cache.put(AAPL_CIK, FACTS)
        assert cache.invalidate([AAPL_CIK, '0000000009']) == 1
        assert cache.get(AAPL_CIK) is None

    def test_sweep_watermark_roundtrip(self, cache):
        assert cache.last_sweep() is None
        cache.record_sweep(date(2026, 9, 4))
        assert cache.last_sweep() == date(2026, 9, 4)
        cache.record_sweep(date(2026, 9, 5))
        assert cache.last_sweep() == date(2026, 9, 5)

    def test_unreadable_state_file_reads_as_no_sweep(self, cache):
        cache.record_sweep(date(2026, 9, 4))
        with open(cache._state_path(), 'w', encoding='utf-8') as f:
            f.write('{not json')
        assert cache.last_sweep() is None


# --- client integration ----------------------------------------------------

def _client(tmp_path, **kw):
    return SECXBRLClient(cik_map={'AAPL': AAPL_CIK}, name_map={'AAPL': 'Apple'},
                         facts_cache=str(tmp_path / 'sec_facts'), **kw)


class TestClientUsesTheCache:

    def test_disk_hit_avoids_the_network(self, tmp_path):
        c = _client(tmp_path)
        c._facts_cache.put(AAPL_CIK, FACTS)
        c._request_bytes = lambda *a, **k: pytest.fail('network used on a hit')
        assert c.fetch_company_facts('AAPL') == FACTS

    def test_fetch_populates_the_cache_for_the_next_process(self, tmp_path):
        calls = []
        c = _client(tmp_path)
        c._request_json = lambda url, **k: (calls.append(url), FACTS)[1]
        assert c.fetch_company_facts('AAPL') == FACTS
        fresh = _client(tmp_path)
        fresh._request_bytes = lambda *a, **k: pytest.fail('should be cached')
        assert fresh.fetch_company_facts('AAPL') == FACTS
        assert len(calls) == 1

    def test_failed_fetch_is_not_cached(self, tmp_path):
        """A transient failure must not read as 'no XBRL data' next run."""
        c = _client(tmp_path)
        c._request_json = lambda url, **k: None
        assert c.fetch_company_facts('AAPL') is None
        assert c._facts_cache.entry_count() == 0

    def test_no_cache_configured_keeps_memory_only_behaviour(self, tmp_path):
        c = SECXBRLClient(cik_map={'AAPL': AAPL_CIK}, name_map={})
        assert c._facts_cache is None and c.facts_cache_stats() is None
        c._request_json = lambda url, **k: FACTS
        assert c.fetch_company_facts('AAPL') == FACTS
        assert c._cache['AAPL'] == FACTS     # the in-memory dict tests rely on

    def test_unmapped_ticker_never_touches_the_cache(self, tmp_path):
        c = _client(tmp_path)
        c._request_bytes = lambda *a, **k: pytest.fail('no CIK: no request')
        assert c.fetch_company_facts('NOPE') is None


class TestGzipTransport:

    def _resp(self, body, encoding=None):
        class _R:
            headers = {'Content-Encoding': encoding} if encoding else {}

            def read(self_inner):
                return body

            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *a):
                return False
        return _R()

    def test_asks_for_gzip_and_decompresses(self, tmp_path, monkeypatch):
        seen = {}
        payload = json.dumps(FACTS).encode()

        def fake_urlopen(req, **kw):
            seen['headers'] = dict(req.header_items())
            return self._resp(gzip.compress(payload), 'gzip')

        monkeypatch.setattr('urllib.request.urlopen', fake_urlopen)
        c = _client(tmp_path)
        assert c._request_json('https://example.test/x') == FACTS
        assert seen['headers'].get('Accept-encoding') == 'gzip'

    def test_identity_response_still_works(self, tmp_path, monkeypatch):
        payload = json.dumps(FACTS).encode()
        monkeypatch.setattr('urllib.request.urlopen',
                            lambda req, **kw: self._resp(payload))
        assert _client(tmp_path)._request_json('https://example.test/x') == FACTS

    def test_absent_codes_are_distinguished_from_failures(self, tmp_path, monkeypatch):
        def raise_http(code):
            def _f(req, **kw):
                raise urllib.error.HTTPError('u', code, 'no', {}, None)
            return _f
        c = _client(tmp_path)
        monkeypatch.setattr('urllib.request.urlopen', raise_http(403))
        assert c._request_bytes('https://x.test', absent_codes=(403, 404)) is ABSENT
        assert c._request_bytes('https://x.test') is None       # not declared absent
        monkeypatch.setattr('urllib.request.urlopen', raise_http(500))
        assert c._request_bytes('https://x.test', absent_codes=(403, 404)) is None

    def test_bad_json_is_not_an_exception(self, tmp_path, monkeypatch):
        monkeypatch.setattr('urllib.request.urlopen',
                            lambda req, **kw: self._resp(b'{not json'))
        assert _client(tmp_path)._request_json('https://x.test') is None


INDEX = (
    'CIK|Company Name|Form Type|Date Filed|File Name\n'
    '320193|APPLE INC|10-Q|20260902|edgar/data/320193/x.txt\n'
    '21344|COCA COLA CO|8-K|20260902|edgar/data/21344/y.txt\n'
    '1682852|MODERNA|10-K/A|20260902|edgar/data/1682852/z.txt\n'
    '1000|SOME FUND|4|20260902|edgar/data/1000/a.txt\n'
    '1001|FOREIGN CO|6-K|20260902|edgar/data/1001/b.txt\n'
    'malformed line without pipes\n'
)


class TestFilingIndexSweep:

    def _client_with_index(self, tmp_path, per_day):
        """per_day: {date: index text, ABSENT, or None (unreadable)}."""
        c = _client(tmp_path)
        asked = []

        def fake_bytes(url, timeout=20, absent_codes=()):
            day = url.rsplit('.', 2)[-2][-8:]
            asked.append(day)
            val = per_day.get(day, ABSENT)
            return val.encode() if isinstance(val, str) else val

        c._request_bytes = fake_bytes
        c._asked = asked
        return c

    def test_only_fact_bearing_forms_invalidate(self, tmp_path):
        c = self._client_with_index(tmp_path, {'20260902': INDEX})
        got = c._daily_index_ciks(date(2026, 9, 2))
        # 10-Q, 10-K/A, 6-K and 8-K all contribute statement facts; ownership
        # forms (4) never do. See _FACT_BEARING_FORMS for the measurements.
        assert got == {'0000320193', '0001682852', '0000001001', '0000021344'}

    def test_absent_index_means_nothing_filed(self, tmp_path):
        c = self._client_with_index(tmp_path, {})
        assert c._daily_index_ciks(date(2026, 9, 6)) == set()

    def test_unreadable_index_is_not_an_empty_day(self, tmp_path):
        c = self._client_with_index(tmp_path, {'20260902': None})
        assert c._daily_index_ciks(date(2026, 9, 2)) is None

    def test_sweep_evicts_the_filers_that_filed(self, tmp_path):
        c = self._client_with_index(tmp_path, {'20260902': INDEX})
        c._facts_cache.put(AAPL_CIK, FACTS)          # filed a 10-Q
        c._facts_cache.put('0000021344', FACTS)      # filed an 8-K
        c._facts_cache.put('0000000999', FACTS)      # filed nothing
        c._facts_cache.record_sweep(date(2026, 9, 1))
        out = c.refresh_stale_facts(today=date(2026, 9, 3))
        assert out['invalidated'] == 2
        assert c._facts_cache.get(AAPL_CIK) is None
        assert c._facts_cache.get('0000021344') is None
        assert c._facts_cache.get('0000000999') == FACTS
        assert c._facts_cache.last_sweep() == date(2026, 9, 2)

    def test_first_sweep_only_starts_the_clock(self, tmp_path):
        c = self._client_with_index(tmp_path, {'20260902': INDEX})
        out = c.refresh_stale_facts(today=date(2026, 9, 3))
        assert out['reason'] == 'first sweep' and c._asked == []
        assert c._facts_cache.last_sweep() == date(2026, 9, 2)

    def test_already_current_does_no_work(self, tmp_path):
        c = self._client_with_index(tmp_path, {})
        c._facts_cache.record_sweep(date(2026, 9, 2))
        assert c.refresh_stale_facts(today=date(2026, 9, 3))['reason'] == 'already current'
        assert c._asked == []

    def test_watermark_stops_at_the_first_unreadable_day(self, tmp_path):
        """Advancing past a day whose index never loaded would lose its
        filings for good, so the sweep stops there and resumes next run."""
        c = self._client_with_index(tmp_path, {
            '20260902': INDEX, '20260903': None, '20260904': INDEX})
        c._facts_cache.put(AAPL_CIK, FACTS)
        c._facts_cache.record_sweep(date(2026, 9, 1))
        out = c.refresh_stale_facts(today=date(2026, 9, 5))
        assert out['stalled_on'] == '2026-09-03'
        assert c._facts_cache.last_sweep() == date(2026, 9, 2)
        assert c._facts_cache.get(AAPL_CIK) is None      # 09-02 still applied
        assert '20260904' not in c._asked                # stopped, not skipped

    def test_long_gap_falls_back_to_the_age_backstop(self, tmp_path):
        c = self._client_with_index(tmp_path, {})
        c._facts_cache.record_sweep(date(2025, 1, 1))
        out = c.refresh_stale_facts(today=date(2026, 9, 5))
        assert 'gap' in out['reason'] and c._asked == []
        assert c._facts_cache.last_sweep() == date(2025, 1, 1)

    def test_sweep_without_a_cache_is_a_no_op(self):
        c = SECXBRLClient(cik_map={}, name_map={})
        assert c.refresh_stale_facts()['reason'] == 'no persistent cache'


class TestPruning:
    """Entries past the backstop can never be served, so they are disk leaks.

    Only filers that left the universe get here — an active one is rewritten
    by its next fetch."""

    def test_prune_removes_only_expired_entries(self, tmp_path):
        c = SECFactsCache(cache_dir=str(tmp_path / 'f'), max_age_days=7)
        fresh = c.put(AAPL_CIK, FACTS)
        stale = c.put('0000021344', FACTS)
        old = time.time() - 8 * 86400
        os.utime(stale, (old, old))
        assert c.prune_expired() == 1
        assert os.path.exists(fresh) and not os.path.exists(stale)
        assert c.prune_expired() == 0

    def test_prune_is_safe_on_a_missing_directory(self, tmp_path):
        assert SECFactsCache(cache_dir=str(tmp_path / 'nope')).prune_expired() == 0

    def test_no_backstop_means_no_pruning(self, tmp_path):
        """max_age_days<=0 disables expiry; only filing sweeps evict then."""
        c = SECFactsCache(cache_dir=str(tmp_path / 'f'), max_age_days=0)
        assert c.max_age_days is None
        path = c.put(AAPL_CIK, FACTS)
        old = time.time() - 3650 * 86400
        os.utime(path, (old, old))
        assert c.prune_expired() == 0 and c.get(AAPL_CIK) == FACTS

    def test_sweep_prunes(self, tmp_path):
        c = _client(tmp_path)
        c._request_bytes = lambda *a, **k: ABSENT
        c._facts_cache.max_age_days = 7
        stale = c._facts_cache.put('0000007777', FACTS)
        old = time.time() - 9 * 86400
        os.utime(stale, (old, old))
        c._facts_cache.record_sweep(date(2026, 9, 1))
        assert c.refresh_stale_facts(today=date(2026, 9, 3))['pruned'] == 1
