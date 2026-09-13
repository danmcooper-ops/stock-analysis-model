# tests/test_sec_facts_release.py
"""SECXBRLClient.release_facts: Phase 1 of analyze_stock drops each raw
companyfacts blob (7-27 MB as Python objects) once the gzipped copy is on
disk, so a full-universe sweep no longer holds one per US filer in memory.
The next fetch_company_facts must come back from disk, never the network."""

from data.sec_xbrl_client import SECXBRLClient


def _stub_facts(tag='Revenues'):
    return {'cik': 1, 'facts': {'us-gaap': {tag: {'units': {'USD': [
        {'form': '10-K', 'fy': 2024, 'fp': 'FY', 'val': 100,
         'filed': '2025-01-15', 'start': '2024-01-01', 'end': '2024-12-31',
         'accn': '0000000001-25-000001'}]}}}}}


def _client(tmp_path, monkeypatch, facts_cache):
    client = SECXBRLClient(
        cik_map={'TEST': '0000000001', 'NOCIK': None},
        name_map={'TEST': 'Test Co', 'NOCIK': 'No Cik Co'},
        email='test@example.com', request_delay=0,
        facts_cache=str(tmp_path / 'facts') if facts_cache else None)
    calls = []

    def fake_request(url, **kwargs):
        calls.append(url)
        return _stub_facts()
    monkeypatch.setattr(client, '_request_json', fake_request)
    return client, calls


def test_release_then_refetch_comes_from_disk_not_network(tmp_path, monkeypatch):
    client, calls = _client(tmp_path, monkeypatch, facts_cache=True)
    first = client.fetch_company_facts('TEST')
    assert first is not None and len(calls) == 1
    assert 'TEST' in client._cache

    assert client.release_facts('TEST') is True
    assert 'TEST' not in client._cache

    again = client.fetch_company_facts('TEST')
    assert again == first
    assert len(calls) == 1, 'a released blob must be re-read from disk, not re-downloaded'
    assert 'TEST' in client._cache  # re-hydrated for the caller that asked


def test_release_is_a_noop_without_a_disk_cache(tmp_path, monkeypatch):
    client, calls = _client(tmp_path, monkeypatch, facts_cache=False)
    client.fetch_company_facts('TEST')
    assert client.release_facts('TEST') is False
    assert client._cache['TEST'] is not None, 'nothing on disk -> keep the blob'


def test_release_keeps_the_no_cik_sentinel(tmp_path, monkeypatch):
    client, calls = _client(tmp_path, monkeypatch, facts_cache=True)
    assert client.fetch_company_facts('NOCIK') is None
    assert 'NOCIK' in client._cache and client._cache['NOCIK'] is None
    assert client.release_facts('NOCIK') is False
    assert 'NOCIK' in client._cache
    assert calls == []


def test_release_keeps_blob_when_disk_copy_is_missing(tmp_path, monkeypatch):
    client, calls = _client(tmp_path, monkeypatch, facts_cache=True)
    client.fetch_company_facts('TEST')
    path = client._facts_cache.path_for('0000000001')
    path and __import__('os').remove(path)  # simulate a failed/evicted write
    assert client.release_facts('TEST') is False
    assert client._cache['TEST'] is not None


def test_release_unknown_ticker_is_false(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch, facts_cache=True)
    assert client.release_facts('NEVER_FETCHED') is False
