"""enrich_xbrl.enrich must release each companyfacts blob once its KPIs are
computed. Holding one per filer OOM-killed 05c-enrich-xbrl on the 2026-09-14
cloud run (rc=137 at ~1,150 of 2,471 tickers)."""

import scripts.enrich_xbrl as enrich_xbrl
from data.sec_xbrl_client import SECXBRLClient
from tests.test_sec_facts_release import _stub_facts

_TICKERS = ('AAA', 'BBB', 'CCC')


def _patch_clients(tmp_path, monkeypatch):
    cik_map = {tk: f'{i + 1:010d}' for i, tk in enumerate(_TICKERS)}

    def load_cik_map(self):
        self._cik_map = cik_map
    monkeypatch.setattr(enrich_xbrl.SECLegalClient, '_load_cik_map', load_cik_map)

    built = []
    calls = []

    def make_xbrl(**kwargs):
        kwargs['facts_cache'] = str(tmp_path / 'facts')
        client = SECXBRLClient(**kwargs)

        def fake_request(url, **kw):
            calls.append(url)
            return _stub_facts()
        client._request_json = fake_request
        client.refresh_stale_facts = lambda *a, **kw: {}
        built.append(client)
        return client
    monkeypatch.setattr(enrich_xbrl, 'SECXBRLClient', make_xbrl)
    return built, calls


def test_enrich_holds_no_blobs_after_the_loop(tmp_path, monkeypatch):
    built, calls = _patch_clients(tmp_path, monkeypatch)
    records = [{'ticker': tk, 'sector': 'Technology', 'mcap': 1e9,
                'edgar_history': {}} for tk in _TICKERS]

    enrich_xbrl.enrich(records, verbose=False)

    (client,) = built
    assert len(calls) == len(_TICKERS)
    assert not [tk for tk, blob in client._cache.items() if blob is not None], \
        'every companyfacts blob must be released after its record is enriched'
    assert all(r['_provenance']['enrichments']['xbrl']['applied'] for r in records)

    # A released blob comes back from disk, not the network.
    assert client.fetch_company_facts('AAA') is not None
    assert len(calls) == len(_TICKERS)
