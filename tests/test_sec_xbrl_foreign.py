# tests/test_sec_xbrl_foreign.py
"""Foreign-issuer support: 20-F / 40-F filings, IFRS taxonomy, FX conversion.

These tests use synthetic XBRL fact stubs to avoid live SEC API calls.
"""

import pytest


from data import sec_xbrl_client as xbrl_mod
from data.sec_xbrl_client import SECXBRLClient


def _make_client():
    return SECXBRLClient(
        cik_map={'TEST': '0000000001'},
        name_map={'TEST': 'Test Co'},
        email='test@example.com',
        request_delay=0,
    )


def _us_gaap_revenue_facts(years_values, form='10-K'):
    """Build a minimal US-GAAP companyfacts stub for the Revenues concept."""
    entries = []
    for fy, val in years_values.items():
        entries.append({
            'form': form, 'fy': fy, 'fp': 'FY', 'val': val,
            'filed': f'{fy + 1}-01-15',
            'start': f'{fy}-01-01', 'end': f'{fy}-12-31',
        })
    return {'facts': {'us-gaap': {'Revenues': {'units': {'USD': entries}}}}}


def _ifrs_revenue_facts(years_values, currency='EUR', form='20-F'):
    """Build a minimal IFRS companyfacts stub for the Revenue concept."""
    entries = []
    for fy, val in years_values.items():
        entries.append({
            'form': form, 'fy': fy, 'fp': 'FY', 'val': val,
            'filed': f'{fy + 1}-03-15',
            'start': f'{fy}-01-01', 'end': f'{fy}-12-31',
        })
    return {'facts': {'ifrs-full': {'Revenue': {'units': {currency: entries}}}}}


class TestFormTypeFilter:
    """Phase 1A: 40-F (Canadian annual) must pass the form filter."""

    def test_40f_accepted_by_annual_extractor(self):
        c = _make_client()
        facts = _us_gaap_revenue_facts({2022: 100, 2023: 110, 2024: 120}, form='40-F')
        vals = c._extract_annual_values(facts, ['Revenues'], form_filter='10-K')
        assert vals == {2022: 100, 2023: 110, 2024: 120}

    def test_20f_still_accepted(self):
        c = _make_client()
        facts = _us_gaap_revenue_facts({2023: 50, 2024: 55}, form='20-F')
        vals = c._extract_annual_values(facts, ['Revenues'], form_filter='10-K')
        assert vals == {2023: 50, 2024: 55}

    def test_unknown_form_rejected(self):
        c = _make_client()
        facts = _us_gaap_revenue_facts({2024: 999}, form='8-K')
        vals = c._extract_annual_values(facts, ['Revenues'], form_filter='10-K')
        assert vals == {}


class TestIFRSTaxonomy:
    """Phase 2A: IFRS facts must be discoverable alongside US-GAAP."""

    def test_extract_concept_annual_falls_back_to_ifrs(self):
        c = _make_client()
        facts = _ifrs_revenue_facts({2022: 1e9, 2023: 1.1e9, 2024: 1.2e9},
                                    currency='USD', form='20-F')
        vals, tax, ccy = c._extract_concept_annual(facts, 'revenue')
        assert tax == 'ifrs-full'
        assert ccy == 'USD'
        assert vals == {2022: 1e9, 2023: 1.1e9, 2024: 1.2e9}

    def test_extract_concept_annual_prefers_us_gaap(self):
        """If both taxonomies have data, US-GAAP wins."""
        c = _make_client()
        facts = {
            'facts': {
                'us-gaap': _us_gaap_revenue_facts({2024: 100})['facts']['us-gaap'],
                'ifrs-full': _ifrs_revenue_facts({2024: 999},
                                                  currency='USD')['facts']['ifrs-full'],
            }
        }
        vals, tax, _ccy = c._extract_concept_annual(facts, 'revenue')
        assert tax == 'us-gaap'
        assert vals[2024] == 100

    def test_ifrs_alternate_tag_revenue_from_contracts(self):
        """IFRS filers sometimes tag revenue as RevenueFromContractsWithCustomers."""
        c = _make_client()
        entries = [{
            'form': '20-F', 'fy': 2024, 'fp': 'FY', 'val': 5e9,
            'filed': '2025-03-15',
            'start': '2024-01-01', 'end': '2024-12-31',
        }]
        facts = {'facts': {'ifrs-full': {
            'RevenueFromContractsWithCustomers': {'units': {'USD': entries}}
        }}}
        vals, tax, _ccy = c._extract_concept_annual(facts, 'revenue')
        assert tax == 'ifrs-full'
        assert vals == {2024: 5e9}


class TestCurrencyDetection:
    def test_detect_usd(self):
        c = _make_client()
        facts = _us_gaap_revenue_facts({2024: 100})
        assert c._detect_currency(facts, 'revenue') == 'USD'

    def test_detect_eur(self):
        c = _make_client()
        facts = _ifrs_revenue_facts({2024: 100}, currency='EUR')
        assert c._detect_currency(facts, 'revenue') == 'EUR'

    def test_detect_jpy(self):
        c = _make_client()
        facts = _ifrs_revenue_facts({2024: 1e12}, currency='JPY')
        assert c._detect_currency(facts, 'revenue') == 'JPY'

    def test_prefers_usd_when_both_present(self):
        c = _make_client()
        entries_eur = [{
            'form': '20-F', 'fy': 2024, 'fp': 'FY', 'val': 100,
            'filed': '2025-03-15',
            'start': '2024-01-01', 'end': '2024-12-31',
        }]
        entries_usd = [{
            'form': '20-F', 'fy': 2024, 'fp': 'FY', 'val': 110,
            'filed': '2025-03-15',
            'start': '2024-01-01', 'end': '2024-12-31',
        }]
        facts = {'facts': {'ifrs-full': {'Revenue': {
            'units': {'EUR': entries_eur, 'USD': entries_usd}
        }}}}
        assert c._detect_currency(facts, 'revenue') == 'USD'

    def test_rejects_non_currency_keys(self):
        """Unit keys like 'shares' and 'pure' must not be returned as currency."""
        c = _make_client()
        entries = [{
            'form': '10-K', 'fy': 2024, 'fp': 'FY', 'val': 1e9,
            'filed': '2025-01-15',
            'start': '2024-01-01', 'end': '2024-12-31',
        }]
        facts = {'facts': {'us-gaap': {'Revenues': {
            'units': {'shares': entries, 'pure': entries}
        }}}}
        assert c._detect_currency(facts, 'revenue') is None


class TestFXConversion:
    """Phase 2B: native-currency values must be multiplied through FX rates."""

    def test_apply_fx_annual_basic(self):
        from data.sec_xbrl_client import _apply_fx_annual
        rates = {2020: 0.165, 2021: 0.159, 2022: 0.142}
        vals = {2020: 100, 2021: 200, 2022: 300}
        out = _apply_fx_annual(vals, rates)
        assert out[2020] == pytest.approx(16.5)
        assert out[2021] == pytest.approx(31.8)
        assert out[2022] == pytest.approx(42.6)

    def test_apply_fx_passes_through_missing_year(self):
        """Years with no FX rate keep native magnitude (better than dropping)."""
        from data.sec_xbrl_client import _apply_fx_annual
        rates = {2024: 1.10}
        vals = {2000: 100, 2024: 200}
        out = _apply_fx_annual(vals, rates)
        assert out[2000] == 100  # pre-2003 EUR has no rate
        assert out[2024] == pytest.approx(220)

    def test_apply_fx_empty_rates_passthrough(self):
        from data.sec_xbrl_client import _apply_fx_annual
        assert _apply_fx_annual({2024: 100}, {}) == {2024: 100}

    def test_fetch_historical_converts_eur_to_usd(self, monkeypatch):
        """End-to-end: IFRS / EUR filing returns USD-denominated history."""
        c = _make_client()
        facts = _ifrs_revenue_facts(
            {2020: 5e9, 2021: 6e9, 2022: 7e9}, currency='EUR', form='20-F')

        monkeypatch.setattr(c, 'fetch_company_facts', lambda tk: facts)
        monkeypatch.setattr(xbrl_mod, '_get_fx_rates_to_usd',
                            lambda ccy: {2020: 1.20, 2021: 1.18, 2022: 1.05})

        h = c.fetch_historical_financials('TEST')
        assert h['reporting_currency'] == 'EUR'
        assert h['fx_converted'] is True
        assert h['revenue_history'][2020] == pytest.approx(6e9)   # 5B × 1.20
        assert h['revenue_history'][2021] == pytest.approx(7.08e9)  # 6B × 1.18
        assert h['revenue_history'][2022] == pytest.approx(7.35e9)  # 7B × 1.05

    def test_fetch_historical_us_filer_unchanged(self, monkeypatch):
        """USD filer must skip FX conversion entirely (no regression)."""
        c = _make_client()
        facts = _us_gaap_revenue_facts({2022: 100e9, 2023: 110e9, 2024: 120e9})

        monkeypatch.setattr(c, 'fetch_company_facts', lambda tk: facts)
        # Make FX fetch explode if it's called — must NOT be invoked for USD.
        monkeypatch.setattr(xbrl_mod, '_get_fx_rates_to_usd',
                            lambda ccy: (_ for _ in ()).throw(AssertionError(
                                'FX must not be fetched for USD filer')))

        h = c.fetch_historical_financials('TEST')
        assert h['reporting_currency'] == 'USD'
        assert h['fx_converted'] is False
        assert h['revenue_history'] == {2022: 100e9, 2023: 110e9, 2024: 120e9}

    def test_fetch_historical_40f_canadian_in_cad(self, monkeypatch):
        """Canadian filers (40-F + CAD) should flow through identically to 20-F."""
        c = _make_client()
        # Canadian 40-F filers usually report in US-GAAP but in CAD units.
        entries = [{
            'form': '40-F', 'fy': fy, 'fp': 'FY', 'val': val,
            'filed': f'{fy + 1}-02-15',
            'start': f'{fy}-01-01', 'end': f'{fy}-12-31',
        } for fy, val in [(2022, 50e9), (2023, 55e9), (2024, 60e9)]]
        facts = {'facts': {'us-gaap': {'Revenues': {'units': {'CAD': entries}}}}}

        monkeypatch.setattr(c, 'fetch_company_facts', lambda tk: facts)
        monkeypatch.setattr(xbrl_mod, '_get_fx_rates_to_usd',
                            lambda ccy: {2022: 0.77, 2023: 0.74, 2024: 0.73})

        h = c.fetch_historical_financials('TEST')
        assert h['reporting_currency'] == 'CAD'
        assert h['fx_converted'] is True
        assert h['revenue_history'][2024] == pytest.approx(60e9 * 0.73)


class TestBuildYfinanceShapeCapexDA:
    """build_yfinance_shape must emit Capex / D&A / current-asset rows.

    Before 2026-07, the XBRL cash-flow frame carried only Operating Cash
    Flow — so calculate_fundamental_growth returned {} and the FCF
    extractor returned None for every XBRL-sourced record (100% of
    'sec_xbrl+yfinance' rows had fundamental_growth=None and no DCF).
    """

    @staticmethod
    def _flow_entries(years_values, form='10-K'):
        return [{
            'form': form, 'fy': fy, 'fp': 'FY', 'val': val,
            'filed': f'{fy + 1}-01-15',
            'start': f'{fy}-01-01', 'end': f'{fy}-12-31',
        } for fy, val in years_values.items()]

    def _facts(self):
        f = self._flow_entries
        gaap = {
            'Revenues':                                   {'units': {'USD': f({2023: 100e9, 2024: 110e9})}},
            'NetIncomeLoss':                              {'units': {'USD': f({2023: 10e9, 2024: 12e9})}},
            'OperatingIncomeLoss':                        {'units': {'USD': f({2023: 15e9, 2024: 18e9})}},
            'NetCashProvidedByUsedInOperatingActivities': {'units': {'USD': f({2023: 14e9, 2024: 16e9})}},
            'PaymentsToAcquirePropertyPlantAndEquipment': {'units': {'USD': f({2023: 3e9, 2024: 4e9})}},
            'DepreciationDepletionAndAmortization':       {'units': {'USD': f({2023: 2e9, 2024: 2.5e9})}},
            'AssetsCurrent':                              {'units': {'USD': f({2023: 40e9, 2024: 45e9})}},
            'LiabilitiesCurrent':                         {'units': {'USD': f({2023: 30e9, 2024: 32e9})}},
            'StockholdersEquity':                         {'units': {'USD': f({2023: 60e9, 2024: 70e9})}},
            'Assets':                                     {'units': {'USD': f({2023: 150e9, 2024: 160e9})}},
        }
        return {'facts': {'us-gaap': gaap}}

    def test_cash_flow_frame_has_capex_and_da(self, monkeypatch):
        c = _make_client()
        monkeypatch.setattr(c, 'fetch_company_facts', lambda tk: self._facts())
        shape = c.build_yfinance_shape('TEST')
        cf = shape['cash_flow']
        latest = cf.iloc[:, 0]   # 2024 column first
        # yfinance sign convention: capex stored negative
        assert latest['Capital Expenditure'] == -4e9
        assert latest['Depreciation And Amortization'] == 2.5e9
        assert latest['Operating Cash Flow'] == 16e9

    def test_balance_frame_has_current_rows(self, monkeypatch):
        c = _make_client()
        monkeypatch.setattr(c, 'fetch_company_facts', lambda tk: self._facts())
        shape = c.build_yfinance_shape('TEST')
        latest_bs = shape['balance_sheet'].iloc[:, 0]
        assert latest_bs['Current Assets'] == 45e9
        assert latest_bs['Current Liabilities'] == 32e9

    def test_fundamental_growth_computes_from_shape(self, monkeypatch):
        from models.ratios import calculate_fundamental_growth
        c = _make_client()
        monkeypatch.setattr(c, 'fetch_company_facts', lambda tk: self._facts())
        shape = c.build_yfinance_shape('TEST')
        fg = calculate_fundamental_growth(shape, roic_override=0.15)
        assert fg, 'fundamental growth must compute once Capex/D&A rows exist'
        # RR = (capex - da + dWC) / NOPAT = (4e9 - 2.5e9 + (13e9 - 10e9)) / (18e9*(1-0.21))
        assert 0 < fg['fundamental_growth'] <= 0.30
        assert fg['roic_used'] == 0.15

    def test_missing_capex_da_leaves_rows_none(self, monkeypatch):
        """Filers without capex/D&A tags degrade to None rows, not crashes."""
        facts = self._facts()
        del facts['facts']['us-gaap']['PaymentsToAcquirePropertyPlantAndEquipment']
        del facts['facts']['us-gaap']['DepreciationDepletionAndAmortization']
        c = _make_client()
        monkeypatch.setattr(c, 'fetch_company_facts', lambda tk: facts)
        shape = c.build_yfinance_shape('TEST')
        latest = shape['cash_flow'].iloc[:, 0]
        import pandas as pd
        assert pd.isna(latest['Capital Expenditure'])
        assert pd.isna(latest['Depreciation And Amortization'])


class TestCompanyFactsNotFound:
    """Foreign ADR/12g3-2(b) CIKs 404 on companyfacts every night: remembered
    for the run, while transient failures are still retried."""

    def _patch_urlopen(self, monkeypatch, exc):
        calls = []

        def fake_urlopen(req, context=None, timeout=None):
            calls.append(req.full_url)
            raise exc

        monkeypatch.setattr(xbrl_mod.urllib.request, 'urlopen', fake_urlopen)
        return calls

    def test_404_is_cached_for_the_run(self, monkeypatch):
        import urllib.error
        c = _make_client()
        calls = self._patch_urlopen(monkeypatch, urllib.error.HTTPError(
            'https://data.sec.gov/x', 404, 'Not Found', {}, None))
        assert c.fetch_company_facts('TEST') is None
        assert c.fetch_company_facts('TEST') is None
        assert len(calls) == 1
        assert c.facts_stats['not_found'] == 1 and c.facts_stats['failures'] == 0
        assert c.facts_stats['mem_hits'] == 1

    def test_404_never_reaches_the_disk_cache(self, monkeypatch, tmp_path):
        import urllib.error
        c = SECXBRLClient(cik_map={'TEST': '0000000001'}, name_map={}, email='t@e.com',
                          request_delay=0, facts_cache=str(tmp_path))
        self._patch_urlopen(monkeypatch, urllib.error.HTTPError(
            'https://data.sec.gov/x', 404, 'Not Found', {}, None))
        assert c.fetch_company_facts('TEST') is None
        assert c._facts_cache.get('0000000001') is None
        assert not any(p.is_file() for p in tmp_path.rglob('*.json.gz'))

    def test_timeout_is_retried(self, monkeypatch):
        c = _make_client()
        calls = self._patch_urlopen(monkeypatch, TimeoutError('timed out'))
        assert c.fetch_company_facts('TEST') is None
        assert c.fetch_company_facts('TEST') is None
        assert len(calls) == 2
        assert c.facts_stats['failures'] == 2 and c.facts_stats['not_found'] == 0

    def test_5xx_is_retried(self, monkeypatch):
        import urllib.error
        c = _make_client()
        calls = self._patch_urlopen(monkeypatch, urllib.error.HTTPError(
            'https://data.sec.gov/x', 503, 'Unavailable', {}, None))
        assert c.fetch_company_facts('TEST') is None
        assert c.fetch_company_facts('TEST') is None
        assert len(calls) == 2


class TestPredecessorCik:
    """XOM: SEC maps the ticker to a 2026 holding-company registrant with no
    history; the predecessor's facts stand in until the successor has 2 years."""

    SUCC, PRED = '0002115436', '0000034088'

    def _client(self, monkeypatch, blobs, facts_cache=None):
        c = SECXBRLClient(cik_map={'XOM': self.SUCC}, name_map={}, email='t@e.com',
                          request_delay=0, facts_cache=facts_cache,
                          cik_predecessors={self.SUCC: self.PRED})
        calls = []

        def fake_request(url, timeout=20, absent_codes=()):
            calls.append(url)
            for cik, blob in blobs.items():
                if f'CIK{cik}' in url:
                    return blob
            raise AssertionError(url)

        monkeypatch.setattr(c, '_request_json', fake_request)
        return c, calls

    def test_thin_successor_uses_predecessor(self, monkeypatch):
        thin = {'cik': 2115436, 'facts': {'dei': {}}}
        full = _us_gaap_revenue_facts({2023: 344e9, 2024: 339e9, 2025: 320e9})
        c, calls = self._client(monkeypatch, {self.SUCC: thin, self.PRED: full})
        facts = c.fetch_company_facts('XOM')
        assert facts['_predecessor_cik'] == self.PRED
        assert 'Revenues' in facts['facts']['us-gaap']
        assert '_predecessor_cik' not in full          # the source blob is not mutated
        h = c.fetch_historical_financials('XOM')
        assert h['predecessor_cik'] == self.PRED
        assert h['revenue_history'] == {2023: 344e9, 2024: 339e9, 2025: 320e9}
        assert len(calls) == 2                          # second read is a memory hit

    def test_one_year_successor_is_still_thin(self, monkeypatch):
        succ = _us_gaap_revenue_facts({2026: 80e9})
        full = _us_gaap_revenue_facts({2024: 339e9, 2025: 320e9})
        c, _ = self._client(monkeypatch, {self.SUCC: succ, self.PRED: full})
        assert c.fetch_company_facts('XOM')['_predecessor_cik'] == self.PRED

    def test_full_successor_is_preferred(self, monkeypatch):
        succ = _us_gaap_revenue_facts({2025: 320e9, 2026: 330e9})
        c, calls = self._client(monkeypatch, {self.SUCC: succ, self.PRED: {}})
        facts = c.fetch_company_facts('XOM')
        assert facts is succ and '_predecessor_cik' not in facts
        assert len(calls) == 1
        assert 'predecessor_cik' not in c.fetch_historical_financials('XOM')

    def test_successor_404_uses_predecessor(self, monkeypatch):
        full = _us_gaap_revenue_facts({2024: 339e9, 2025: 320e9})
        c, _ = self._client(monkeypatch, {self.SUCC: xbrl_mod.ABSENT, self.PRED: full})
        assert c.fetch_company_facts('XOM')['_predecessor_cik'] == self.PRED

    def test_disk_cache_keeps_each_cik_untagged(self, monkeypatch, tmp_path):
        thin = {'facts': {}}
        full = _us_gaap_revenue_facts({2024: 339e9, 2025: 320e9})
        c, _ = self._client(monkeypatch, {self.SUCC: thin, self.PRED: full},
                            facts_cache=str(tmp_path))
        assert c.fetch_company_facts('XOM')['_predecessor_cik'] == self.PRED
        assert '_predecessor_cik' not in c._facts_cache.get(self.PRED)
        assert c._facts_cache.get(self.SUCC) == thin

    def test_no_override_leaves_thin_successor(self, monkeypatch):
        thin = {'facts': {}}
        c, calls = self._client(monkeypatch, {self.SUCC: thin})
        c._cik_predecessors = {}
        assert c.fetch_company_facts('XOM') is thin and len(calls) == 1

    def test_config_maps_xom(self):
        from scripts.config import SEC_CIK_PREDECESSORS
        assert SEC_CIK_PREDECESSORS[self.SUCC] == self.PRED
