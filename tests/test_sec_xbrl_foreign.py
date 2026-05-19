# tests/test_sec_xbrl_foreign.py
"""Foreign-issuer support: 20-F / 40-F filings, IFRS taxonomy, FX conversion.

These tests use synthetic XBRL fact stubs to avoid live SEC API calls.
"""
import sys
import os

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
