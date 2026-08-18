# tests/test_debt_resolution.py
"""Total-debt resolution: component tags, priority ladder, frame + history.

Synthetic companyfacts stubs throughout — no live SEC calls. Covers the
2026-07 debt-layer fix: the old single-concept 'total_debt' read long-term
tags only and understated leverage by the short-term portion for every
XBRL-sourced ticker.
"""
import sys
import os

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data import sec_xbrl_client as xbrl_mod
from data.sec_xbrl_client import SECXBRLClient
from models.ratios import compute_ratios
from models.quality import calculate_altman_z, get_net_debt
from scripts.backfill_edgar_hist import _needs_refetch


def _make_client():
    return SECXBRLClient(
        cik_map={'TEST': '0000000001'},
        name_map={'TEST': 'Test Co'},
        email='test@example.com',
        request_delay=0,
    )


def _pit(fy, val, form='10-K'):
    """Point-in-time (balance sheet) entry: end date only, no duration."""
    return {'form': form, 'fy': fy, 'fp': 'FY', 'val': val,
            'filed': f'{fy + 1}-02-01', 'end': f'{fy}-12-31'}


def _flow(fy, val, form='10-K'):
    """Duration (income/cash-flow) entry spanning the fiscal year."""
    return {'form': form, 'fy': fy, 'fp': 'FY', 'val': val,
            'filed': f'{fy + 1}-02-01',
            'start': f'{fy}-01-01', 'end': f'{fy}-12-31'}


def _facts(tag_entries, taxonomy='us-gaap', currency='USD'):
    """Build a companyfacts stub: {tag: [entries]} -> full JSON shape."""
    return {'facts': {taxonomy: {
        tag: {'units': {currency: entries}}
        for tag, entries in tag_entries.items()
    }}}


# ---------------------------------------------------------------------------
# Resolver priority ladder
# ---------------------------------------------------------------------------

class TestResolverTiers:
    def test_tier1_noncurrent_plus_debt_current(self):
        c = _make_client()
        facts = _facts({
            'LongTermDebtNoncurrent': [_pit(2024, 80.0)],
            'DebtCurrent': [_pit(2024, 20.0)],
        })
        vals, tagged = c._resolve_total_debt_annual(facts)
        assert tagged is True
        assert vals == {2024: 100.0}

    def test_tier2_component_sum(self):
        c = _make_client()
        facts = _facts({
            'LongTermDebtNoncurrent': [_pit(2024, 70.0)],
            'LongTermDebtCurrent': [_pit(2024, 10.0)],
            'ShortTermBorrowings': [_pit(2024, 5.0)],
        })
        vals, _ = c._resolve_total_debt_annual(facts)
        assert vals == {2024: 85.0}

    def test_tier3_ltd_total_plus_commercial_paper(self):
        c = _make_client()
        facts = _facts({
            'LongTermDebt': [_pit(2024, 90.0)],
            'CommercialPaper': [_pit(2024, 7.0)],
        })
        vals, _ = c._resolve_total_debt_annual(facts)
        assert vals == {2024: 97.0}

    def test_no_double_count_when_both_ltd_forms_tagged(self):
        # LongTermDebt INCLUDES current maturities; a filer tagging both it
        # and the noncurrent/current split must resolve via the split (tier
        # 1/2), never LongTermDebt + current again.
        c = _make_client()
        facts = _facts({
            'LongTermDebt': [_pit(2024, 100.0)],           # 90 nc + 10 cur
            'LongTermDebtNoncurrent': [_pit(2024, 90.0)],
            'LongTermDebtCurrent': [_pit(2024, 10.0)],
        })
        vals, _ = c._resolve_total_debt_annual(facts)
        assert vals == {2024: 100.0}   # not 110

    def test_untagged_returns_empty_and_untagged_flag(self):
        c = _make_client()
        facts = _facts({'Revenues': [_flow(2024, 500.0)]})
        vals, tagged = c._resolve_total_debt_annual(facts)
        assert vals == {}
        assert tagged is False

    def test_tagged_but_unresolvable_year_is_omitted(self):
        # Only a current-debt component in 2023: leverage UNKNOWN for that
        # year (not "current debt only") — omit, don't understate.
        c = _make_client()
        facts = _facts({
            'LongTermDebtNoncurrent': [_pit(2024, 80.0)],
            'DebtCurrent': [_pit(2024, 20.0), _pit(2023, 15.0)],
        })
        vals, tagged = c._resolve_total_debt_annual(facts)
        assert tagged is True
        assert 2023 not in vals
        assert vals == {2024: 100.0}

    def test_ifrs_borrowings_via_concept_resolver(self):
        c = _make_client()
        facts = _facts({
            'NoncurrentBorrowings': [_pit(2024, 60.0, form='20-F')],
            'CurrentBorrowings': [_pit(2024, 15.0, form='20-F')],
        }, taxonomy='ifrs-full', currency='EUR')
        vals, tax, ccy, tagged = c._resolve_total_debt_concept(facts)
        assert tax == 'ifrs-full'
        assert ccy == 'EUR'
        assert tagged is True
        assert vals == {2024: 75.0}


# ---------------------------------------------------------------------------
# build_yfinance_shape: frame rows
# ---------------------------------------------------------------------------

def _full_facts(with_debt=True, with_liabilities=True):
    tags = {
        'Revenues': [_flow(2023, 900.0), _flow(2024, 1000.0)],
        'NetIncomeLoss': [_flow(2023, 90.0), _flow(2024, 100.0)],
        'OperatingIncomeLoss': [_flow(2023, 120.0), _flow(2024, 130.0)],
        'Assets': [_pit(2023, 950.0), _pit(2024, 1000.0)],
        'StockholdersEquity': [_pit(2023, 380.0), _pit(2024, 400.0)],
        'AssetsCurrent': [_pit(2024, 300.0)],
        'LiabilitiesCurrent': [_pit(2024, 150.0)],
        'CashAndCashEquivalentsAtCarryingValue': [_pit(2024, 50.0)],
        'RetainedEarningsAccumulatedDeficit': [_pit(2024, 220.0)],
    }
    if with_debt:
        tags['LongTermDebtNoncurrent'] = [_pit(2023, 180.0), _pit(2024, 200.0)]
        tags['DebtCurrent'] = [_pit(2023, 30.0), _pit(2024, 40.0)]
    if with_liabilities:
        tags['Liabilities'] = [_pit(2024, 600.0)]
    return _facts(tags)


def _shape(facts):
    c = _make_client()
    c._cache['TEST'] = facts
    return c.build_yfinance_shape('TEST')


class TestBuildYfinanceShape:
    def test_total_debt_row_includes_current_portion(self):
        shape = _shape(_full_facts())
        bs = shape['balance_sheet']
        latest = bs.iloc[:, 0]
        assert latest['Total Debt'] == 240.0   # 200 nc + 40 current

    def test_total_liabilities_row_from_tag(self):
        shape = _shape(_full_facts())
        latest = shape['balance_sheet'].iloc[:, 0]
        assert latest['Total Liabilities Net Minority Interest'] == 600.0

    def test_total_liabilities_falls_back_to_assets_minus_equity(self):
        shape = _shape(_full_facts(with_liabilities=False))
        latest = shape['balance_sheet'].iloc[:, 0]
        assert latest['Total Liabilities Net Minority Interest'] == 600.0  # 1000-400

    def test_retained_earnings_row_present(self):
        shape = _shape(_full_facts())
        latest = shape['balance_sheet'].iloc[:, 0]
        assert latest['Retained Earnings'] == 220.0

    def test_unlevered_filer_gets_explicit_zero_debt(self):
        shape = _shape(_full_facts(with_debt=False))
        latest = shape['balance_sheet'].iloc[:, 0]
        assert latest['Total Debt'] == 0.0
        # Net debt reads -cash, not unknown.
        assert get_net_debt(shape) == -50.0

    def test_compute_ratios_de_is_true_debt_to_equity(self):
        shape = _shape(_full_facts())
        ratios = compute_ratios(shape)
        assert ratios['Debt-to-Equity'] == pytest.approx(240.0 / 400.0)

    def test_altman_z_computable_on_xbrl_frame(self):
        shape = _shape(_full_facts())
        shape['info']['marketCap'] = 2000.0
        z = calculate_altman_z(shape)
        assert z is not None
        # x4 uses market equity / total liabilities = 2000/600; sanity-check
        # the score lands in a plausible band rather than pinning exact math.
        assert 2.0 < z < 10.0


# ---------------------------------------------------------------------------
# fetch_historical_financials: new series
# ---------------------------------------------------------------------------

class TestHistoricalSeries:
    def test_debt_and_cash_history_keys_always_present(self):
        c = _make_client()
        c._cache['TEST'] = _full_facts()
        hist = c.fetch_historical_financials('TEST')
        assert 'total_debt_history' in hist
        assert 'cash_history' in hist
        assert hist['total_debt_history'] == {2023: 210.0, 2024: 240.0}
        assert hist['cash_history'] == {2024: 50.0}

    def test_unlevered_filer_zero_filled_over_revenue_span(self):
        c = _make_client()
        c._cache['TEST'] = _full_facts(with_debt=False)
        hist = c.fetch_historical_financials('TEST')
        assert hist['total_debt_history'] == {2023: 0.0, 2024: 0.0}

    def test_fx_applied_to_debt_and_cash(self, monkeypatch):
        c = _make_client()
        tags = {
            'Revenue': [_flow(2024, 1000.0, form='20-F')],
            'ProfitLoss': [_flow(2024, 100.0, form='20-F')],
            'NoncurrentBorrowings': [_pit(2024, 200.0, form='20-F')],
            'CurrentBorrowings': [_pit(2024, 50.0, form='20-F')],
            'CashAndCashEquivalents': [_pit(2024, 80.0, form='20-F')],
        }
        c._cache['TEST'] = _facts(tags, taxonomy='ifrs-full', currency='EUR')
        monkeypatch.setattr(xbrl_mod, '_get_fx_rates_to_usd',
                            lambda ccy: {2024: 1.1})
        hist = c.fetch_historical_financials('TEST')
        assert hist['fx_converted'] is True
        assert hist['total_debt_history'][2024] == pytest.approx(275.0)
        assert hist['cash_history'][2024] == pytest.approx(88.0)


# ---------------------------------------------------------------------------
# Backfill refetch detection
# ---------------------------------------------------------------------------

class TestBackfillDetect:
    _BASE = {
        'revenue_history': {'2024': 1.0},
        'earnings_history': {'2024': 1.0},
        'operating_income_history': {'2024': 1.0},
    }

    def test_missing_debt_key_triggers_refetch(self):
        assert _needs_refetch(dict(self._BASE)) is True

    def test_present_but_empty_debt_series_skips(self):
        eh = dict(self._BASE)
        eh['total_debt_history'] = {}   # debt-free filer — legitimate
        assert _needs_refetch(eh) is False

    def test_empty_history_still_triggers(self):
        assert _needs_refetch({}) is True
