# tests/test_wacc_inputs.py
"""Inputs feeding WACC: the XBRL interest-expense tag, the Yahoo-beta CAPM
fallback and the equity-model discount-rate bounds.

Synthetic companyfacts stubs; no live SEC or Yahoo calls.
"""
import math

import pytest

from data.sec_xbrl_client import SECXBRLClient
from models.ratios import calculate_wacc
from scripts.analyze_stock import select_cost_of_equity, _bound_re_for_models
from scripts.config import RE_CAP_SPREAD, SECTOR_CONFIG, SECTOR_DEFAULT


def _annual_entries(years_values):
    return [{
        'form': '10-K', 'fy': fy, 'fp': 'FY', 'val': val,
        'filed': f'{fy + 1}-01-15',
        'start': f'{fy}-01-01', 'end': f'{fy}-12-31',
    } for fy, val in years_values.items()]


def _client_with(concepts):
    c = SECXBRLClient(cik_map={'TEST': '0000000001'}, name_map={'TEST': 'Test Co'},
                      email='test@example.com', request_delay=0)
    c._cache['TEST'] = {'facts': {'us-gaap': {
        tag: {'units': {'USD': _annual_entries(v)}} for tag, v in concepts.items()
    }}}
    return c


YEARS = (2022, 2023, 2024, 2025)
_BASE = {
    'Revenues':              {y: 1000.0 for y in YEARS},
    'NetIncomeLoss':         {y: 150.0 for y in YEARS},
    'OperatingIncomeLoss':   {y: 200.0 for y in YEARS},
    'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest':
                             {y: 190.0 for y in YEARS},
    'IncomeTaxExpenseBenefit': {y: 40.0 for y in YEARS},
    'StockholdersEquity':    {y: 800.0 for y in YEARS},
    'Assets':                {y: 2000.0 for y in YEARS},
    'LongTermDebtNoncurrent': {y: 400.0 for y in YEARS},
    'DebtCurrent':           {y: 100.0 for y in YEARS},
}


class TestXbrlInterestExpenseTag:
    def test_2024_taxonomy_tag_populates_latest_year(self):
        """Filers that moved to InterestExpenseNonoperating for FY2024+ must
        not leave the latest-year row NaN (which silently swapped the real
        cost of debt for Rf + credit spread)."""
        c = _client_with({
            **_BASE,
            'InterestExpense':             {2022: 20.0, 2023: 21.0},
            'InterestExpenseNonoperating': {2024: 22.0, 2025: 23.0},
        })
        inc = c.build_yfinance_shape('TEST')['income_statement']
        row = inc.loc['Interest Expense']
        assert row.iloc[0] == 23.0
        assert row.iloc[1] == 22.0
        assert row.iloc[2] == 21.0

    def test_legacy_only_filer_unchanged(self):
        c = _client_with({**_BASE, 'InterestExpense': {y: 20.0 for y in YEARS}})
        inc = c.build_yfinance_shape('TEST')['income_statement']
        assert inc.loc['Interest Expense'].iloc[0] == 20.0

    def test_wacc_uses_the_real_rate(self):
        c = _client_with({**_BASE,
                          'InterestExpenseNonoperating': {y: 30.0 for y in YEARS}})
        shape = c.build_yfinance_shape('TEST')
        shape['info'] = {'marketCap': 4500.0}
        wacc = calculate_wacc(shape, 0.10, risk_free_rate=0.04)
        # E/V = 0.9, D/V = 0.1, Kd = 30/500 = 6%, T = 40/190
        expected = 0.9 * 0.10 + 0.1 * 0.06 * (1 - 40 / 190)
        assert wacc == pytest.approx(expected, rel=1e-6)


class TestYahooBetaFallback:
    def test_blume_adjusted(self):
        """Path 2 (Yahoo beta) is Blume-adjusted like path 1."""
        fin = {'info': {'beta': 1.5}}
        re, label, diag = select_cost_of_equity(fin, 0.04, erp=0.05)
        adj = (2 / 3) * 1.5 + (1 / 3)
        assert label == 'capm (yahoo beta)'
        assert re == pytest.approx(0.04 + adj * 0.05)
        assert diag['beta_source'] == 'yahoo'
        assert diag['shrunk_beta'] == pytest.approx(adj)

    def test_beta_of_one_unchanged(self):
        fin = {'info': {'beta': 1.0}}
        re, _, _ = select_cost_of_equity(fin, 0.04, erp=0.05)
        assert re == pytest.approx(0.09)

    def test_ggm_path_no_double_count(self):
        fin = {'info': {'dividendRate': 3.0, 'currentPrice': 100.0}}
        re, label, _ = select_cost_of_equity(fin, 0.04, erp=0.05)
        assert label == 'ggm'
        assert re == pytest.approx(0.03 + 0.03)


class TestBoundReForModels:
    def test_cap_sits_above_wacc_cap(self):
        cfg = SECTOR_CONFIG['Technology']
        assert _bound_re_for_models(0.20, 'Technology') == pytest.approx(
            cfg['wacc_cap'] + RE_CAP_SPREAD)

    def test_high_betas_no_longer_collapse(self):
        """Beta 1.6 vs beta 2.0 tech names got the same 13% before."""
        rf, erp = 0.0425, 0.055
        lo = _bound_re_for_models(rf + 1.6 * erp, 'Technology')
        hi = _bound_re_for_models(rf + 2.0 * erp, 'Technology')
        assert hi > lo

    def test_floor_shared_with_wacc(self):
        assert _bound_re_for_models(0.02, 'Utilities') == pytest.approx(
            SECTOR_CONFIG['Utilities']['wacc_floor'])

    def test_inside_band_untouched(self):
        assert _bound_re_for_models(0.10, 'Unknown Sector') == pytest.approx(0.10)
        assert SECTOR_DEFAULT['wacc_floor'] < 0.10 < SECTOR_DEFAULT['wacc_cap']

    def test_none_passthrough(self):
        assert _bound_re_for_models(None, 'Technology') is None
        assert math.isfinite(_bound_re_for_models(0.5, 'Energy'))
