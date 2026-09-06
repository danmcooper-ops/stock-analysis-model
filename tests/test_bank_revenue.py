# tests/test_bank_revenue.py
"""Bank revenue = net interest income + noninterest income, and the
Financial Services mask on Moat: Margin Advantage.

Banks tag no revenue total in the alias list: their ASC 606 line is fee
income only. In the 2026-09-03 snapshot 144 of 224 banks had no revenue
(Rev Durability, Rev Volatility, SBC/Rev and Margin Advantage all N/A) and
the other 80 carried the fee slice, which read as a median operating margin
of 174% and let 78 of them pass Margin Advantage. Synthetic companyfacts
stubs; no live SEC API calls.
"""

import pytest


from data.sec_xbrl_client import SECXBRLClient
from scripts.scoring import apply_screening_matrix

YEARS = (2021, 2022, 2023, 2024)


def _make_client():
    return SECXBRLClient(cik_map={'TEST': '0000000001'}, name_map={'TEST': 'Test Co'},
                         email='test@example.com', request_delay=0)


def _entries(years_values):
    return [{'form': '10-K', 'fy': fy, 'fp': 'FY', 'val': val,
             'filed': f'{fy + 1}-01-15',
             'start': f'{fy}-01-01', 'end': f'{fy}-12-31'}
            for fy, val in years_values.items()]


def _facts(concepts):
    return {'facts': {'us-gaap': {
        tag: {'units': {'USD': _entries(vals)}} for tag, vals in concepts.items()
    }}}


BANK = {
    'NetIncomeLoss':            {y: 150.0 for y in YEARS},
    'InterestIncomeExpenseNet': {y: 600.0 + 10 * i for i, y in enumerate(YEARS)},
    'NoninterestIncome':        {y: 200.0 for y in YEARS},
    # The ASC 606 fee slice a bank tags as its only "revenue" line.
    'RevenueFromContractWithCustomerExcludingAssessedTax':
                                {y: 40.0 for y in YEARS},
}


class TestBankRevenueResolution:
    def test_history_uses_net_revenue_not_fee_slice(self):
        c = _make_client(); c._cache['TEST'] = _facts(BANK)
        h = c.fetch_historical_financials('TEST')
        assert h['revenue_history'] == {2021: 800.0, 2022: 810.0, 2023: 820.0, 2024: 830.0}

    def test_shape_total_revenue_row_matches(self):
        c = _make_client(); c._cache['TEST'] = _facts(BANK)
        inc = c.build_yfinance_shape('TEST')['income_statement']
        assert inc.iloc[:, 0]['Total Revenue'] == 830.0

    def test_bank_with_no_fee_slice_still_gets_revenue(self):
        facts = {k: v for k, v in BANK.items()
                 if k != 'RevenueFromContractWithCustomerExcludingAssessedTax'}
        c = _make_client(); c._cache['TEST'] = _facts(facts)
        h = c.fetch_historical_financials('TEST')
        assert h['revenue_history'][2024] == 830.0

    def test_tagged_revenues_equal_to_sum_is_kept(self):
        """JPM-shaped: Revenues is already NII + noninterest income."""
        facts = dict(BANK, Revenues={y: 800.0 + 10 * i for i, y in enumerate(YEARS)})
        c = _make_client(); c._cache['TEST'] = _facts(facts)
        h = c.fetch_historical_financials('TEST')
        assert h['revenue_history'][2024] == 830.0

    def test_larger_tagged_total_is_not_reduced(self):
        """A filer whose Revenues line is gross (before interest expense)
        keeps it: the derivation only ever raises a component."""
        facts = dict(BANK, Revenues={y: 1500.0 for y in YEARS})
        c = _make_client(); c._cache['TEST'] = _facts(facts)
        h = c.fetch_historical_financials('TEST')
        assert h['revenue_history'][2024] == 1500.0

    def test_partial_components_leave_other_years_alone(self):
        facts = dict(BANK)
        facts['NoninterestIncome'] = {2024: 200.0}
        c = _make_client(); c._cache['TEST'] = _facts(facts)
        h = c.fetch_historical_financials('TEST')
        assert h['revenue_history'][2024] == 830.0
        assert h['revenue_history'][2023] == 40.0     # only the fee slice exists

    def test_non_bank_untouched(self):
        c = _make_client(); c._cache['TEST'] = _facts({
            'Revenues':      {y: 1000.0 for y in YEARS},
            'NetIncomeLoss': {y: 150.0 for y in YEARS},
            'InterestIncomeExpenseNet': {y: 5.0 for y in YEARS},   # no NoninterestIncome
        })
        h = c.fetch_historical_financials('TEST')
        assert h['revenue_history'] == {y: 1000.0 for y in YEARS}


class TestMarginAdvantageMaskedForFinancials:
    def test_bank_margin_advantage_inapplicable(self):
        bank = {'ticker': 'BANK', 'sector': 'Financial Services',
                'operating_margin': 1.74, '_sector_median_opm': 0.23}
        tech = {'ticker': 'TECH', 'sector': 'Technology',
                'operating_margin': 0.30, '_sector_median_opm': 0.20}
        apply_screening_matrix([bank, tech])
        assert bank['_gate_margin_advantage'] is None
        assert bank['_gp_margin_advantage'] is None
        assert tech['_gate_margin_advantage'] == pytest.approx(0.10)
        assert tech['_gp_margin_advantage'] is True
