"""EV→common-equity bridge used by the forward and reverse DCF.

Regression tests for the September DCF accuracy review:

1. Net debt read plain cash ahead of the cash + short-term-investments
   total, so mega-caps that park liquidity in marketable securities had
   net debt overstated by the whole securities book.
2. The bridge stopped at net debt — minority interest and preferred stock
   rank ahead of common equity and must be subtracted too.
3. The SEC XBRL shape only emitted plain cash, so the fix above was dead on
   the path every US filer takes; it now composes the combined row from
   cash + short-term investments and emits minority interest / preferred.
"""

import pandas as pd
import pytest

from data.sec_xbrl_client import SECXBRLClient
from models.dcf import reverse_dcf
from models.field_keys import CASH_KEYS
from models.quality import (get_net_debt, get_non_common_claims,
                            get_ev_to_equity_bridge)
from scripts.analyze_stock import run_forward_dcf


def _bs(**rows):
    col = pd.Timestamp('2024-12-31')
    return pd.DataFrame({col: rows})


class TestNetDebtLiquidity:
    def test_cash_keys_prefer_combined_total(self):
        assert CASH_KEYS[0] == 'Cash Cash Equivalents And Short Term Investments'

    def test_short_term_investments_reduce_net_debt(self):
        fin = {'balance_sheet': _bs(**{
            'Total Debt': 100.0,
            'Cash And Cash Equivalents': 30.0,
            'Cash Cash Equivalents And Short Term Investments': 90.0,
        })}
        assert get_net_debt(fin) == pytest.approx(10.0)

    def test_plain_cash_still_used_when_combined_absent(self):
        fin = {'balance_sheet': _bs(**{'Total Debt': 100.0,
                                       'Cash And Cash Equivalents': 30.0})}
        assert get_net_debt(fin) == pytest.approx(70.0)


class TestNonCommonClaims:
    def test_none_without_balance_sheet(self):
        assert get_non_common_claims({}) is None
        assert get_ev_to_equity_bridge({'balance_sheet': pd.DataFrame()}) is None

    def test_zero_when_lines_absent(self):
        fin = {'balance_sheet': _bs(**{'Total Debt': 10.0})}
        assert get_non_common_claims(fin) == 0
        assert get_ev_to_equity_bridge(fin) == pytest.approx(10.0)

    def test_bridge_adds_minority_and_preferred(self):
        fin = {'balance_sheet': _bs(**{
            'Total Debt': 100.0,
            'Cash And Cash Equivalents': 40.0,
            'Minority Interest': 25.0,
            'Preferred Stock Equity': 15.0,
        })}
        assert get_non_common_claims(fin) == pytest.approx(40.0)
        assert get_ev_to_equity_bridge(fin) == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Forward DCF uses the full bridge; reverse DCF solves on the same basis
# ---------------------------------------------------------------------------

YEARS = pd.to_datetime(['2021-12-31', '2022-12-31', '2023-12-31', '2024-12-31'])


def _yf_data(extra_bs=None):
    cf = pd.DataFrame({y: {'Free Cash Flow': 1000.0,
                           'Operating Cash Flow': 1300.0,
                           'Depreciation And Amortization': 300.0,
                           'Capital Expenditure': -300.0,
                           'Stock Based Compensation': 0.0}
                       for y in YEARS})
    inc = pd.DataFrame({y: {'Total Revenue': 10_000.0, 'Operating Income': 1200.0}
                        for y in YEARS})
    bs_rows = {'Total Debt': 2000.0, 'Cash And Cash Equivalents': 500.0}
    bs_rows.update(extra_bs or {})
    bs = pd.DataFrame({YEARS[-1]: bs_rows})
    info = {'marketCap': 20_000.0, 'sharesOutstanding': 1_000.0, 'currentPrice': 20.0}
    return {'cash_flow': cf, 'income_statement': inc, 'balance_sheet': bs, 'info': info}


class TestForwardDcfBridge:
    def test_minority_and_preferred_lower_fair_value(self):
        base, *_ = run_forward_dcf(_yf_data(), wacc=0.10)
        claims, *_ = run_forward_dcf(
            _yf_data({'Minority Interest': 300.0, 'Preferred Stock Equity': 200.0}),
            wacc=0.10)
        # 500 of claims over 1000 shares = exactly $0.50/share less.
        assert base - claims == pytest.approx(0.5, abs=1e-9)

    def test_short_term_investments_raise_fair_value(self):
        base, *_ = run_forward_dcf(_yf_data(), wacc=0.10)
        rich, *_ = run_forward_dcf(
            _yf_data({'Cash Cash Equivalents And Short Term Investments': 1500.0}),
            wacc=0.10)
        assert rich - base == pytest.approx(1.0, abs=1e-9)


class TestReverseDcfMatchesHeadline:
    @pytest.mark.parametrize('exit_multiple', [None, 12.0])
    def test_price_at_fair_value_recovers_estimated_growth(self, exit_multiple):
        yf = _yf_data({'Minority Interest': 300.0})
        fv, _, fcf_growth, diag, _ = run_forward_dcf(yf, wacc=0.10, exit_multiple=exit_multiple)
        assert fv and fv > 0
        if exit_multiple:
            assert diag['exit_multiple'] == exit_multiple and diag['base_ebitda'] > 0
        else:
            assert diag['exit_multiple'] is None
        rev = reverse_dcf(fv, diag['base_fcf'], 0.10, 1000.0, diag['equity_bridge'],
                          terminal_g=diag['term_g'],
                          base_ebitda=diag['base_ebitda'],
                          exit_multiple=diag['exit_multiple'])
        assert rev['converged']
        assert rev['implied_growth'] == pytest.approx(fcf_growth, abs=1e-5)


# ---------------------------------------------------------------------------
# SEC XBRL shape emits the rows the bridge reads
# ---------------------------------------------------------------------------

def _pit(fy, val):
    return {'form': '10-K', 'fy': fy, 'fp': 'FY', 'val': val,
            'filed': f'{fy + 1}-02-01', 'end': f'{fy}-12-31'}


def _flow(fy, val):
    return {'form': '10-K', 'fy': fy, 'fp': 'FY', 'val': val,
            'filed': f'{fy + 1}-02-01', 'start': f'{fy}-01-01', 'end': f'{fy}-12-31'}


def _client_with(tag_entries):
    c = SECXBRLClient(cik_map={'TEST': '0000000001'}, name_map={'TEST': 'Test Co'},
                      email='test@example.com', request_delay=0)
    c._cache['TEST'] = {'facts': {'us-gaap': {
        tag: {'units': {'USD': entries}} for tag, entries in tag_entries.items()}}}
    return c


class TestXbrlBridgeRows:
    def test_combined_cash_composed_from_separate_tags(self):
        c = _client_with({
            'Revenues': [_flow(2024, 1000.0)],
            'CashAndCashEquivalentsAtCarryingValue': [_pit(2024, 30.0)],
            'MarketableSecuritiesCurrent': [_pit(2024, 60.0)],
            'LongTermDebtNoncurrent': [_pit(2024, 100.0)],
            'MinorityInterest': [_pit(2024, 7.0)],
            'PreferredStockValue': [_pit(2024, 3.0)],
        })
        shape = c.build_yfinance_shape('TEST')
        latest = shape['balance_sheet'].iloc[:, 0]
        assert latest['Cash And Cash Equivalents'] == 30.0
        assert latest['Cash Cash Equivalents And Short Term Investments'] == 90.0
        assert latest['Minority Interest'] == 7.0
        assert latest['Preferred Stock Equity'] == 3.0
        assert get_net_debt(shape) == pytest.approx(10.0)
        assert get_ev_to_equity_bridge(shape) == pytest.approx(20.0)

    def test_combined_tag_alone_fills_both_rows(self):
        c = _client_with({
            'Revenues': [_flow(2024, 1000.0)],
            'CashCashEquivalentsAndShortTermInvestments': [_pit(2024, 80.0)],
            'LongTermDebtNoncurrent': [_pit(2024, 100.0)],
        })
        latest = c.build_yfinance_shape('TEST')['balance_sheet'].iloc[:, 0]
        assert latest['Cash And Cash Equivalents'] == 80.0
        assert latest['Cash Cash Equivalents And Short Term Investments'] == 80.0

    def test_plain_cash_alone_gives_equal_rows(self):
        c = _client_with({
            'Revenues': [_flow(2024, 1000.0)],
            'CashAndCashEquivalentsAtCarryingValue': [_pit(2024, 30.0)],
        })
        latest = c.build_yfinance_shape('TEST')['balance_sheet'].iloc[:, 0]
        assert latest['Cash Cash Equivalents And Short Term Investments'] == 30.0
        assert pd.isna(latest['Minority Interest'])
