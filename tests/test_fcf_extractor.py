# tests/test_fcf_extractor.py
"""Tests for the `_fcf_series_from_cashflow` helper in analyze_stock.py.

This helper exists because yfinance's API stopped consistently surfacing
the synthetic 'Free Cash Flow' row for many large-cap tickers after its
2023 rewrite. Without an OCF + Capex fallback the DCF stage silently
skipped ~80% of the universe (including every mega-cap).
"""

import pandas as pd
import pytest


from scripts.analyze_stock import _fcf_series_from_cashflow


def _cf(rows, periods=None):
    """Build a yfinance-shaped cashflow DataFrame from {row_name: [vals]}."""
    if periods is None:
        periods = ['2023-12-31', '2024-12-31', '2025-12-31'][: len(next(iter(rows.values())))]
    return pd.DataFrame(rows, index=periods).T  # rows-as-index, columns-as-periods


class TestFcfSeriesFromCashflow:

    # --- Preferred path: pre-computed 'Free Cash Flow' row -------------

    def test_uses_pre_computed_free_cash_flow_when_present(self):
        cf = _cf({'Free Cash Flow': [100.0, 110.0, 121.0]})
        fcf = _fcf_series_from_cashflow(cf)
        assert fcf is not None
        assert list(fcf.values) == [100.0, 110.0, 121.0]

    def test_dropna_inside_free_cash_flow_row(self):
        cf = _cf({'Free Cash Flow': [100.0, float('nan'), 121.0]})
        fcf = _fcf_series_from_cashflow(cf)
        assert fcf is not None
        assert len(fcf) == 2
        assert sorted(fcf.values.tolist()) == [100.0, 121.0]

    # --- Fallback path: derive from OCF + Capex ------------------------

    def test_derives_from_ocf_plus_capex_when_fcf_missing(self):
        # Capex is signed negative in yfinance, so OCF + Capex = FCF.
        cf = _cf({
            'Operating Cash Flow': [200.0, 220.0, 250.0],
            'Capital Expenditure': [-60.0, -70.0, -80.0],
        })
        fcf = _fcf_series_from_cashflow(cf)
        assert fcf is not None
        assert list(fcf.values) == [140.0, 150.0, 170.0]

    def test_derives_using_alternative_ocf_key(self):
        # 'Cash From Operations' is the second OPERATING_CF_KEYS entry.
        cf = _cf({
            'Cash From Operations': [200.0, 220.0],
            'Capital Expenditure': [-60.0, -70.0],
        })
        fcf = _fcf_series_from_cashflow(cf)
        assert fcf is not None
        assert list(fcf.values) == [140.0, 150.0]

    def test_derives_using_alternative_capex_key(self):
        cf = _cf({
            'Operating Cash Flow': [200.0, 220.0],
            'Purchase Of Ppe': [-60.0, -70.0],
        })
        fcf = _fcf_series_from_cashflow(cf)
        assert fcf is not None
        assert list(fcf.values) == [140.0, 150.0]

    def test_aapl_shaped_case_no_fcf_row_but_full_ocf_and_capex(self):
        # Reproduces the AAPL pattern that was silently broken: yfinance
        # returns OCF and Capex but not 'Free Cash Flow'.
        cf = _cf({
            'Operating Cash Flow': [110_000e6, 118_000e6, 125_000e6],
            'Capital Expenditure': [-10_700e6, -11_200e6, -11_800e6],
        })
        fcf = _fcf_series_from_cashflow(cf)
        assert fcf is not None
        assert fcf.iloc[-1] == pytest.approx(113_200e6)

    # --- Edge / failure cases ------------------------------------------

    def test_returns_none_for_none_input(self):
        assert _fcf_series_from_cashflow(None) is None

    def test_returns_none_for_empty_dataframe(self):
        assert _fcf_series_from_cashflow(pd.DataFrame()) is None

    def test_returns_none_when_no_usable_rows(self):
        cf = _cf({'Net Income': [50.0, 60.0]})
        assert _fcf_series_from_cashflow(cf) is None

    def test_returns_none_when_only_ocf_available(self):
        cf = _cf({'Operating Cash Flow': [200.0, 220.0]})
        assert _fcf_series_from_cashflow(cf) is None

    def test_returns_none_when_only_capex_available(self):
        cf = _cf({'Capital Expenditure': [-60.0, -70.0]})
        assert _fcf_series_from_cashflow(cf) is None

    def test_returns_none_when_periods_dont_overlap(self):
        # OCF and Capex on disjoint periods → no common index, no FCF.
        ocf = pd.Series([200.0, 220.0], index=['2023-12-31', '2024-12-31'], name='Operating Cash Flow')
        capex = pd.Series([-60.0, -70.0], index=['2021-12-31', '2022-12-31'], name='Capital Expenditure')
        cf = pd.DataFrame([ocf, capex])
        assert _fcf_series_from_cashflow(cf) is None

    def test_fcf_row_present_but_all_null_falls_back_to_derivation(self):
        # If yfinance surfaces 'Free Cash Flow' as a row but it's all-null,
        # we should still derive from OCF + Capex rather than give up.
        cf = _cf({
            'Free Cash Flow': [float('nan'), float('nan'), float('nan')],
            'Operating Cash Flow': [200.0, 220.0, 250.0],
            'Capital Expenditure': [-60.0, -70.0, -80.0],
        })
        fcf = _fcf_series_from_cashflow(cf)
        assert fcf is not None
        assert list(fcf.values) == [140.0, 150.0, 170.0]
