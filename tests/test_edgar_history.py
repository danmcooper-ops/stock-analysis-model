# tests/test_edgar_history.py
import sys
import os

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.analyze_stock import _flow_to_annual, _stock_to_annual


class TestFlowToAnnual:
    def test_int_keyed_passthrough(self):
        """New EDGAR client format: int year keys, one value per FY."""
        history = {2020: 100.0, 2021: 110.0, 2022: 121.0}
        assert _flow_to_annual(history) == {2020: 100.0, 2021: 110.0, 2022: 121.0}

    def test_year_string_keyed_passthrough(self):
        """JSON round-trip turns int year keys into 4-char digit strings."""
        history = {'2020': 100.0, '2021': 110.0, '2022': 121.0}
        assert _flow_to_annual(history) == {2020: 100.0, 2021: 110.0, 2022: 121.0}

    def test_legacy_full_year_only(self):
        """Legacy date-keyed format: one annual value per year passes through."""
        history = {'2020-12-31': 100.0, '2021-12-31': 110.0}
        assert _flow_to_annual(history) == {2020: 100.0, 2021: 110.0}

    def test_legacy_four_quarters_summed(self):
        """Legacy date-keyed format: four quarterly entries sum to annual."""
        history = {
            '2020-03-31': 25.0,
            '2020-06-30': 25.0,
            '2020-09-30': 25.0,
            '2020-12-31': 25.0,
        }
        assert _flow_to_annual(history) == {2020: 100.0}

    def test_legacy_mixed_period_year_dropped(self):
        """Legacy date-keyed format: 2 or 3 entries per year are dropped (the
        ambiguous mixed-period case from the old quarterly extractor)."""
        history = {
            '2020-03-31': 25.0,
            '2020-12-31': 100.0,  # FY + Q1 → ambiguous, drop
            '2021-12-31': 110.0,
        }
        assert _flow_to_annual(history) == {2021: 110.0}

    def test_empty(self):
        assert _flow_to_annual({}) == {}
        assert _flow_to_annual(None) == {}

    def test_none_values_skipped(self):
        history = {2020: 100.0, 2021: None, 2022: 121.0}
        assert _flow_to_annual(history) == {2020: 100.0, 2022: 121.0}


class TestStockToAnnual:
    def test_latest_per_year(self):
        """Point-in-time series: keep latest observation per calendar year."""
        history = {
            '2020-03-31': 100.0,
            '2020-12-31': 95.0,   # later in 2020 → keep
            '2021-06-30': 90.0,
        }
        assert _stock_to_annual(history) == {2020: 95.0, 2021: 90.0}

    def test_empty(self):
        assert _stock_to_annual({}) == {}
        assert _stock_to_annual(None) == {}


# ---------------------------------------------------------------------------
# derive_edgar_metrics — through-cycle operating margin (2026-07 rebalance)
# ---------------------------------------------------------------------------

class TestOpMarginHistory:
    def _hist(self, years):
        """edgar_history with flat revenue 1000 and op income by year."""
        return {
            'revenue_history': {y: 1000.0 for y in years},
            'operating_income_history': {y: 150.0 + (y % 2) * 50.0
                                         for y in years},
        }

    def test_ten_year_average(self):
        from scripts.analyze_stock import derive_edgar_metrics
        years = list(range(2015, 2025))  # 10 years
        m = derive_edgar_metrics(self._hist(years))
        assert m['op_margin_hist_years'] == 10
        # margins alternate 0.15 / 0.20 → mean 0.175
        assert m['op_margin_avg_10y'] == pytest.approx(0.175)

    def test_window_capped_at_ten_years(self):
        from scripts.analyze_stock import derive_edgar_metrics
        years = list(range(2009, 2025))  # 16 years
        m = derive_edgar_metrics(self._hist(years))
        assert m['op_margin_hist_years'] == 10

    def test_missing_series_yields_zero_years(self):
        from scripts.analyze_stock import derive_edgar_metrics
        m = derive_edgar_metrics({'revenue_history': {2024: 1000.0}})
        assert m['op_margin_avg_10y'] is None
        assert m['op_margin_hist_years'] == 0

    def test_zero_revenue_years_excluded(self):
        from scripts.analyze_stock import derive_edgar_metrics
        hist = {
            'revenue_history': {2022: 0.0, 2023: 1000.0, 2024: 1000.0},
            'operating_income_history': {2022: 100.0, 2023: 200.0, 2024: 200.0},
        }
        m = derive_edgar_metrics(hist)
        assert m['op_margin_hist_years'] == 2
        assert m['op_margin_avg_10y'] == pytest.approx(0.20)


class TestIntCovEdgar:
    """EBIT / interest expense derived from EDGAR — the fallback for rows
    where yfinance surfaces no income statement."""

    def test_uses_latest_common_year(self):
        from scripts.analyze_stock import derive_edgar_metrics
        hist = {
            'operating_income_history': {2023: 8000.0, 2024: 8699.0, 2025: 8127.0},
            'interest_expense_history': {2023: 900.0, 2024: 1058.0, 2025: 1344.0},
        }
        m = derive_edgar_metrics(hist)
        # latest common year 2025: 8127 / 1344
        assert m['int_cov_edgar'] == pytest.approx(6.046875)

    def test_intersects_years_across_the_two_series(self):
        """The newest op-income year with no matching interest year is skipped
        rather than pairing mismatched periods."""
        from scripts.analyze_stock import derive_edgar_metrics
        hist = {
            'operating_income_history': {2023: 1000.0, 2024: 2000.0},
            'interest_expense_history': {2023: 100.0},
        }
        m = derive_edgar_metrics(hist)
        assert m['int_cov_edgar'] == pytest.approx(10.0)

    def test_negative_ebit_propagates(self):
        from scripts.analyze_stock import derive_edgar_metrics
        hist = {
            'operating_income_history': {2024: -500.0},
            'interest_expense_history': {2024: 200.0},
        }
        m = derive_edgar_metrics(hist)
        assert m['int_cov_edgar'] == pytest.approx(-2.5)

    def test_zero_or_missing_interest_yields_none(self):
        from scripts.analyze_stock import derive_edgar_metrics
        assert derive_edgar_metrics({
            'operating_income_history': {2024: 500.0},
            'interest_expense_history': {2024: 0.0},
        })['int_cov_edgar'] is None
        assert derive_edgar_metrics({
            'operating_income_history': {2024: 500.0},
        })['int_cov_edgar'] is None

    def test_absent_history_yields_none(self):
        from scripts.analyze_stock import derive_edgar_metrics
        assert derive_edgar_metrics({})['int_cov_edgar'] is None
        assert derive_edgar_metrics(None)['int_cov_edgar'] is None
