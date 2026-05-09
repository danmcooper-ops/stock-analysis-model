# tests/test_edgar_history.py
import sys
import os

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
