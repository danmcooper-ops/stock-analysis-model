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

    def test_legacy_quarterly_partial_year_dropped(self):
        """Quarterly-keyed series with a partial fiscal year (in-progress
        current year, or a backfill cutoff that only captured one quarter)
        must drop the partial — treating one quarter as the full year was
        the source of the rev_cagr_5y bug (e.g. CRCT showing -34%)."""
        history = {
            '2019-12-31': 487.0,                       # Q4-only, partial
            '2020-03-31': 144.0, '2020-06-30': 235.0,
            '2020-09-30': 209.0, '2020-12-31': 371.0,  # full 2020
            '2021-03-31': 162.0,                       # Q1-only, partial
        }
        # Only the year with 4 quarterly entries survives.
        assert _flow_to_annual(history) == {2020: 144.0 + 235.0 + 209.0 + 371.0}

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
