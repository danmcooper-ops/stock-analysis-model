# tests/test_snapshot_cache.py
"""Tests for snapshot cache and point-in-time financial slicing."""

import sys
import os
import pytest
import pandas as pd
import numpy as np
from datetime import date

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.snapshot_cache import SnapshotCache
from data.time_slice import slice_financials_as_of, _is_available


# ======================================================================
# SnapshotCache tests
# ======================================================================

class TestSnapshotCacheSaveLoad:
    """Round-trip serialization tests."""

    def test_save_and_load_roundtrip(self, sample_financials, tmp_path):
        cache = SnapshotCache(cache_dir=str(tmp_path))
        cache.save('AAPL', sample_financials, as_of=date(2026, 3, 14))
        loaded = cache.load('AAPL', date(2026, 3, 14))

        assert loaded is not None
        assert set(loaded.keys()) == set(sample_financials.keys())

        # Info dict preserved
        assert loaded['info']['marketCap'] == sample_financials['info']['marketCap']
        assert loaded['info']['sector'] == 'Technology'

        # DataFrame shapes preserved
        assert loaded['balance_sheet'].shape == sample_financials['balance_sheet'].shape
        assert loaded['income_statement'].shape == sample_financials['income_statement'].shape
        assert loaded['cash_flow'].shape == sample_financials['cash_flow'].shape

    def test_dataframe_values_preserved(self, sample_financials, tmp_path):
        cache = SnapshotCache(cache_dir=str(tmp_path))
        cache.save('MSFT', sample_financials, as_of=date(2026, 1, 15))
        loaded = cache.load('MSFT', date(2026, 1, 15))

        orig_bs = sample_financials['balance_sheet']
        load_bs = loaded['balance_sheet']
        # Check a specific value
        assert load_bs.iloc[0, 0] == pytest.approx(orig_bs.iloc[0, 0])

    def test_nan_and_none_handling(self, sample_financials, tmp_path):
        """NaN and inf should serialise as None and round-trip correctly."""
        cache = SnapshotCache(cache_dir=str(tmp_path))
        financials = dict(sample_financials)
        # Add a field with NaN
        info = dict(financials['info'])
        info['testField'] = float('nan')
        financials['info'] = info
        cache.save('TST', financials, as_of=date(2026, 2, 1))
        loaded = cache.load('TST', date(2026, 2, 1))
        assert loaded['info']['testField'] is None

    def test_none_dataframes_passthrough(self, sample_financials, tmp_path):
        """growth_estimates and earnings_history may be None."""
        cache = SnapshotCache(cache_dir=str(tmp_path))
        financials = dict(sample_financials)
        financials['growth_estimates'] = None
        financials['earnings_history'] = None
        cache.save('NONE', financials, as_of=date(2026, 1, 1))
        loaded = cache.load('NONE', date(2026, 1, 1))
        assert loaded['growth_estimates'] is None
        assert loaded['earnings_history'] is None


class TestSnapshotCacheDateSelection:
    """Tests for date-based loading logic."""

    def test_load_returns_closest_prior_date(self, sample_financials, tmp_path):
        cache = SnapshotCache(cache_dir=str(tmp_path))
        cache.save('GOOG', sample_financials, as_of=date(2026, 1, 10))
        cache.save('GOOG', sample_financials, as_of=date(2026, 2, 15))
        cache.save('GOOG', sample_financials, as_of=date(2026, 3, 20))

        # Query for Feb 20 — should get Feb 15 snapshot (closest <= Feb 20)
        loaded = cache.load('GOOG', date(2026, 2, 20))
        assert loaded is not None

    def test_load_returns_none_for_future_date(self, sample_financials, tmp_path):
        cache = SnapshotCache(cache_dir=str(tmp_path))
        cache.save('META', sample_financials, as_of=date(2026, 6, 1))

        # Query for May — no snapshot on or before May
        loaded = cache.load('META', date(2026, 5, 1))
        assert loaded is None

    def test_load_exact_date_match(self, sample_financials, tmp_path):
        cache = SnapshotCache(cache_dir=str(tmp_path))
        cache.save('AMZN', sample_financials, as_of=date(2026, 4, 4))

        loaded = cache.load('AMZN', date(2026, 4, 4))
        assert loaded is not None

    def test_load_returns_none_for_empty_cache(self, tmp_path):
        cache = SnapshotCache(cache_dir=str(tmp_path))
        loaded = cache.load('NOPE', date(2026, 1, 1))
        assert loaded is None


class TestSnapshotCacheMetadata:
    """Tests for available_dates, has_snapshot, all_tickers."""

    def test_available_dates_sorted(self, sample_financials, tmp_path):
        cache = SnapshotCache(cache_dir=str(tmp_path))
        cache.save('TSLA', sample_financials, as_of=date(2026, 3, 1))
        cache.save('TSLA', sample_financials, as_of=date(2026, 1, 1))
        cache.save('TSLA', sample_financials, as_of=date(2026, 2, 1))

        dates = cache.available_dates('TSLA')
        assert dates == [date(2026, 1, 1), date(2026, 2, 1), date(2026, 3, 1)]

    def test_has_snapshot(self, sample_financials, tmp_path):
        cache = SnapshotCache(cache_dir=str(tmp_path))
        cache.save('V', sample_financials, as_of=date(2026, 5, 5))
        assert cache.has_snapshot('V', date(2026, 5, 5)) is True
        assert cache.has_snapshot('V', date(2026, 5, 6)) is False

    def test_all_tickers(self, sample_financials, tmp_path):
        cache = SnapshotCache(cache_dir=str(tmp_path))
        cache.save('AAPL', sample_financials, as_of=date(2026, 1, 1))
        cache.save('MSFT', sample_financials, as_of=date(2026, 1, 1))
        cache.save('GOOG', sample_financials, as_of=date(2026, 1, 1))

        tickers = cache.all_tickers()
        assert tickers == ['AAPL', 'GOOG', 'MSFT']

    def test_ticker_case_normalised(self, sample_financials, tmp_path):
        cache = SnapshotCache(cache_dir=str(tmp_path))
        cache.save('aapl', sample_financials, as_of=date(2026, 1, 1))
        assert cache.has_snapshot('AAPL', date(2026, 1, 1)) is True


# ======================================================================
# Time-slice tests
# ======================================================================

class TestTimeSlice:
    """Tests for point-in-time financial slicing."""

    def test_slice_removes_future_columns(self, sample_financials):
        # Sample data has columns: 2024-12-31 and 2023-12-31
        # With 90-day lag, 2024-12-31 is available after ~2025-03-31
        # as_of = 2025-02-01 → only 2023-12-31 should survive (avail ~2024-03-31)
        sliced = slice_financials_as_of(sample_financials, date(2025, 2, 1))
        assert sliced['balance_sheet'].shape[1] == 1  # Only 2023

    def test_slice_keeps_available_columns(self, sample_financials):
        # as_of = 2025-06-01 → both columns available
        # 2023-12-31 + 90d = 2024-03-31 ✓, 2024-12-31 + 90d = 2025-03-31 ✓
        sliced = slice_financials_as_of(sample_financials, date(2025, 6, 1))
        assert sliced['balance_sheet'].shape[1] == 2

    def test_slice_respects_reporting_lag(self, sample_financials):
        # 2024-12-31 + 90 days = 2025-03-31
        # as_of = 2025-03-30 → NOT available yet
        sliced = slice_financials_as_of(sample_financials, date(2025, 3, 30))
        assert sliced['balance_sheet'].shape[1] == 1

        # as_of = 2025-03-31 → available
        sliced2 = slice_financials_as_of(sample_financials, date(2025, 3, 31))
        assert sliced2['balance_sheet'].shape[1] == 2

    def test_slice_empty_when_all_future(self, sample_financials):
        # as_of = 2024-01-01 → 2023-12-31 + 90d = 2024-03-31, not available yet
        sliced = slice_financials_as_of(sample_financials, date(2024, 1, 1))
        assert sliced['balance_sheet'].shape[1] == 0

    def test_slice_info_passthrough(self, sample_financials):
        sliced = slice_financials_as_of(sample_financials, date(2025, 6, 1))
        assert sliced['info'] is sample_financials['info']  # Same object

    def test_slice_none_input(self):
        assert slice_financials_as_of(None, date(2025, 1, 1)) is None

    def test_custom_reporting_lag(self, sample_financials):
        # With 0-day lag, 2024-12-31 is available on 2024-12-31
        sliced = slice_financials_as_of(sample_financials, date(2025, 1, 1),
                                        reporting_lag_days=0)
        assert sliced['balance_sheet'].shape[1] == 2


class TestIsAvailable:
    """Tests for the _is_available helper."""

    def test_available_after_lag(self):
        col = pd.Timestamp('2024-12-31')
        assert _is_available(col, date(2025, 4, 1), 90) is True

    def test_not_available_before_lag(self):
        col = pd.Timestamp('2024-12-31')
        assert _is_available(col, date(2025, 3, 30), 90) is False

    def test_available_on_exact_lag_date(self):
        col = pd.Timestamp('2024-12-31')
        assert _is_available(col, date(2025, 3, 31), 90) is True

    def test_zero_lag(self):
        col = pd.Timestamp('2024-06-30')
        assert _is_available(col, date(2024, 6, 30), 0) is True
