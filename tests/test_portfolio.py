# tests/test_portfolio.py
import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.portfolio import position_sizes, concentration_analysis


# ---------------------------------------------------------------------------
# position_sizes
# ---------------------------------------------------------------------------

class TestPositionSizes:
    def test_basic_weights_sum_to_one(self):
        """Weights should sum to approximately 1.0."""
        data = [
            {'ticker': 'A', 'mos': 0.20, 'rating': 'BUY', 'mc_cv': 0.15},
            {'ticker': 'B', 'mos': 0.15, 'rating': 'BUY', 'mc_cv': 0.20},
            {'ticker': 'C', 'mos': 0.10, 'rating': 'LEAN BUY', 'mc_cv': 0.25},
        ]
        weights = position_sizes(data)
        assert len(weights) > 0
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_max_weight_enforced(self):
        """No single position should exceed max_weight when enough candidates."""
        data = [
            {'ticker': f'S{i}', 'mos': 0.10 + i * 0.02, 'rating': 'BUY', 'mc_cv': 0.15}
            for i in range(20)
        ]
        weights = position_sizes(data, max_weight=0.08)
        assert len(weights) > 0
        for w in weights.values():
            assert w <= 0.08 + 1e-6  # tolerance for floating point

    def test_empty_input(self):
        """Empty list → empty dict."""
        assert position_sizes([]) == {}

    def test_higher_mos_gets_higher_weight(self):
        """Higher MoS stock should get higher weight."""
        data = [
            {'ticker': 'A', 'mos': 0.30, 'rating': 'BUY', 'mc_cv': 0.15},
            {'ticker': 'B', 'mos': 0.05, 'rating': 'BUY', 'mc_cv': 0.15},
        ]
        weights = position_sizes(data, max_weight=1.0)
        assert weights['A'] > weights['B']

    def test_no_buys_returns_empty(self):
        """No BUY/LEAN BUY stocks → empty dict."""
        data = [
            {'ticker': 'A', 'mos': 0.20, 'rating': 'HOLD', 'mc_cv': 0.15},
            {'ticker': 'B', 'mos': 0.10, 'rating': 'PASS', 'mc_cv': 0.20},
        ]
        assert position_sizes(data) == {}

    def test_negative_mos_excluded(self):
        """Stocks with negative MoS should be excluded."""
        data = [
            {'ticker': 'A', 'mos': 0.20, 'rating': 'BUY', 'mc_cv': 0.15},
            {'ticker': 'B', 'mos': -0.10, 'rating': 'BUY', 'mc_cv': 0.15},
        ]
        weights = position_sizes(data)
        assert 'B' not in weights

    def test_lower_cv_gets_higher_weight(self):
        """Lower MC CV (higher confidence) → higher weight."""
        data = [
            {'ticker': 'A', 'mos': 0.20, 'rating': 'BUY', 'mc_cv': 0.05},
            {'ticker': 'B', 'mos': 0.20, 'rating': 'BUY', 'mc_cv': 0.50},
        ]
        weights = position_sizes(data, max_weight=1.0)
        assert weights['A'] > weights['B']


# ---------------------------------------------------------------------------
# concentration_analysis
# ---------------------------------------------------------------------------

class TestConcentrationAnalysis:
    def test_basic(self):
        """Should return dict with expected keys."""
        data = [
            {'ticker': 'A', 'sector': 'Tech'},
            {'ticker': 'B', 'sector': 'Health'},
            {'ticker': 'C', 'sector': 'Tech'},
        ]
        result = concentration_analysis(data)
        assert 'sector_weights' in result
        assert 'top_sector' in result
        assert 'concentration_flag' in result
        assert 'hhi' in result
        assert 'n_sectors' in result

    def test_concentration_flag_single_sector(self):
        """All stocks in one sector → concentration flag."""
        data = [
            {'ticker': 'A', 'sector': 'Tech'},
            {'ticker': 'B', 'sector': 'Tech'},
            {'ticker': 'C', 'sector': 'Tech'},
        ]
        result = concentration_analysis(data)
        assert result['concentration_flag'] is True
        assert result['top_sector'] == 'Tech'
        assert result['top_sector_weight'] == pytest.approx(1.0)

    def test_no_flag_diversified(self):
        """Evenly spread across 5+ sectors → no flag."""
        data = [
            {'ticker': str(i), 'sector': f'Sector{i}'}
            for i in range(10)
        ]
        result = concentration_analysis(data)
        assert result['concentration_flag'] is False

    def test_hhi_calculation(self):
        """HHI for two equal sectors = 0.5."""
        data = [
            {'ticker': 'A', 'sector': 'X'},
            {'ticker': 'B', 'sector': 'Y'},
        ]
        result = concentration_analysis(data)
        assert result['hhi'] == pytest.approx(0.5)

    def test_empty_input(self):
        """Empty list → no flag, zero HHI."""
        result = concentration_analysis([])
        assert result['concentration_flag'] is False
        assert result['hhi'] == 0.0

    def test_respects_position_weights(self):
        """When position_weight is provided, use it for sector allocation."""
        data = [
            {'ticker': 'A', 'sector': 'Tech', 'position_weight': 0.80},
            {'ticker': 'B', 'sector': 'Health', 'position_weight': 0.20},
        ]
        result = concentration_analysis(data)
        assert result['top_sector'] == 'Tech'
        assert result['top_sector_weight'] == pytest.approx(0.80)
        assert result['concentration_flag'] is True
