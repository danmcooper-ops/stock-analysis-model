# tests/test_calibrate.py
"""Tests for walk-forward calibration: window splitting, objectives, grid search."""

import sys
import os
import pytest
import json
from datetime import date

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.calibrate import (
    generate_windows,
    compute_objective, hit_rate_objective, alpha_objective,
    information_ratio_objective, composite_objective,
    _generate_grid, _apply_derived_params, _sample_grid,
    grid_search, regularized_objective, compute_stability,
    _discover_snapshot_dates,
)
from scripts.param_set import default_params


# ======================================================================
# Window generation
# ======================================================================

class TestGenerateWindows:

    def test_basic_split(self):
        dates = [date(2026, 1, d) for d in range(1, 8)]  # 7 dates
        wins = generate_windows(dates, train_size=3, test_size=1, step=1)
        assert len(wins) == 4  # 7 - (3+1) + 1 = 4
        assert len(wins[0]['train_dates']) == 3
        assert len(wins[0]['test_dates']) == 1

    def test_insufficient_snapshots(self):
        dates = [date(2026, 1, 1), date(2026, 1, 2)]
        wins = generate_windows(dates, train_size=3, test_size=1)
        assert wins == []

    def test_exact_minimum(self):
        dates = [date(2026, 1, d) for d in range(1, 5)]  # 4 dates
        wins = generate_windows(dates, train_size=3, test_size=1)
        assert len(wins) == 1
        assert wins[0]['train_dates'] == dates[:3]
        assert wins[0]['test_dates'] == [dates[3]]

    def test_step_parameter(self):
        dates = [date(2026, 1, d) for d in range(1, 11)]  # 10 dates
        wins = generate_windows(dates, train_size=3, test_size=1, step=2)
        # Positions: 0, 2, 4, 6 → 4 windows (6+4=10 ✓)
        assert len(wins) == 4

    def test_windows_non_overlapping_test(self):
        dates = [date(2026, 1, d) for d in range(1, 8)]
        wins = generate_windows(dates, train_size=3, test_size=1, step=1)
        # Each window has different test dates (due to step=1, they overlap in train)
        test_dates = [w['test_dates'][0] for w in wins]
        assert len(test_dates) == len(set(test_dates))

    def test_larger_test_window(self):
        dates = [date(2026, 1, d) for d in range(1, 9)]  # 8 dates
        wins = generate_windows(dates, train_size=3, test_size=2, step=1)
        assert len(wins) == 4  # 8 - (3+2) + 1 = 4
        assert len(wins[0]['test_dates']) == 2


# ======================================================================
# Objective functions
# ======================================================================

def _make_metrics(buy_returns, non_buy_returns=None):
    """Helper to create backtest-compatible metric dicts."""
    details = []
    for er in buy_returns:
        details.append({'rating': 'BUY', 'excess_return': er})
    for er in (non_buy_returns or []):
        details.append({'rating': 'HOLD', 'excess_return': er})
    return [{'details': details}]


class TestHitRateObjective:

    def test_all_winners(self):
        m = _make_metrics([0.05, 0.10, 0.03])
        assert hit_rate_objective(m) == pytest.approx(1.0)

    def test_all_losers(self):
        m = _make_metrics([-0.05, -0.10, -0.03])
        assert hit_rate_objective(m) == pytest.approx(0.0)

    def test_mixed(self):
        m = _make_metrics([0.05, -0.03, 0.10, -0.01])
        assert hit_rate_objective(m) == pytest.approx(0.5)

    def test_empty(self):
        assert hit_rate_objective([{'details': []}]) == 0.0

    def test_ignores_non_buy(self):
        m = _make_metrics([0.05], non_buy_returns=[-0.10])
        assert hit_rate_objective(m) == pytest.approx(1.0)


class TestAlphaObjective:

    def test_positive_alpha(self):
        m = _make_metrics([0.10, 0.20, 0.30])
        assert alpha_objective(m) == pytest.approx(0.20)

    def test_negative_alpha(self):
        m = _make_metrics([-0.10, -0.20])
        assert alpha_objective(m) == pytest.approx(-0.15)

    def test_empty(self):
        assert alpha_objective([{'details': []}]) == 0.0


class TestInformationRatioObjective:

    def test_positive_ir(self):
        m = _make_metrics([0.10, 0.12, 0.08])
        ir = information_ratio_objective(m)
        assert ir > 0  # Positive alpha, low volatility

    def test_insufficient_data(self):
        m = _make_metrics([0.10, 0.05])  # Only 2 points, need >= 3
        assert information_ratio_objective(m) == 0.0


class TestCompositeObjective:

    def test_perfect_scores(self):
        # All beat SPY, high alpha, all FV within 20%
        m = [{
            'details': [
                {'rating': 'BUY', 'excess_return': 0.08,
                 'dcf_fv': 100, 'end_price': 105},
                {'rating': 'BUY', 'excess_return': 0.05,
                 'dcf_fv': 100, 'end_price': 95},
            ]
        }]
        obj = composite_objective(m)
        assert obj > 0.5  # Should be good across all dimensions

    def test_compute_objective_dispatches(self):
        m = _make_metrics([0.05, -0.03])
        assert compute_objective(m, 'hit_rate') == pytest.approx(0.5)
        assert compute_objective(m, 'alpha') == pytest.approx(0.01)


# ======================================================================
# Grid search
# ======================================================================

class TestGridGeneration:

    def test_small_grid(self):
        space = {'x': (0.0, 1.0, 0.5)}  # 3 values: 0.0, 0.5, 1.0
        grid = _generate_grid(space)
        assert len(grid) == 3
        assert grid[0] == {'x': 0.0}
        assert grid[2] == {'x': 1.0}

    def test_multi_dim_grid(self):
        space = {
            'a': (0.0, 1.0, 1.0),  # 2 values
            'b': (0.0, 1.0, 0.5),  # 3 values
        }
        grid = _generate_grid(space)
        assert len(grid) == 6  # 2 × 3

    def test_apply_derived_params_computes_growth_weight(self):
        candidate = {
            'score_weight_valuation': 0.30,
            'score_weight_quality': 0.25,
            'score_weight_moat': 0.25,
        }
        params = _apply_derived_params(candidate)
        assert params is not None
        # growth = 1.0 - (val + qual + moat + ownership_default)
        # = 1.0 - (0.30 + 0.25 + 0.25 + 0.10) = 0.10
        assert params['score_weight_growth'] == pytest.approx(0.10)

    def test_apply_derived_params_rejects_negative_growth(self):
        candidate = {
            'score_weight_valuation': 0.45,
            'score_weight_quality': 0.40,
            'score_weight_moat': 0.20,
        }
        # Sum = 1.05, growth = -0.05 → rejected
        params = _apply_derived_params(candidate)
        assert params is None

    def test_apply_derived_computes_blend_mult_weight(self):
        candidate = {'blend_dcf_weight': 0.65}
        params = _apply_derived_params(candidate)
        assert params is not None
        assert params['blend_mult_weight'] == pytest.approx(0.35)


class TestGridSearch:

    def test_finds_optimal_in_small_space(self):
        # Simple quadratic: maximize f(x) = -(x - 0.5)^2
        space = {'erp': (0.03, 0.08, 0.01)}

        def evaluate(params):
            x = params['erp']
            return -(x - 0.055) ** 2

        results = grid_search(evaluate, space, max_evaluations=100)
        assert len(results) > 0
        best = results[0]['params']['erp']
        assert abs(best - 0.055) <= 0.01  # Within one step

    def test_respects_max_evaluations(self):
        space = {
            'erp': (0.03, 0.08, 0.005),        # 11 values
            'blend_trigger': (1.0, 2.0, 0.1),    # 11 values
        }
        # Full grid = 121, limit to 50
        results = grid_search(lambda p: 0.5, space, max_evaluations=50)
        assert len(results) <= 50


class TestSampling:

    def test_sample_respects_size(self):
        grid = [{'x': i} for i in range(1000)]
        sampled = _sample_grid(grid, 50)
        assert len(sampled) == 50

    def test_sample_returns_full_grid_when_small(self):
        grid = [{'x': i} for i in range(10)]
        sampled = _sample_grid(grid, 100)
        assert len(sampled) == 10

    def test_sample_deterministic(self):
        grid = [{'x': i} for i in range(100)]
        s1 = _sample_grid(grid, 20, seed=42)
        s2 = _sample_grid(grid, 20, seed=42)
        assert s1 == s2


# ======================================================================
# Overfitting prevention
# ======================================================================

class TestRegularization:

    def test_default_params_no_penalty(self):
        p = default_params()
        penalised = regularized_objective(0.70, p, lambda_reg=0.05)
        assert penalised == pytest.approx(0.70)  # No deviation = no penalty

    def test_large_deviation_penalised(self):
        p = default_params()
        p['erp'] = 0.10  # Big jump from default 0.055
        penalised = regularized_objective(0.70, p, lambda_reg=0.05)
        assert penalised < 0.70

    def test_lambda_zero_no_penalty(self):
        p = default_params()
        p['erp'] = 0.10
        penalised = regularized_objective(0.70, p, lambda_reg=0.0)
        assert penalised == pytest.approx(0.70)


class TestStability:

    def test_stable_params(self):
        windows = [
            {'best_params': {'erp': 0.055, 'blend_trigger': 1.5}},
            {'best_params': {'erp': 0.055, 'blend_trigger': 1.5}},
        ]
        s = compute_stability(windows)
        assert s['erp'] == pytest.approx(0.0)

    def test_unstable_params(self):
        windows = [
            {'best_params': {'erp': 0.04}},
            {'best_params': {'erp': 0.07}},
        ]
        s = compute_stability(windows)
        assert s['erp'] > 0.01

    def test_empty_windows(self):
        assert compute_stability([]) == {}


# ======================================================================
# Snapshot discovery
# ======================================================================

class TestSnapshotDiscovery:

    def test_discovers_dates(self, tmp_path):
        for d in ['2026-03-08', '2026-03-09', '2026-03-10']:
            (tmp_path / f'results_{d}.json').write_text('{}')
        dates = _discover_snapshot_dates(str(tmp_path))
        assert len(dates) == 3
        assert dates[0] == date(2026, 3, 8)
        assert dates[-1] == date(2026, 3, 10)

    def test_ignores_non_result_files(self, tmp_path):
        (tmp_path / 'results_2026-03-08.json').write_text('{}')
        (tmp_path / 'backtest_2026-03-08.json').write_text('{}')
        (tmp_path / 'stock_analysis.html').write_text('')
        dates = _discover_snapshot_dates(str(tmp_path))
        assert len(dates) == 1

    def test_empty_directory(self, tmp_path):
        dates = _discover_snapshot_dates(str(tmp_path))
        assert dates == []

    def test_nonexistent_directory(self):
        dates = _discover_snapshot_dates('/nonexistent/path')
        assert dates == []
