# tests/test_backtest.py
"""Tests for the merged backtest module: window splitting, objectives, grid search."""

import pytest
from datetime import date


from scripts.backtest import (
    generate_windows,
    compute_objective, hit_rate_objective, alpha_objective,
    information_ratio_objective, composite_objective, rank_ic_objective,
    is_matured, _spaced_dates, _calibrated_weights,
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


class TestRankICObjective:

    def _metrics(self, pairs):
        return [{'details': [
            {'_composite_score': s, 'excess_return': er} for s, er in pairs
        ]}]

    def test_perfect_positive_correlation(self):
        m = self._metrics([(i, i * 0.01) for i in range(20)])
        assert rank_ic_objective(m) == pytest.approx(1.0)

    def test_perfect_negative_correlation(self):
        m = self._metrics([(i, -i * 0.01) for i in range(20)])
        assert rank_ic_objective(m) == pytest.approx(-1.0)

    def test_insufficient_data_returns_zero(self):
        m = self._metrics([(1, 0.01), (2, 0.02)])  # n < 10
        assert rank_ic_objective(m) == 0.0

    def test_skips_rows_missing_returns(self):
        # Rows without excess_return must not poison the correlation.
        pairs = [(i, i * 0.01) for i in range(15)]
        m = self._metrics(pairs)
        m[0]['details'].append({'_composite_score': 99, 'excess_return': None})
        assert rank_ic_objective(m) == pytest.approx(1.0)

    def test_dispatch_default_is_rank_ic(self):
        m = self._metrics([(i, i * 0.01) for i in range(20)])
        # unknown name falls back to rank_ic
        assert compute_objective(m, 'rank_ic') == pytest.approx(1.0)
        assert compute_objective(m, 'nonexistent') == pytest.approx(1.0)


class TestSpacedDates:

    def test_thins_daily_dates_to_spacing(self):
        dates = [date(2026, 1, 1) + __import__('datetime').timedelta(days=i)
                 for i in range(100)]
        picked = _spaced_dates(dates, 30)
        assert picked[0] == date(2026, 1, 1)
        gaps = [(b - a).days for a, b in zip(picked, picked[1:], strict=False)]
        assert all(g >= 30 for g in gaps)
        assert len(picked) == 4  # days 0, 30, 60, 90

    def test_already_spaced_dates_unchanged(self):
        dates = [date(2026, 1, 1), date(2026, 4, 1), date(2026, 7, 1)]
        assert _spaced_dates(dates, 30) == dates

    def test_empty(self):
        assert _spaced_dates([], 30) == []


class TestCalibratedWeights:

    def test_reduces_full_paramset_to_weight_block(self):
        full = default_params()
        w = _calibrated_weights(full)
        assert set(w) == {'score_weight_valuation', 'score_weight_quality',
                          'score_weight_moat', 'score_weight_growth',
                          'score_weight_ownership'}
        # no fair-value params leak into "calibrated" output
        assert 'erp' not in w and 'analyst_haircut' not in w

    def test_search_space_has_no_dead_dimensions(self):
        # Frozen-FV re-scoring only responds to score weights; anything else
        # in SEARCH_SPACE would be swept as a silent no-op.
        from scripts.backtest import SEARCH_SPACE
        assert all(k.startswith('score_weight_') for k in SEARCH_SPACE)


class TestIsMatured:

    def test_matured_when_window_elapsed(self):
        assert is_matured('2026-04-20', 30, today=date(2026, 6, 8)) is True

    def test_immature_when_window_in_future(self):
        assert is_matured('2026-06-07', 30, today=date(2026, 6, 8)) is False

    def test_exact_boundary_is_matured(self):
        # run_date + horizon == today → matured (window fully elapsed)
        assert is_matured('2026-05-09', 30, today=date(2026, 6, 8)) is True

    def test_one_day_short_is_immature(self):
        assert is_matured('2026-05-10', 30, today=date(2026, 6, 8)) is False


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
        # Ownership is held at its config default; pick V/Q/M so that
        # growth = 1.0 - (V + Q + M + Own_default) lands at exactly 0.05
        # without re-encoding a stale ownership-weight literal.
        own = default_params()['score_weight_ownership']
        wv, wq, wm = 0.30, 0.25, round(1.0 - 0.30 - 0.25 - own - 0.05, 4)
        candidate = {
            'score_weight_valuation': wv,
            'score_weight_quality': wq,
            'score_weight_moat': wm,
        }
        params = _apply_derived_params(candidate)
        assert params is not None
        # growth = 1.0 - (V + Q + M + Own_default), by construction = 0.05
        assert params['score_weight_growth'] == pytest.approx(0.05)

    def test_apply_derived_params_rejects_negative_growth(self):
        # Pick V+Q+M+Own_default > 1.0 so the derived growth would be < 0
        # and `_apply_derived_params` must reject the candidate, regardless
        # of what the live ownership default happens to be.
        own = default_params()['score_weight_ownership']
        wv, wq, wm = 0.45, 0.40, round(1.0 - 0.45 - 0.40 - own + 0.05, 4)
        candidate = {
            'score_weight_valuation': wv,
            'score_weight_quality': wq,
            'score_weight_moat': wm,
        }
        # By construction sum > 1.0, growth < 0 → rejected
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


class TestForwardReturnJoin:
    """The calibration objectives must consume REAL joined forward returns
    (the old code read _excess_return keys that nothing ever wrote)."""

    def test_hit_rate_excludes_none_returns(self):
        m = [{'details': [
            {'rating': 'BUY', 'excess_return': None},
            {'rating': 'BUY', 'excess_return': 0.05},
        ]}]
        assert hit_rate_objective(m) == pytest.approx(1.0)

    def test_hit_rate_all_none_is_zero_not_crash(self):
        m = [{'details': [{'rating': 'BUY', 'excess_return': None}]}]
        assert hit_rate_objective(m) == 0.0

    def test_evaluate_joins_forward_returns(self):
        # Unified path: forward returns live on row['_fwd'][horizon]
        # (written by annotate_snapshot_returns); the evaluator must surface
        # them as excess_return/end_price and leave missing tickers as None.
        from scripts.backtest import _evaluate_params_on_snapshots
        snap = {'date': '2026-03-01', 'results': [
            {'ticker': 'AAA', 'price': 100.0, 'dcf_fv': 150.0,
             '_fwd': {90: {'excess_return': 0.15, 'ret': 0.20,
                           'end_price': 120.0, 'spy_return': 0.05}}},
            {'ticker': 'BBB', 'price': 50.0, 'dcf_fv': 40.0},
        ]}
        metrics = _evaluate_params_on_snapshots([snap], default_params(), [90])
        by_t = {d['ticker']: d for d in metrics[0]['details']}
        assert by_t['AAA']['excess_return'] == pytest.approx(0.15)
        assert by_t['AAA']['end_price'] == pytest.approx(120.0)
        assert by_t['BBB']['excess_return'] is None

    def test_search_space_only_rescorable_params(self):
        from scripts.backtest import SEARCH_SPACE, THRESHOLD_SPACE
        dcf_stage = {'erp', 'blend_trigger', 'blend_dcf_weight',
                     'growth_weight_analyst_lt', 'growth_weight_fundamental',
                     'analyst_haircut', 'margin_trend_sensitivity'}
        assert not dcf_stage & set(SEARCH_SPACE)
        assert not dcf_stage & set(THRESHOLD_SPACE)
        # thresholds are searchable, but opt-in (calibrate --include-thresholds)
        assert 'rating_threshold_buy' in THRESHOLD_SPACE
        assert 'rating_threshold_lean' in THRESHOLD_SPACE


# ======================================================================
# Corpus consistency, readiness, evidence floor, live-rating re-score
# ======================================================================

from datetime import timedelta  # noqa: E402

from scripts.backtest import (  # noqa: E402
    MIN_CONSISTENT_DATE, MIN_EFFECTIVE_N, IC_SIGNIFICANCE_PERIODS,
    FV_ACCURACY_MIN_HORIZON,
    snapshot_is_current, parse_since, filter_snapshot_dates,
    _filter_consistent_snapshots, readiness_report, composite_ic_summary,
    aggregate_buckets, walk_forward_calibrate, analyze_run,
    _evaluate_params_on_snapshots, load_results,
)


def _full_gate_row(**over):
    """A row carrying a non-None value for every current gate field."""
    from scripts.scoring import GATES
    row = {g.field: 1.0 for g in GATES}
    row.update(over)
    return row


class TestSnapshotConsistency:

    def test_current_snapshot_passes(self):
        ok, missing = snapshot_is_current({'results': [_full_gate_row()]})
        assert ok and missing == []

    def test_missing_gate_field_is_reported(self):
        from scripts.scoring import GATES
        field = sorted({g.field for g in GATES})[0]
        row = _full_gate_row()
        row[field] = None
        ok, missing = snapshot_is_current({'results': [row]})
        assert not ok and missing == [field]

    def test_field_present_on_any_row_counts(self):
        # Sparse gates are fine: one row with the value is enough.
        from scripts.scoring import GATES
        field = sorted({g.field for g in GATES})[0]
        sparse = _full_gate_row(); sparse[field] = None
        ok, _ = snapshot_is_current({'results': [sparse, _full_gate_row()]})
        assert ok

    def test_filter_drops_before_since_and_inconsistent(self):
        from scripts.scoring import GATES
        field = sorted({g.field for g in GATES})[0]
        old = {'date': '2026-05-01', 'results': [_full_gate_row()]}
        stale = {'date': '2026-08-01', 'results': [_full_gate_row(**{field: None})]}
        good = {'date': '2026-08-02', 'results': [_full_gate_row()]}
        kept, skipped = _filter_consistent_snapshots([old, stale, good])
        assert kept == [good]
        assert [d for d, _ in skipped] == ['2026-05-01', '2026-08-01']
        assert 'since' in skipped[0][1] and field in skipped[1][1]

    def test_since_none_keeps_old_dates(self):
        old = {'date': '2026-05-01', 'results': [_full_gate_row()]}
        kept, skipped = _filter_consistent_snapshots([old], since=None)
        assert kept == [old] and skipped == []

    def test_parse_since(self):
        assert parse_since(None) == MIN_CONSISTENT_DATE
        assert parse_since('none') is None
        assert parse_since('2026-08-01') == date(2026, 8, 1)

    def test_filter_snapshot_dates(self):
        dates = [date(2026, 6, 1), MIN_CONSISTENT_DATE, date(2026, 8, 1)]
        assert filter_snapshot_dates(dates) == dates[1:]
        assert filter_snapshot_dates(dates, None) == dates

    def test_load_results_skips_replay_files(self, tmp_path):
        (tmp_path / 'results_2026-08-01.json').write_text('{"date": "2026-08-01"}')
        (tmp_path / 'results_2026-08-01_replay.json').write_text('{"date": "x"}')
        assert [r['date'] for r in load_results(str(tmp_path))] == ['2026-08-01']


class TestReadiness:

    def _daily(self, start, n):
        return [start + timedelta(days=i) for i in range(n)]

    def test_effective_n_is_span_over_horizon(self):
        dates = self._daily(date(2026, 7, 3), 120)   # 2026-07-03 .. 2026-10-30
        rep = readiness_report(dates, [30], today=date(2026, 11, 1))
        r = rep[30]
        assert r['n_snapshots'] == 120
        # matured: run_date + 30 <= 11-01 -> up to 10-02 -> 92 snapshots
        assert r['n_matured'] == 92
        assert r['span_days'] == 91 and r['effective_n'] == 4

    def test_milestone_dates(self):
        dates = self._daily(date(2026, 7, 3), 60)
        rep = readiness_report(dates, [30], today=date(2026, 9, 3),
                               train_size=3, test_size=1)
        r = rep[30]
        # k-th spaced date matures at first + k*h
        assert r['date_walk_forward'] == date(2026, 7, 3) + timedelta(days=4 * 30)
        assert r['date_calibrate'] == date(2026, 7, 3) + timedelta(days=MIN_EFFECTIVE_N * 30)
        assert r['date_ic_test'] == date(2026, 7, 3) + timedelta(days=IC_SIGNIFICANCE_PERIODS * 30)

    def test_milestones_none_once_reached(self):
        dates = self._daily(date(2026, 1, 1), 600)
        rep = readiness_report(dates, [30], today=date(2027, 12, 31))
        r = rep[30]
        assert r['effective_n'] >= IC_SIGNIFICANCE_PERIODS
        assert r['date_walk_forward'] is None
        assert r['date_calibrate'] is None and r['date_ic_test'] is None

    def test_empty_dates(self):
        rep = readiness_report([], [30, 90], today=date(2026, 9, 3))
        assert rep[30]['effective_n'] == 0 and rep[90]['date_calibrate'] is None


class TestCompositeIC:

    def _metric(self, run_date, pairs, horizon=30):
        return {'run_date': run_date, 'horizon': horizon, 'details': [
            {'_composite_score': s, 'excess_return': e} for s, e in pairs]}

    def test_per_snapshot_mean_not_pooled(self):
        # Two snapshots with perfect but opposite within-day correlations.
        up = [(i, i * 0.01) for i in range(20)]
        down = [(i, -i * 0.01) for i in range(20)]
        m = [self._metric('2026-07-03', up), self._metric('2026-08-03', down)]
        assert rank_ic_objective(m) == pytest.approx(0.0)
        s = composite_ic_summary(m)[30]
        assert s['n_snapshots'] == 2 and s['mean_ic'] == pytest.approx(0.0)
        assert s['frac_positive'] == pytest.approx(0.5)
        assert s['effective_n'] == 31 // 30 + 1

    def test_pooled_vs_per_snapshot_differ(self):
        # A high-dispersion day with zero IC must not swamp a low-dispersion
        # day with perfect IC: per-snapshot mean is 0.5, pooled would not be.
        good = [(i, i * 0.001) for i in range(20)]
        noise = [(i, ((i * 7) % 20) * 1.0) for i in range(20)]
        m = [self._metric('2026-07-03', good), self._metric('2026-07-04', noise)]
        assert rank_ic_objective(m) == pytest.approx(
            (1.0 + rank_ic_objective([m[1]])) / 2)

    def test_thin_snapshots_skipped(self):
        m = [self._metric('2026-07-03', [(1, 0.1), (2, 0.2)])]
        assert composite_ic_summary(m) == {}
        assert rank_ic_objective(m) == 0.0

    def test_aggregate_buckets(self):
        m = [{'run_date': '2026-07-03', 'horizon': 30, 'details': [
            {'rating': 'BUY', 'excess_return': 0.1},
            {'rating': 'BUY', 'excess_return': -0.05},
            {'rating': 'PASS', 'excess_return': None},
        ]}]
        b = aggregate_buckets(m)[30]
        assert b['BUY']['n'] == 2 and b['BUY']['hit_rate'] == pytest.approx(0.5)
        assert 'PASS' not in b


class TestCompositeObjectiveNoFV:

    def test_fv_term_dropped(self):
        # Identical ratings/returns, wildly different FV accuracy -> same score.
        a = [{'details': [{'rating': 'BUY', 'excess_return': 0.05,
                           'dcf_fv': 100, 'end_price': 100}]}]
        b = [{'details': [{'rating': 'BUY', 'excess_return': 0.05,
                           'dcf_fv': 100, 'end_price': 10}]}]
        assert composite_objective(a) == composite_objective(b)
        assert composite_objective(a) == pytest.approx(0.6 * 1.0 + 0.4 * 0.5)


class _NoNetClient:
    def fetch_history(self, *a, **k):
        return None


class TestFVAccuracyHorizon:

    def _snapshot(self):
        rows = []
        for i in range(8):
            rows.append({'ticker': f'T{i}', 'rating': 'HOLD', 'dcf_fv': 100.0,
                         '_gates_passed_num': i, 'price': 90.0,
                         '_fwd': {90: {'excess_return': 0.01 * i, 'ret': 0.02 * i,
                                       'end_price': 100.0 + i, 'spy_return': 0.0,
                                       'start_price': 90.0},
                                  365: {'excess_return': 0.01 * i, 'ret': 0.02 * i,
                                        'end_price': 100.0 + i, 'spy_return': 0.0,
                                        'start_price': 90.0}}})
        return {'date': '2026-07-03', 'results': rows}

    def test_no_fv_metrics_below_min_horizon(self, tmp_path):
        snap = self._snapshot()
        # annotate_snapshot_returns reads the sidecar cache first; none exists,
        # and the no-network client yields nothing, so the pre-set _fwd stands.
        out = analyze_run(snap, 90, _NoNetClient(), cache_dir=str(tmp_path),
                          today=date(2027, 1, 1))
        assert out['n_stocks'] == 8 and out['fv_metrics'] is None
        assert out['details'][0]['_composite_score'] is None

    def test_fv_metrics_at_year_horizon(self, tmp_path):
        assert FV_ACCURACY_MIN_HORIZON == 365
        out = analyze_run(self._snapshot(), 365, _NoNetClient(),
                          cache_dir=str(tmp_path), today=date(2027, 8, 1))
        assert out['fv_metrics'] is not None and out['fv_metrics']['n'] == 8


class TestEvaluateUsesLiveRatingPath:

    def test_caps_apply_to_rescored_rating(self):
        # Two rows with identical (BUY-level) scores; one carries a Beneish
        # flag. The live pipeline caps it at HOLD, so the evaluator must too.
        base = _full_gate_row(price=100.0, _fv_effective=200.0, mos=0.5,
                              edgar_history={'years_available': 10},
                              avg_dollar_volume_3m=5e7,
                              _fwd={30: {'excess_return': 0.1, 'end_price': 110.0}})
        clean = dict(base, ticker='CLEAN')
        flagged = dict(base, ticker='FLAG', beneish_flag=True)
        snap = {'date': '2026-07-03', 'results': [clean, flagged]}
        params = default_params()
        params['rating_threshold_buy'] = 0   # any composite -> BUY
        params['rating_threshold_lean'] = -1
        params['rating_threshold_pass'] = -2
        metrics = _evaluate_params_on_snapshots([snap], params, [30])
        by_t = {d['ticker']: d for d in metrics[0]['details']}
        assert by_t['CLEAN']['rating'] == 'BUY'
        assert by_t['FLAG']['rating'] == 'HOLD'
        assert clean['rating_raw'] == 'BUY' and flagged['_rating_cap'] == 'HOLD'


class TestCalibrateEvidenceFloor:

    def _write_corpus(self, tmp_path, start, n_days):
        for i in range(n_days):
            d = start + timedelta(days=i)
            snap = {'date': d.isoformat(), 'count': 1,
                    'results': [_full_gate_row(ticker='AAA', price=1.0)]}
            (tmp_path / f'results_{d.isoformat()}.json').write_text(
                __import__('json').dumps(snap))

    def test_refuses_below_min_effective_n(self, tmp_path):
        # 40 daily snapshots from the consistency floor; at 30d ~2 periods.
        self._write_corpus(tmp_path, MIN_CONSISTENT_DATE, 40)
        res = walk_forward_calibrate(results_dir=str(tmp_path), horizons=[30],
                                     yf_client=_NoNetClient(), prices_dir=None,
                                     cache_dir=str(tmp_path / 'returns'),
                                     today=date(2026, 9, 3))
        assert res['n_windows'] == 0 and res.get('refused') is True
        assert res['effective_independent_obs'] < MIN_EFFECTIVE_N
        assert 'readiness' in res['overall']['error']
        assert res['since'] == MIN_CONSISTENT_DATE.isoformat()

    def test_since_filter_and_inconsistent_snapshots_recorded(self, tmp_path):
        self._write_corpus(tmp_path, MIN_CONSISTENT_DATE, 5)
        # One pre-rebalance file, one post-rebalance file missing a gate field.
        (tmp_path / 'results_2026-05-01.json').write_text(
            '{"date": "2026-05-01", "results": [{"ticker": "AAA"}]}')
        (tmp_path / 'results_2026-07-20.json').write_text(
            '{"date": "2026-07-20", "results": [{"ticker": "AAA"}]}')
        res = walk_forward_calibrate(results_dir=str(tmp_path), horizons=[30],
                                     yf_client=_NoNetClient(), prices_dir=None,
                                     cache_dir=str(tmp_path / 'returns'),
                                     today=date(2026, 9, 3))
        assert res['n_snapshots_before_since'] == 1
        assert res['skipped_inconsistent'] == ['2026-07-20']
        assert res['n_windows'] == 0

    def test_force_passes_floor_but_coverage_guard_still_refuses(self, tmp_path):
        self._write_corpus(tmp_path, MIN_CONSISTENT_DATE, 40)
        res = walk_forward_calibrate(results_dir=str(tmp_path), horizons=[30],
                                     yf_client=_NoNetClient(), prices_dir=None,
                                     cache_dir=str(tmp_path / 'returns'),
                                     today=date(2026, 9, 3), force=True)
        # Past the evidence floor; no price data -> 0 annotated rows guard.
        assert res.get('refused') is None
        assert res['n_windows'] == 0 and res['annotated_rows'] == 0
