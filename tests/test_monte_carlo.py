"""Focused tests for the Monte Carlo simulators and their pipeline wiring.

Covers models.dcf.monte_carlo_dcf and models.ddm.monte_carlo_ddm beyond the
smoke checks in test_dcf.py / test_ddm.py:

- Agreement with the point estimates: with the sampling sigmas collapsed
  (they floor at 0.001 inside the simulators) the median must land on the
  deterministic two-stage value, INCLUDING inside the minimum-spread wall
  where both sides must substitute the same effective terminal growth.
- Structural invariants: percentile ordering, bookkeeping consistency,
  exact homogeneity in cash flow / share count, monotone response to
  leverage / discount rate / growth, convergence at the production
  iteration count, and independence from numpy's global RNG state.
- Input validation: every path that used to crash (ZeroDivisionError,
  IndexError, TypeError) or silently produce a plausible-looking wrong
  number now returns None with a RuntimeWarning, mirroring the point
  estimates' contract.
- Pipeline wiring: run_forward_dcf / run_ddm_valuation surface the MC
  fields consistently with the point estimates they accompany.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from models.dcf import (
    monte_carlo_dcf, two_stage_ev, two_stage_ev_exit_multiple,
    fair_value_per_share,
)
from models.ddm import monte_carlo_ddm, two_stage_ddm


# Sigmas are floored at 0.001 inside the simulators, so "tiny" is not zero:
# the draws still wobble by ~0.1pp, which is why the collapse tests use a
# 1% tolerance rather than exact equality.
TINY_DCF = dict(growth_sigma=1e-9, wacc_sigma=1e-9, tg_sigma=1e-9)
TINY_DDM = dict(g_sigma=1e-9, re_sigma=1e-9, tg_sigma=1e-9)

# (base_fcf, growth, wacc, tg, net_debt, shares)
BENIGN_DCF = (1e9, 0.10, 0.09, 0.03, 5e9, 1e9)
# (dps, g, re, tg)
BENIGN_DDM = (2.0, 0.07, 0.10, 0.03)

PCT_KEYS = ('p10_fv', 'median_fv', 'p90_fv', 'mean_fv', 'std_fv')


def _dcf_point(base_fcf, g, w, tg, net_debt, shares, **kw):
    return fair_value_per_share(two_stage_ev(base_fcf, g, w, tg, **kw), net_debt, shares)


def _quiet(fn, *args, **kwargs):
    """Run fn asserting it emits no RuntimeWarning; return its result."""
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        return fn(*args, **kwargs)


def _invalid(fn, match, *args, **kwargs):
    """Assert fn returns None AND warns with a message matching `match`."""
    with pytest.warns(RuntimeWarning, match=match):
        result = fn(*args, **kwargs)
    assert result is None
    return result


def _assert_bookkeeping(res):
    n = res['n_iterations']
    assert isinstance(res['n_valid'], int) and 1 <= res['n_valid'] <= n
    assert res['n_valid'] == pytest.approx(n * (1 - res['invalid_rate']), abs=1e-6)
    assert 0.0 <= res['clip_rate'] <= 1.0
    assert 0.0 <= res['invalid_rate'] <= 0.9  # >90% invalid returns None instead
    assert res['std_fv'] >= 0
    assert res['cv'] == pytest.approx(res['std_fv'] / res['mean_fv'])
    assert 0 <= res['p10_fv'] <= res['median_fv'] <= res['p90_fv']
    for k in PCT_KEYS:
        assert np.isfinite(res[k]), k


# ===========================================================================
# monte_carlo_dcf
# ===========================================================================

class TestDcfMatchesPointEstimate:
    @pytest.mark.parametrize('args, band_tol', [
        (BENIGN_DCF, 0.10),
        # Levered, thin equity: equity is ~15% of EV, so the residual 0.1pp
        # sigma floor is amplified ~7x in the tails — only pin the median.
        ((100.0, 0.03, 0.10, 0.025, 1200.0, 1000.0), None),
        ((500.0, -0.05, 0.085, 0.02, 200.0, 100.0), 0.10),   # shrinking FCF
        ((100.0, 0.04, 0.055, 0.03, 0.0, 1000.0), 0.10),     # exactly at the 2.5% wall
    ])
    def test_ggm_median_collapses_to_point_estimate(self, args, band_tol):
        res = _quiet(monte_carlo_dcf, *args, n_iterations=1000, **TINY_DCF)
        point = _dcf_point(*args)
        assert res['median_fv'] == pytest.approx(point, rel=1e-2)
        if band_tol is not None:
            # The band collapses too, but the 0.1pp sigma floor still moves
            # the terminal spread by ~5% relative in the thin-spread cases.
            assert res['p10_fv'] == pytest.approx(point, rel=band_tol)
            assert res['p90_fv'] == pytest.approx(point, rel=band_tol)

    def test_inside_min_spread_wall_uses_same_effective_terminal_growth(self):
        """wacc - tg = 2% < 2.5%: two_stage_ev substitutes tg = wacc - 2.5%.
        The MC must clip every terminal-growth draw to the same wall — a
        looser wall here is exactly the bias the clip_rate is meant to flag."""
        args = (100.0, 0.04, 0.05, 0.03, 0.0, 1000.0)
        res = _quiet(monte_carlo_dcf, *args, n_iterations=1000, **TINY_DCF)
        assert res['median_fv'] == pytest.approx(_dcf_point(*args), rel=1e-2)
        # Every tg draw hit the wall, no wacc draw hit its floor → exactly 1/2.
        assert res['clip_rate'] == pytest.approx(0.5)

    def test_exit_multiple_leg_collapses_to_average_of_both_methods(self):
        base_fcf, g, w, tg, nd, sh = BENIGN_DCF
        res = _quiet(monte_carlo_dcf, *BENIGN_DCF, base_ebitda=2e9, exit_multiple=12.0,
                     n_iterations=1000, exit_mult_sigma=1e-9, **TINY_DCF)
        p_ggm = _dcf_point(*BENIGN_DCF)
        p_exit = fair_value_per_share(
            two_stage_ev_exit_multiple(base_fcf, g, w, tg, 2e9, 12.0), nd, sh)
        assert p_exit != pytest.approx(p_ggm, rel=0.05)  # the leg really differs
        assert res['median_fv'] == pytest.approx((p_ggm + p_exit) / 2, rel=1e-2)

    def test_exit_multiple_floor_is_applied_to_the_draws(self):
        """A multiple below the floor is valued AT the floor, matching the
        pipeline's EXIT_MULT_MIN treatment of the point estimate."""
        base_fcf, g, w, tg, nd, sh = BENIGN_DCF
        res = _quiet(monte_carlo_dcf, *BENIGN_DCF, base_ebitda=2e9, exit_multiple=4.0,
                     exit_mult_floor=5.0, n_iterations=500, exit_mult_sigma=1e-9,
                     **TINY_DCF)
        p_ggm = _dcf_point(*BENIGN_DCF)
        p_exit_at_floor = fair_value_per_share(
            two_stage_ev_exit_multiple(base_fcf, g, w, tg, 2e9, 5.0), nd, sh)
        assert res['median_fv'] == pytest.approx((p_ggm + p_exit_at_floor) / 2, rel=1e-2)

    def test_stage1_equal_to_total_years_skips_the_fade(self):
        args = (100.0, 0.08, 0.10, 0.03, 0.0, 10.0)
        res = _quiet(monte_carlo_dcf, *args, total_years=5, stage1_years=5,
                     n_iterations=500, **TINY_DCF)
        point = _dcf_point(*args, total_years=5, stage1_years=5)
        assert res['median_fv'] == pytest.approx(point, rel=1e-2)

    @pytest.mark.parametrize('args', [
        BENIGN_DCF,
        (100.0, 0.03, 0.10, 0.025, 0.0, 1000.0),
        (500.0, 0.08, 0.085, 0.03, 200.0, 100.0),
    ])
    def test_default_sigma_median_tracks_point_estimate(self, args):
        """With production-style sigmas the distribution is right-skewed, so
        the mean drifts above the point but the median should stay on it."""
        res = _quiet(monte_carlo_dcf, *args, n_iterations=2000)
        point = _dcf_point(*args)
        assert res['median_fv'] == pytest.approx(point, rel=0.05)
        assert res['mean_fv'] == pytest.approx(point, rel=0.10)
        assert res['p10_fv'] < point < res['p90_fv']


class TestDcfInvariants:
    def test_bookkeeping_consistent(self):
        res = _quiet(monte_carlo_dcf, *BENIGN_DCF, n_iterations=500)
        assert res['n_iterations'] == 500
        _assert_bookkeeping(res)

    def test_bookkeeping_consistent_with_wipeouts(self):
        res = _quiet(monte_carlo_dcf, 100.0, 0.03, 0.10, 0.025, 1200.0, 1000.0,
                     n_iterations=800)
        assert res['invalid_rate'] > 0
        _assert_bookkeeping(res)

    def test_exactly_homogeneous_in_base_fcf_and_shares(self):
        """Same seed → same draws, so scaling cash flow or share count must
        scale every $/share statistic exactly (and leave the CV untouched)."""
        base = _quiet(monte_carlo_dcf, 100.0, 0.05, 0.09, 0.03, 0.0, 10.0, n_iterations=300)
        double_fcf = _quiet(monte_carlo_dcf, 200.0, 0.05, 0.09, 0.03, 0.0, 10.0, n_iterations=300)
        half_shares = _quiet(monte_carlo_dcf, 100.0, 0.05, 0.09, 0.03, 0.0, 5.0, n_iterations=300)
        for k in PCT_KEYS:
            assert double_fcf[k] == pytest.approx(2 * base[k], rel=1e-12)
            assert half_shares[k] == pytest.approx(2 * base[k], rel=1e-12)
        assert double_fcf['cv'] == pytest.approx(base['cv'], rel=1e-12)
        assert half_shares['cv'] == pytest.approx(base['cv'], rel=1e-12)

    def test_median_monotone_decreasing_in_net_debt(self):
        meds = [monte_carlo_dcf(100.0, 0.05, 0.09, 0.03, nd, 10.0, n_iterations=300)['median_fv']
                for nd in (0.0, 200.0, 500.0, 900.0)]
        assert all(a > b for a, b in zip(meds, meds[1:], strict=False))

    def test_median_monotone_decreasing_in_discount_rate(self):
        meds = [monte_carlo_dcf(100.0, 0.05, w, 0.03, 0.0, 10.0, n_iterations=300)['median_fv']
                for w in (0.07, 0.09, 0.11, 0.14)]
        assert all(a > b for a, b in zip(meds, meds[1:], strict=False))

    def test_median_monotone_increasing_in_growth(self):
        meds = [monte_carlo_dcf(100.0, g, 0.09, 0.03, 0.0, 10.0, n_iterations=300,
                                growth_sigma=0.02)['median_fv']
                for g in (-0.05, 0.0, 0.05, 0.10)]
        assert all(a < b for a, b in zip(meds, meds[1:], strict=False))

    def test_wider_terminal_growth_sigma_widens_band(self):
        narrow = monte_carlo_dcf(*BENIGN_DCF, n_iterations=500, tg_sigma=0.001)
        wide = monte_carlo_dcf(*BENIGN_DCF, n_iterations=500, tg_sigma=0.015)
        assert (wide['p90_fv'] - wide['p10_fv']) > (narrow['p90_fv'] - narrow['p10_fv'])

    @pytest.mark.parametrize('args', [BENIGN_DCF, (100.0, 0.03, 0.10, 0.025, 0.0, 1000.0)])
    def test_production_iteration_count_is_converged(self, args):
        """config.MC_ITERATIONS = 250 is justified by convergence; pin it."""
        small = monte_carlo_dcf(*args, n_iterations=250)
        large = monte_carlo_dcf(*args, n_iterations=5000)
        for k in ('median_fv', 'p10_fv', 'p90_fv'):
            assert small[k] == pytest.approx(large[k], rel=0.05), k

    def test_essentially_no_clipping_for_benign_inputs(self):
        """tg - wacc ~ N(-6pp, 1.1pp): the 2.5% wall is a 3-sigma event, so
        at most a draw or two in a thousand should touch it."""
        res = monte_carlo_dcf(*BENIGN_DCF, n_iterations=1000)
        assert res['clip_rate'] < 0.005
        assert res['invalid_rate'] == 0.0

    def test_clip_rate_counts_wacc_floor_hits(self):
        """WACC ~ N(3.1%, 1pp) → ~46% of draws fall below the 3% floor. The
        tg wall is checked AFTER flooring, so with tg = 0 it never fires:
        clip_rate ≈ 0.46 / 2."""
        res = monte_carlo_dcf(100.0, 0.02, 0.031, 0.0, 0.0, 1000.0, n_iterations=4000,
                              wacc_sigma=0.01, tg_sigma=1e-9, growth_sigma=1e-9)
        assert 0.20 < res['clip_rate'] < 0.27
        assert res['median_fv'] > 0

    def test_returns_none_when_nearly_every_draw_is_insolvent(self):
        assert monte_carlo_dcf(100.0, 0.03, 0.10, 0.025, 1e6, 1000.0, n_iterations=300) is None

    def test_independent_of_global_numpy_rng_state(self):
        np.random.seed(1)
        r1 = monte_carlo_dcf(*BENIGN_DCF, n_iterations=200)
        np.random.seed(999)
        np.random.random(37)
        r2 = monte_carlo_dcf(*BENIGN_DCF, n_iterations=200)
        assert r1 == r2

    def test_single_iteration_is_degenerate_but_well_formed(self):
        res = _quiet(monte_carlo_dcf, *BENIGN_DCF, n_iterations=1)
        assert res['n_valid'] == 1
        assert res['p10_fv'] == res['median_fv'] == res['p90_fv']
        assert res['std_fv'] == 0.0 and res['cv'] == 0.0


class TestDcfInputValidation:
    @pytest.mark.parametrize('bad', [0, -5, 250.0, True, None])
    def test_bad_iteration_count_returns_none_with_warning(self, bad):
        _invalid(monte_carlo_dcf, 'n_iterations', *BENIGN_DCF, n_iterations=bad)

    @pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf'), 'abc'])
    def test_bad_net_debt_returns_none_with_warning(self, bad):
        base_fcf, g, w, tg, _, sh = BENIGN_DCF
        _invalid(monte_carlo_dcf, 'net_debt', base_fcf, g, w, tg, bad, sh, n_iterations=100)

    def test_none_net_debt_means_zero(self):
        base_fcf, g, w, tg, _, sh = BENIGN_DCF
        assert (monte_carlo_dcf(base_fcf, g, w, tg, None, sh, n_iterations=200)
                == monte_carlo_dcf(base_fcf, g, w, tg, 0.0, sh, n_iterations=200))

    @pytest.mark.parametrize('years_kw', [
        dict(total_years=0), dict(total_years=10.0), dict(stage1_years=11),
        dict(stage1_years=-1), dict(stage1_years=5.0),
    ])
    def test_bad_projection_years_return_none_with_warning(self, years_kw):
        _invalid(monte_carlo_dcf, 'years', *BENIGN_DCF, n_iterations=100, **years_kw)

    @pytest.mark.parametrize('override, match', [
        (dict(base_fcf=float('nan')), 'base_fcf'),
        (dict(growth_rate=2.0), 'growth_rate'),
        (dict(discount_rate=0.60), 'discount_rate'),
        (dict(discount_rate=0.0), 'discount_rate'),
        (dict(terminal_growth=0.20), 'terminal_growth'),
        (dict(shares_outstanding=None), 'shares_outstanding'),
    ])
    def test_core_input_bounds_match_point_estimate(self, override, match):
        kw = dict(zip(('base_fcf', 'growth_rate', 'discount_rate', 'terminal_growth',
                       'net_debt', 'shares_outstanding'), BENIGN_DCF, strict=True))
        kw.update(override)
        _invalid(monte_carlo_dcf, match, n_iterations=100, **kw)

    @pytest.mark.parametrize('exit_multiple, base_ebitda', [
        (-3.0, 2e9), (0.0, 2e9), (float('nan'), 2e9), (float('inf'), 2e9),
        (10.0, float('nan')), ('12x', 2e9),
    ])
    def test_garbage_exit_inputs_drop_the_leg_with_warning(self, exit_multiple, base_ebitda):
        """Previously a 0 / negative / NaN multiple was floored (or NaN'd to
        zero) and silently blended in. Now the leg is skipped loudly and the
        result is the GGM-only distribution."""
        ggm_only = _quiet(monte_carlo_dcf, *BENIGN_DCF, n_iterations=200)
        with pytest.warns(RuntimeWarning, match='exit-multiple leg skipped'):
            res = monte_carlo_dcf(*BENIGN_DCF, n_iterations=200,
                                  base_ebitda=base_ebitda, exit_multiple=exit_multiple)
        assert res == ggm_only

    @pytest.mark.parametrize('exit_multiple, base_ebitda', [
        (12.0, None), (None, 2e9), (None, None), (12.0, 0.0), (12.0, -5e8),
    ])
    def test_absent_or_nonpositive_ebitda_skips_leg_quietly(self, exit_multiple, base_ebitda):
        ggm_only = _quiet(monte_carlo_dcf, *BENIGN_DCF, n_iterations=200)
        res = _quiet(monte_carlo_dcf, *BENIGN_DCF, n_iterations=200,
                     base_ebitda=base_ebitda, exit_multiple=exit_multiple)
        assert res == ggm_only

    def test_valid_exit_leg_actually_changes_the_result(self):
        ggm_only = monte_carlo_dcf(*BENIGN_DCF, n_iterations=200)
        blended = monte_carlo_dcf(*BENIGN_DCF, n_iterations=200,
                                  base_ebitda=2e9, exit_multiple=12.0)
        assert blended['median_fv'] != pytest.approx(ggm_only['median_fv'], rel=1e-3)


# ===========================================================================
# monte_carlo_ddm
# ===========================================================================

class TestDdmMatchesPointEstimate:
    @pytest.mark.parametrize('args', [
        BENIGN_DDM,
        (2.0, -0.02, 0.08, 0.02),     # shrinking dividend
        (2.0, 0.05, 0.05, 0.03),      # exactly at the 2% wall
        (2.0, 0.05, 0.045, 0.03),     # inside the wall: effective tg = re - 2%
        (2.0, 0.05, 0.04, 0.03),      # deeper inside the wall
        (0.5, 0.12, 0.15, 0.04),
    ])
    def test_median_collapses_to_two_stage_point_estimate(self, args):
        dps, g, re, tg = args
        res = _quiet(monte_carlo_ddm, dps, g, re, tg, n=1000, **TINY_DDM)
        point = two_stage_ddm(dps, g, tg, re)
        assert res['median_fv'] == pytest.approx(point, rel=1e-2)

    def test_spread_wall_no_longer_inflates_the_upper_tail(self):
        """Regression: with re - tg sitting exactly on the 2% wall, the old
        1% MC wall let ~half the draws capitalise the terminal dividend on a
        spread the point estimate never uses — p90 came out at ~2x the point
        value. The MC wall now matches two_stage_ddm's min_spread."""
        dps, g, re, tg = 2.0, 0.05, 0.05, 0.03
        res = _quiet(monte_carlo_ddm, dps, g, re, tg, n=5000)
        point = two_stage_ddm(dps, g, tg, re)
        assert res['clip_rate'] > 0.2                      # the wall IS binding
        assert res['median_fv'] == pytest.approx(point, rel=0.10)
        assert res['p90_fv'] < 1.25 * point

    def test_default_sigma_median_tracks_point_estimate(self):
        dps, g, re, tg = BENIGN_DDM
        res = _quiet(monte_carlo_ddm, dps, g, re, tg, n=2000)
        point = two_stage_ddm(dps, g, tg, re)
        assert res['median_fv'] == pytest.approx(point, rel=0.05)
        assert res['p10_fv'] < point < res['p90_fv']

    def test_projection_years_are_honoured(self):
        dps, g, re, tg = BENIGN_DDM
        for years in (1, 3, 8):
            res = _quiet(monte_carlo_ddm, dps, g, re, tg, n=500, years=years, **TINY_DDM)
            assert res['median_fv'] == pytest.approx(two_stage_ddm(dps, g, tg, re, years=years),
                                                     rel=1e-2)


class TestDdmInvariants:
    def test_bookkeeping_consistent(self):
        res = _quiet(monte_carlo_ddm, *BENIGN_DDM, n=500)
        assert res['n_iterations'] == 500
        _assert_bookkeeping(res)

    def test_exactly_homogeneous_in_dps(self):
        base = _quiet(monte_carlo_ddm, 2.0, 0.07, 0.10, 0.03, n=300)
        double = _quiet(monte_carlo_ddm, 4.0, 0.07, 0.10, 0.03, n=300)
        for k in PCT_KEYS:
            assert double[k] == pytest.approx(2 * base[k], rel=1e-12)
        assert double['cv'] == pytest.approx(base['cv'], rel=1e-12)

    def test_median_monotone_decreasing_in_cost_of_equity(self):
        meds = [monte_carlo_ddm(2.0, 0.05, re, 0.03, n=300)['median_fv']
                for re in (0.07, 0.09, 0.11, 0.14)]
        assert all(a > b for a, b in zip(meds, meds[1:], strict=False))

    def test_median_monotone_increasing_in_growth(self):
        meds = [monte_carlo_ddm(2.0, g, 0.10, 0.03, n=300, g_sigma=0.01)['median_fv']
                for g in (-0.05, 0.0, 0.05, 0.10)]
        assert all(a < b for a, b in zip(meds, meds[1:], strict=False))

    def test_wider_terminal_growth_sigma_widens_band(self):
        narrow = monte_carlo_ddm(*BENIGN_DDM, n=500, tg_sigma=0.001)
        wide = monte_carlo_ddm(*BENIGN_DDM, n=500, tg_sigma=0.015)
        assert (wide['p90_fv'] - wide['p10_fv']) > (narrow['p90_fv'] - narrow['p10_fv'])

    def test_production_iteration_count_is_converged(self):
        small = monte_carlo_ddm(*BENIGN_DDM, n=250)
        large = monte_carlo_ddm(*BENIGN_DDM, n=5000)
        for k in ('median_fv', 'p10_fv', 'p90_fv'):
            assert small[k] == pytest.approx(large[k], rel=0.05), k

    def test_no_clipping_for_benign_inputs(self):
        res = monte_carlo_ddm(*BENIGN_DDM, n=1000)
        assert res['clip_rate'] == 0.0
        assert res['invalid_rate'] == 0.0

    def test_independent_of_global_numpy_rng_state(self):
        np.random.seed(1)
        r1 = monte_carlo_ddm(*BENIGN_DDM, n=200)
        np.random.seed(999)
        np.random.random(37)
        r2 = monte_carlo_ddm(*BENIGN_DDM, n=200)
        assert r1 == r2


class TestDdmInputValidation:
    @pytest.mark.parametrize('override, match', [
        (dict(n=0), r'\bn\b'),
        (dict(n=-1), r'\bn\b'),
        (dict(n=200.0), r'\bn\b'),
        (dict(years=0), 'years'),
        (dict(years=5.0), 'years'),
        (dict(dps=0.0), 'dps'),
        (dict(dps=float('nan')), 'dps'),
        (dict(g=float('nan')), r'\bg\b'),
        (dict(g=5.0), r'\bg\b'),
        (dict(g=-0.9), r'\bg\b'),
        (dict(re=None), r'\bre\b'),
        (dict(re=float('nan')), r'\bre\b'),
        (dict(re=-0.05), r'\bre\b'),
        (dict(re=0.0), r'\bre\b'),
        (dict(re=0.55), r'\bre\b'),
        (dict(tg=float('nan')), r'\btg\b'),
        (dict(tg=0.20), r'\btg\b'),
        (dict(re=0.02, tg=0.03), 're <= tg'),
        (dict(re=0.03, tg=0.03), 're <= tg'),
    ])
    def test_bad_inputs_return_none_with_warning(self, override, match):
        """Each of these previously either crashed (n=0 → ZeroDivisionError,
        years=0 → IndexError, re=None → TypeError) or returned a confident
        number: tg=NaN silently DROPPED the terminal value, re <= tg and
        re < 0 clipped every draw to the walls and reported a distribution
        around a point estimate that is itself undefined."""
        kw = dict(dps=2.0, g=0.07, re=0.10, tg=0.03, n=200)
        kw.update(override)
        _invalid(monte_carlo_ddm, match, **kw)

    @pytest.mark.parametrize('re, tg', [
        (0.02, 0.03), (0.03, 0.03), (0.031, 0.03), (0.05, 0.03), (0.10, 0.03),
    ])
    def test_defined_exactly_when_point_estimate_is_defined(self, re, tg):
        point = two_stage_ddm(2.0, 0.05, tg, re)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            mc = monte_carlo_ddm(2.0, 0.05, re, tg, n=200)
        assert (mc is None) == (point is None)


# ===========================================================================
# Pipeline wiring
# ===========================================================================

def _dcf_yf_data(shares=1_000.0, with_ebitda=False):
    years = pd.to_datetime([f'{2021 + i}-12-31' for i in range(4)])
    cf = pd.DataFrame({y: {'Free Cash Flow': 1000.0, 'Operating Cash Flow': 1200.0,
                           'Depreciation And Amortization': 300.0,
                           'Capital Expenditure': -200.0,
                           'Stock Based Compensation': 0.0} for y in years})
    inc_rows = {'Total Revenue': 10_000.0}
    if with_ebitda:
        inc_rows['Operating Income'] = 1_500.0
    inc = pd.DataFrame({y: inc_rows for y in years})
    info = {'marketCap': 20_000.0, 'sharesOutstanding': shares,
            'totalDebt': 0.0, 'totalCash': 0.0, 'currentPrice': 20.0}
    return {'cash_flow': cf, 'income_statement': inc, 'info': info}


class TestForwardDcfWiring:
    def test_mc_band_replaces_sensitivity_grid_and_brackets_point_fv(self):
        from scripts.analyze_stock import run_forward_dcf
        from scripts.config import MC_ITERATIONS
        fv, sens_range, _, diag, mc = run_forward_dcf(_dcf_yf_data(), wacc=0.10)
        assert fv is not None and mc is not None
        assert mc['n_iterations'] == MC_ITERATIONS
        assert sens_range == (mc['p10_fv'], mc['p90_fv'])
        assert diag['mc_result'] is mc
        assert mc['p10_fv'] <= fv <= mc['p90_fv']
        assert mc['median_fv'] == pytest.approx(fv, rel=0.10)

    def test_exit_multiple_leg_flows_into_mc(self):
        """With EBITDA available the MC blends the exit leg, so its median
        should track the blended point FV rather than the GGM-only value."""
        from scripts.analyze_stock import run_forward_dcf
        fv_ggm_only, _, _, _, mc_ggm = run_forward_dcf(_dcf_yf_data(), wacc=0.10,
                                                       exit_multiple=14.0)
        fv_blend, _, _, diag, mc_blend = run_forward_dcf(
            _dcf_yf_data(with_ebitda=True), wacc=0.10, exit_multiple=14.0)
        assert diag['exit_mult_fv'] is not None
        assert fv_blend != pytest.approx(fv_ggm_only, rel=1e-3)
        assert mc_blend['median_fv'] != pytest.approx(mc_ggm['median_fv'], rel=1e-3)
        assert mc_blend['median_fv'] == pytest.approx(fv_blend, rel=0.10)

    def test_no_shares_means_no_mc(self):
        from scripts.analyze_stock import run_forward_dcf
        fv, sens_range, _, diag, mc = run_forward_dcf(_dcf_yf_data(shares=None), wacc=0.10)
        assert fv is None and mc is None and sens_range is None
        assert diag['mc_result'] is None

    def test_wacc_sigma_override_widens_band(self):
        from scripts.analyze_stock import run_forward_dcf
        _, narrow, _, _, _ = run_forward_dcf(_dcf_yf_data(), wacc=0.10, wacc_sigma=0.002)
        _, wide, _, _, _ = run_forward_dcf(_dcf_yf_data(), wacc=0.10, wacc_sigma=0.03)
        assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def _dividend_series(years=('2021', '2022', '2023', '2024', '2025'), quarterly=0.50, growth=0.05):
    rows = []
    for i, y in enumerate(years):
        q = quarterly * (1 + growth) ** i
        rows += [(f'{y}-03-15', q), (f'{y}-06-15', q), (f'{y}-09-15', q), (f'{y}-12-15', q)]
    return pd.Series([v for _, v in rows], index=pd.to_datetime([d for d, _ in rows]))


def _ddm_yf_data():
    return {'info': {'trailingEps': 5.0, 'dividendRate': 2.4, 'payoutRatio': 0.48}}


class TestDdmValuationWiring:
    def test_mc_fields_populated_and_consistent_with_point_estimate(self):
        from scripts.analyze_stock import run_ddm_valuation
        from scripts.config import DDM_HIGH_GROWTH_YEARS, TERMINAL_GROWTH_RATE
        res = run_ddm_valuation(_ddm_yf_data(), _dividend_series(), cost_of_equity=0.09)
        assert res['ddm_eligible'], res['ddm_reason']
        assert res['ddm_fv'] is not None
        assert res['ddm_mc_p10'] <= res['ddm_mc_median'] <= res['ddm_mc_p90']
        assert res['ddm_mc_cv'] > 0
        point = two_stage_ddm(2.4, res['ddm_growth'], TERMINAL_GROWTH_RATE, 0.09,
                              years=DDM_HIGH_GROWTH_YEARS)
        assert res['ddm_mc_median'] == pytest.approx(point, rel=0.10)
        assert res['ddm_mc_p10'] < point < res['ddm_mc_p90']

    def test_undefined_point_estimate_leaves_mc_fields_empty(self):
        """cost_of_equity <= terminal growth: the two-stage DDM is undefined
        (ddm_fv None). The MC used to fill ddm_mc_* anyway by clipping every
        draw to its walls; it must now stay empty alongside the point FV."""
        from scripts.analyze_stock import run_ddm_valuation
        from scripts.config import TERMINAL_GROWTH_RATE
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            res = run_ddm_valuation(_ddm_yf_data(), _dividend_series(),
                                    cost_of_equity=TERMINAL_GROWTH_RATE - 0.005)
        assert res['ddm_eligible']
        assert res['ddm_fv'] is None
        for k in ('ddm_mc_median', 'ddm_mc_p10', 'ddm_mc_p90', 'ddm_mc_cv'):
            assert res[k] is None, k

    def test_ineligible_payer_has_no_mc_fields(self):
        from scripts.analyze_stock import run_ddm_valuation
        res = run_ddm_valuation({'info': {'trailingEps': 5.0, 'dividendRate': 0.0}},
                                _dividend_series(), cost_of_equity=0.09)
        assert not res['ddm_eligible']
        for k in ('ddm_mc_median', 'ddm_mc_p10', 'ddm_mc_p90', 'ddm_mc_cv'):
            assert res[k] is None, k
