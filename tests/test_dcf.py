# tests/test_dcf.py
import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.dcf import (
    two_stage_ev, fair_value_per_share, dcf_sensitivity,
    two_stage_ev_exit_multiple, monte_carlo_dcf, reverse_dcf,
)


# ---------------------------------------------------------------------------
# two_stage_ev
# ---------------------------------------------------------------------------

class TestTwoStageEV:
    def test_basic_positive(self):
        """Standard inputs should produce a positive EV."""
        ev = two_stage_ev(1e9, 0.10, 0.09, 0.03)
        assert ev is not None
        assert ev > 0

    def test_higher_growth_gives_higher_ev(self):
        """Higher growth rate should yield higher EV, all else equal."""
        ev_low = two_stage_ev(1e9, 0.05, 0.09, 0.03)
        ev_high = two_stage_ev(1e9, 0.15, 0.09, 0.03)
        assert ev_high > ev_low

    def test_higher_discount_gives_lower_ev(self):
        """Higher discount rate should yield lower EV."""
        ev_low_disc = two_stage_ev(1e9, 0.10, 0.08, 0.03)
        ev_high_disc = two_stage_ev(1e9, 0.10, 0.12, 0.03)
        assert ev_low_disc > ev_high_disc

    def test_negative_fcf_returns_none(self):
        """Negative base FCF should return None."""
        ev = two_stage_ev(-1e9, 0.10, 0.09, 0.03)
        assert ev is None

    def test_zero_fcf_returns_none(self):
        """Zero base FCF should return None."""
        ev = two_stage_ev(0, 0.10, 0.09, 0.03)
        assert ev is None

    def test_min_spread_enforced(self):
        """When discount_rate - terminal_growth < min_spread, effective_tg adjusts."""
        # WACC = 0.04, terminal = 0.03, spread = 0.01 < default 0.025
        # Should still produce a valid (finite) result
        ev = two_stage_ev(1e9, 0.08, 0.04, 0.03)
        assert ev is not None
        assert ev > 0
        assert np.isfinite(ev)

    def test_custom_years(self):
        """Non-default year parameters should work."""
        ev = two_stage_ev(1e9, 0.10, 0.09, 0.03, total_years=15, stage1_years=7)
        assert ev is not None
        assert ev > 0


# ---------------------------------------------------------------------------
# fair_value_per_share
# ---------------------------------------------------------------------------

class TestFairValuePerShare:
    def test_basic(self):
        """FV = (EV - Net Debt) / Shares."""
        fv = fair_value_per_share(50e9, 5e9, 1e9)
        assert fv == pytest.approx(45.0)

    def test_net_debt_exceeds_ev(self):
        """When net debt > EV, equity value is negative → None."""
        fv = fair_value_per_share(10e9, 15e9, 1e9)
        assert fv is None

    def test_zero_shares(self):
        """Zero shares → None."""
        fv = fair_value_per_share(50e9, 5e9, 0)
        assert fv is None

    def test_none_ev(self):
        """None EV → None."""
        fv = fair_value_per_share(None, 5e9, 1e9)
        assert fv is None

    def test_negative_net_debt(self):
        """Negative net debt (net cash) → higher equity value."""
        fv = fair_value_per_share(50e9, -10e9, 1e9)
        assert fv == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# dcf_sensitivity
# ---------------------------------------------------------------------------

class TestDCFSensitivity:
    def test_returns_dict(self):
        """Should return a dict with 25 entries."""
        result = dcf_sensitivity(1e9, 0.10, 0.09, 0.03, 5e9, 1e9)
        assert isinstance(result, dict)
        assert len(result) == 25

    def test_center_matches_direct(self):
        """Center of sensitivity grid (0,0) should match direct calculation."""
        ev = two_stage_ev(1e9, 0.10, 0.09, 0.03, total_years=10, stage1_years=5)
        fv_direct = fair_value_per_share(ev, 5e9, 1e9)

        grid = dcf_sensitivity(1e9, 0.10, 0.09, 0.03, 5e9, 1e9,
                               years=10, stage1=5)
        fv_center = grid.get((0.0, 0.0))
        assert fv_center == pytest.approx(fv_direct, rel=1e-6)

    def test_lower_wacc_gives_higher_fv(self):
        """In the grid, lower WACC → higher FV."""
        grid = dcf_sensitivity(1e9, 0.10, 0.09, 0.03, 5e9, 1e9)
        fv_low_wacc = grid.get((-0.01, 0.0))
        fv_high_wacc = grid.get((0.01, 0.0))
        if fv_low_wacc and fv_high_wacc:
            assert fv_low_wacc > fv_high_wacc

    def test_higher_growth_gives_higher_fv(self):
        """In the grid, higher terminal growth → higher FV."""
        grid = dcf_sensitivity(1e9, 0.10, 0.09, 0.03, 5e9, 1e9)
        fv_low_g = grid.get((0.0, -0.005))
        fv_high_g = grid.get((0.0, 0.005))
        if fv_low_g and fv_high_g:
            assert fv_high_g > fv_low_g


# ---------------------------------------------------------------------------
# two_stage_ev_exit_multiple
# ---------------------------------------------------------------------------

class TestExitMultipleTV:
    def test_basic_positive(self):
        """Standard inputs should produce a positive EV."""
        ev = two_stage_ev_exit_multiple(
            base_fcf=1e9, growth_rate=0.10, discount_rate=0.09,
            terminal_growth=0.03, base_ebitda=2e9, exit_multiple=12.0)
        assert ev is not None
        assert ev > 0

    def test_higher_exit_multiple_gives_higher_ev(self):
        """Higher exit multiple → higher EV."""
        ev_low = two_stage_ev_exit_multiple(
            1e9, 0.10, 0.09, 0.03, 2e9, 8.0)
        ev_high = two_stage_ev_exit_multiple(
            1e9, 0.10, 0.09, 0.03, 2e9, 20.0)
        assert ev_high > ev_low

    def test_none_on_missing_ebitda(self):
        """Missing base_ebitda → None."""
        ev = two_stage_ev_exit_multiple(
            1e9, 0.10, 0.09, 0.03, None, 12.0)
        assert ev is None

    def test_none_on_negative_ebitda(self):
        """Negative base_ebitda → None."""
        ev = two_stage_ev_exit_multiple(
            1e9, 0.10, 0.09, 0.03, -2e9, 12.0)
        assert ev is None

    def test_none_on_zero_fcf(self):
        """Zero base_fcf → None."""
        ev = two_stage_ev_exit_multiple(
            0, 0.10, 0.09, 0.03, 2e9, 12.0)
        assert ev is None

    def test_comparable_to_ggm(self):
        """Exit-multiple EV should be in the same order of magnitude as GGM EV."""
        ev_ggm = two_stage_ev(1e9, 0.10, 0.09, 0.03)
        ev_exit = two_stage_ev_exit_multiple(
            1e9, 0.10, 0.09, 0.03, 2e9, 12.0)
        assert ev_ggm is not None and ev_exit is not None
        ratio = ev_exit / ev_ggm
        assert 0.1 < ratio < 10  # within an order of magnitude

    def test_higher_growth_gives_higher_ev(self):
        """Higher growth rate → higher EV."""
        ev_low = two_stage_ev_exit_multiple(
            1e9, 0.05, 0.09, 0.03, 2e9, 12.0)
        ev_high = two_stage_ev_exit_multiple(
            1e9, 0.15, 0.09, 0.03, 2e9, 12.0)
        assert ev_high > ev_low


# ---------------------------------------------------------------------------
# monte_carlo_dcf
# ---------------------------------------------------------------------------

class TestMonteCarloSim:
    def test_basic_returns_dict(self):
        """Standard inputs should return a dict with expected keys."""
        result = monte_carlo_dcf(
            base_fcf=1e9, growth_rate=0.10, discount_rate=0.09,
            terminal_growth=0.03, net_debt=5e9, shares_outstanding=1e9,
            n_iterations=500)
        assert result is not None
        assert isinstance(result, dict)
        for key in ('median_fv', 'mean_fv', 'p10_fv', 'p90_fv',
                     'std_fv', 'cv', 'n_valid'):
            assert key in result

    def test_p10_below_median_below_p90(self):
        """p10 <= median <= p90."""
        result = monte_carlo_dcf(
            1e9, 0.10, 0.09, 0.03, 5e9, 1e9, n_iterations=500)
        assert result['p10_fv'] <= result['median_fv'] <= result['p90_fv']

    def test_positive_fair_values(self):
        """All fair value percentiles should be positive."""
        result = monte_carlo_dcf(
            1e9, 0.10, 0.09, 0.03, 5e9, 1e9, n_iterations=500)
        assert result['p10_fv'] > 0
        assert result['median_fv'] > 0
        assert result['p90_fv'] > 0

    def test_reproducible_with_fixed_seed(self):
        """Same inputs → same outputs (deterministic)."""
        r1 = monte_carlo_dcf(1e9, 0.10, 0.09, 0.03, 5e9, 1e9, n_iterations=200)
        r2 = monte_carlo_dcf(1e9, 0.10, 0.09, 0.03, 5e9, 1e9, n_iterations=200)
        assert r1['median_fv'] == pytest.approx(r2['median_fv'])

    def test_none_on_negative_fcf(self):
        """Negative FCF → None."""
        result = monte_carlo_dcf(
            -1e9, 0.10, 0.09, 0.03, 5e9, 1e9)
        assert result is None

    def test_none_on_zero_shares(self):
        """Zero shares → None."""
        result = monte_carlo_dcf(
            1e9, 0.10, 0.09, 0.03, 5e9, 0)
        assert result is None

    def test_cv_positive(self):
        """Coefficient of variation should be positive."""
        result = monte_carlo_dcf(
            1e9, 0.10, 0.09, 0.03, 5e9, 1e9, n_iterations=500)
        assert result['cv'] > 0

    def test_with_exit_multiple(self):
        """Including exit multiple should still return valid results."""
        result = monte_carlo_dcf(
            base_fcf=1e9, growth_rate=0.10, discount_rate=0.09,
            terminal_growth=0.03, net_debt=5e9, shares_outstanding=1e9,
            base_ebitda=2e9, exit_multiple=12.0, n_iterations=500)
        assert result is not None
        assert result['median_fv'] > 0

    def test_wider_sigma_gives_wider_range(self):
        """Higher sigma → wider p10-p90 spread."""
        r_narrow = monte_carlo_dcf(
            1e9, 0.10, 0.09, 0.03, 5e9, 1e9,
            growth_sigma=0.01, wacc_sigma=0.005, n_iterations=500)
        r_wide = monte_carlo_dcf(
            1e9, 0.10, 0.09, 0.03, 5e9, 1e9,
            growth_sigma=0.05, wacc_sigma=0.03, n_iterations=500)
        narrow_spread = r_narrow['p90_fv'] - r_narrow['p10_fv']
        wide_spread = r_wide['p90_fv'] - r_wide['p10_fv']
        assert wide_spread > narrow_spread


# ---------------------------------------------------------------------------
# reverse_dcf
# ---------------------------------------------------------------------------

class TestReverseDCF:
    def test_basic_convergence(self):
        """Known DCF -> reverse should recover approximately the growth rate."""
        # Forward: compute fair value at 10% growth
        ev = two_stage_ev(1e9, 0.10, 0.09, 0.03)
        fv = fair_value_per_share(ev, 5e9, 1e9)
        # Reverse: solve for implied growth given that fair value as price
        result = reverse_dcf(fv, 1e9, 0.09, 1e9, net_debt=5e9)
        assert result is not None
        assert result['converged'] is True
        assert result['implied_growth'] == pytest.approx(0.10, abs=0.005)

    def test_converged_flag(self):
        """Standard inputs should converge."""
        result = reverse_dcf(150.0, 1e9, 0.09, 1e9, net_debt=5e9)
        assert result is not None
        assert 'converged' in result
        assert 'implied_growth' in result

    def test_none_on_invalid_inputs(self):
        """Invalid inputs -> None."""
        assert reverse_dcf(0, 1e9, 0.09, 1e9) is None
        assert reverse_dcf(-10, 1e9, 0.09, 1e9) is None
        assert reverse_dcf(100, 0, 0.09, 1e9) is None
        assert reverse_dcf(100, -1e9, 0.09, 1e9) is None
        assert reverse_dcf(100, 1e9, 0, 1e9) is None
        assert reverse_dcf(100, 1e9, 0.09, 0) is None

    def test_none_on_zero_price(self):
        """Zero price -> None."""
        assert reverse_dcf(0, 1e9, 0.09, 1e9) is None

    def test_implied_growth_within_range(self):
        """Implied growth should be within reasonable bounds."""
        result = reverse_dcf(100.0, 1e9, 0.09, 1e9, net_debt=5e9)
        assert result is not None
        if result['converged']:
            assert 0 <= result['implied_growth'] <= 0.30

    def test_high_price_implies_high_growth(self):
        """Higher price -> higher implied growth rate."""
        r_low = reverse_dcf(50.0, 1e9, 0.09, 1e9, net_debt=5e9)
        r_high = reverse_dcf(200.0, 1e9, 0.09, 1e9, net_debt=5e9)
        assert r_low is not None and r_high is not None
        assert r_high['implied_growth'] > r_low['implied_growth']
