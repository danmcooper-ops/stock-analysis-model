# tests/test_ddm.py
import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.ddm import (
    two_stage_ddm, ddm_h_model, ddm_eligibility,
    estimate_ddm_growth, monte_carlo_ddm,
)


# ---------------------------------------------------------------------------
# two_stage_ddm
# ---------------------------------------------------------------------------

class TestTwoStageDDM:
    def test_basic_positive(self):
        """Standard inputs should produce a positive value."""
        fv = two_stage_ddm(dps=2.0, high_g=0.07, term_g=0.03, re=0.10)
        assert fv is not None
        assert fv > 0

    def test_higher_growth_gives_higher_value(self):
        """Higher high-growth rate → higher intrinsic value."""
        fv_low = two_stage_ddm(2.0, 0.03, 0.03, 0.10)
        fv_high = two_stage_ddm(2.0, 0.12, 0.03, 0.10)
        assert fv_high > fv_low

    def test_higher_discount_gives_lower_value(self):
        """Higher required return → lower intrinsic value."""
        fv_low_re = two_stage_ddm(2.0, 0.07, 0.03, 0.08)
        fv_high_re = two_stage_ddm(2.0, 0.07, 0.03, 0.14)
        assert fv_low_re > fv_high_re

    def test_none_on_zero_dps(self):
        """Zero DPS → None."""
        assert two_stage_ddm(0, 0.07, 0.03, 0.10) is None

    def test_none_on_negative_dps(self):
        """Negative DPS → None."""
        assert two_stage_ddm(-1.0, 0.07, 0.03, 0.10) is None

    def test_none_when_re_below_tg(self):
        """re ≤ term_g → None (Gordon Growth undefined)."""
        assert two_stage_ddm(2.0, 0.07, 0.10, 0.08) is None

    def test_min_spread_enforced(self):
        """When re - tg < min_spread, effective_tg adjusts and result is finite."""
        fv = two_stage_ddm(dps=2.0, high_g=0.05, term_g=0.09, re=0.10)
        assert fv is not None
        assert np.isfinite(fv)

    def test_custom_years(self):
        """Non-default year count should work."""
        fv = two_stage_ddm(2.0, 0.07, 0.03, 0.10, years=10)
        assert fv is not None
        assert fv > 0


# ---------------------------------------------------------------------------
# ddm_h_model
# ---------------------------------------------------------------------------

class TestHModel:
    def test_basic_positive(self):
        """Standard inputs should produce a positive value."""
        fv = ddm_h_model(dps=2.0, short_g=0.10, long_g=0.03, re=0.10)
        assert fv is not None
        assert fv > 0

    def test_higher_short_g_gives_higher_value(self):
        """Higher short-term growth → higher value."""
        fv_low = ddm_h_model(2.0, 0.05, 0.03, 0.10)
        fv_high = ddm_h_model(2.0, 0.15, 0.03, 0.10)
        assert fv_high > fv_low

    def test_none_on_zero_dps(self):
        """Zero DPS → None."""
        assert ddm_h_model(0, 0.10, 0.03, 0.10) is None

    def test_none_when_re_below_long_g(self):
        """re ≤ long_g → None."""
        assert ddm_h_model(2.0, 0.10, 0.10, 0.08) is None

    def test_longer_half_life_gives_higher_value(self):
        """Longer half-life → more growth premium → higher value."""
        fv_short = ddm_h_model(2.0, 0.12, 0.03, 0.10, half_life=3)
        fv_long = ddm_h_model(2.0, 0.12, 0.03, 0.10, half_life=10)
        assert fv_long > fv_short

    def test_comparable_to_two_stage(self):
        """H-model should be in same order of magnitude as two-stage DDM."""
        fv_ts = two_stage_ddm(2.0, 0.08, 0.03, 0.10, years=5)
        fv_h = ddm_h_model(2.0, 0.08, 0.03, 0.10, half_life=5)
        assert fv_ts is not None and fv_h is not None
        ratio = fv_h / fv_ts
        assert 0.3 < ratio < 3.0


# ---------------------------------------------------------------------------
# ddm_eligibility
# ---------------------------------------------------------------------------

class TestDDMEligibility:
    def test_eligible_basic(self, sample_dividend_history):
        """Stock with 5-year history, normal payout, positive EPS is eligible."""
        result = ddm_eligibility(sample_dividend_history, 0.50, 4.0, 1.76)
        assert result['eligible'] is True
        assert result['consecutive_years'] == 5

    def test_not_eligible_no_dividend(self):
        """Zero DPS → not eligible."""
        result = ddm_eligibility([1.0, 1.1, 1.2], 0.30, 4.0, 0)
        assert result['eligible'] is False
        assert 'No current dividend' in result['reason']

    def test_not_eligible_negative_eps(self):
        """Negative EPS → not eligible."""
        result = ddm_eligibility([1.0, 1.1, 1.2], 0.30, -2.0, 1.2)
        assert result['eligible'] is False
        assert 'Non-positive EPS' in result['reason']

    def test_not_eligible_short_history(self):
        """Only 2 years → not eligible (need 3)."""
        result = ddm_eligibility([1.0, 1.1], 0.30, 4.0, 1.1)
        assert result['eligible'] is False
        assert '2 consecutive' in result['reason']

    def test_payout_flag_over_100(self, sample_dividend_history):
        """Payout > 100% → eligible but payout_flag = True."""
        result = ddm_eligibility(sample_dividend_history, 1.20, 4.0, 1.76)
        assert result['eligible'] is True
        assert result['payout_flag'] is True

    def test_no_history(self):
        """None history → not eligible."""
        result = ddm_eligibility(None, 0.30, 4.0, 2.0)
        assert result['eligible'] is False

    def test_gap_in_history(self):
        """Gap in dividends breaks consecutive count."""
        history = [1.0, 1.1, 0, 1.2, 1.3]  # gap at position 2
        result = ddm_eligibility(history, 0.30, 4.0, 1.3)
        assert result['consecutive_years'] == 2
        assert result['eligible'] is False


# ---------------------------------------------------------------------------
# estimate_ddm_growth
# ---------------------------------------------------------------------------

class TestEstimateDDMGrowth:
    def test_all_three_signals(self, sample_growing_dividend_history):
        """All three signals available → weighted average."""
        result = estimate_ddm_growth(
            sample_growing_dividend_history, payout=0.40, roe=0.18, analyst_ltg=0.08)
        assert result['growth'] is not None
        assert result['signals_used'] == 3
        assert result['div_cagr'] is not None
        assert result['sustainable_growth'] is not None
        assert 0 < result['growth'] < 0.25

    def test_only_cagr_signal(self, sample_growing_dividend_history):
        """Only dividend history available → uses CAGR alone."""
        result = estimate_ddm_growth(
            sample_growing_dividend_history, payout=None, roe=None, analyst_ltg=None)
        assert result['signals_used'] == 1
        assert result['growth'] is not None
        assert result['growth'] == pytest.approx(result['div_cagr'])

    def test_no_signals(self):
        """No valid signals → growth is None."""
        result = estimate_ddm_growth(None, None, None, None)
        assert result['growth'] is None
        assert result['signals_used'] == 0

    def test_cagr_calculation(self):
        """CAGR of [1.0, 2.0] over 1 year = 100%."""
        result = estimate_ddm_growth([1.0, 2.0], None, None, None)
        assert result['div_cagr'] is not None
        # Capped at 25%
        assert result['div_cagr'] == pytest.approx(0.25)

    def test_sustainable_growth(self):
        """ROE=15%, payout=40% → sustainable=15%×60%=9%."""
        result = estimate_ddm_growth(None, payout=0.40, roe=0.15, analyst_ltg=None)
        assert result['sustainable_growth'] == pytest.approx(0.09)
        assert result['signals_used'] == 1

    def test_negative_roe_excluded(self):
        """Negative ROE → sustainable growth not used."""
        result = estimate_ddm_growth([1.0, 1.05], payout=0.40, roe=-0.10, analyst_ltg=None)
        assert result['sustainable_growth'] is None
        assert result['signals_used'] == 1  # only CAGR

    def test_growth_bounded(self, sample_growing_dividend_history):
        """Growth estimate should be capped at 25%."""
        result = estimate_ddm_growth(
            sample_growing_dividend_history, payout=0.10, roe=0.50, analyst_ltg=0.30)
        assert result['growth'] is not None
        assert result['growth'] <= 0.25


# ---------------------------------------------------------------------------
# monte_carlo_ddm
# ---------------------------------------------------------------------------

class TestMonteCarloDDM:
    def test_basic_returns_dict(self):
        """Standard inputs should return a dict with expected keys."""
        result = monte_carlo_ddm(dps=2.0, g=0.07, re=0.10, tg=0.03, n=500)
        assert result is not None
        assert isinstance(result, dict)
        for key in ('median_fv', 'mean_fv', 'p10_fv', 'p90_fv',
                     'std_fv', 'cv', 'n_valid'):
            assert key in result

    def test_p10_below_median_below_p90(self):
        """p10 ≤ median ≤ p90."""
        result = monte_carlo_ddm(2.0, 0.07, 0.10, 0.03, n=500)
        assert result['p10_fv'] <= result['median_fv'] <= result['p90_fv']

    def test_positive_fair_values(self):
        """All percentiles should be positive."""
        result = monte_carlo_ddm(2.0, 0.07, 0.10, 0.03, n=500)
        assert result['p10_fv'] > 0
        assert result['median_fv'] > 0
        assert result['p90_fv'] > 0

    def test_reproducible(self):
        """Fixed seed → deterministic results."""
        r1 = monte_carlo_ddm(2.0, 0.07, 0.10, 0.03, n=200)
        r2 = monte_carlo_ddm(2.0, 0.07, 0.10, 0.03, n=200)
        assert r1['median_fv'] == pytest.approx(r2['median_fv'])

    def test_none_on_zero_dps(self):
        """Zero DPS → None."""
        assert monte_carlo_ddm(0, 0.07, 0.10, 0.03) is None

    def test_cv_positive(self):
        """Coefficient of variation should be positive."""
        result = monte_carlo_ddm(2.0, 0.07, 0.10, 0.03, n=500)
        assert result['cv'] > 0

    def test_wider_sigma_gives_wider_range(self):
        """Higher sigma → wider p10-p90 spread."""
        r_narrow = monte_carlo_ddm(2.0, 0.07, 0.10, 0.03, n=500,
                                   g_sigma=0.01, re_sigma=0.005)
        r_wide = monte_carlo_ddm(2.0, 0.07, 0.10, 0.03, n=500,
                                  g_sigma=0.05, re_sigma=0.03)
        narrow_spread = r_narrow['p90_fv'] - r_narrow['p10_fv']
        wide_spread = r_wide['p90_fv'] - r_wide['p10_fv']
        assert wide_spread > narrow_spread
