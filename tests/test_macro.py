"""Tests for models/macro.py — regime scoring and parameter adjustments."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.macro import (
    _score_vix, _score_yield_curve, _score_credit_spread,
    _score_spy_momentum, _score_industrial_rs,
    assess_macro_regime, compute_macro_adjustments,
)


# ---------------------------------------------------------------------------
# Individual indicator scoring
# ---------------------------------------------------------------------------

class TestScoreVix:
    def test_low_vix_bullish(self):
        assert _score_vix(12) == 1.0

    def test_moderate_vix_neutral(self):
        assert _score_vix(22) == 0.0

    def test_high_vix_bearish(self):
        assert _score_vix(35) == -1.0

    def test_none_returns_zero(self):
        assert _score_vix(None) == 0.0


class TestScoreYieldCurve:
    def test_steep_positive(self):
        assert _score_yield_curve(0.02) == 1.0

    def test_flat(self):
        assert _score_yield_curve(0.0) == 0.0

    def test_inverted(self):
        assert _score_yield_curve(-0.02) == -1.0

    def test_none_returns_zero(self):
        assert _score_yield_curve(None) == 0.0


class TestScoreCreditSpread:
    def test_risk_on(self):
        """HYG outperforms (negative spread) → bullish."""
        assert _score_credit_spread(-0.03) == 1.0

    def test_neutral(self):
        assert _score_credit_spread(0.0) == 0.0

    def test_stress(self):
        """HYG underperforms (positive spread) → bearish."""
        assert _score_credit_spread(0.03) == -1.0


class TestScoreSpyMomentum:
    def test_above_sma_bullish(self):
        assert _score_spy_momentum(1.08) == 1.0

    def test_at_sma_neutral(self):
        assert _score_spy_momentum(1.0) == 0.0

    def test_below_sma_bearish(self):
        assert _score_spy_momentum(0.92) == -1.0


class TestScoreIndustrialRS:
    def test_outperformance(self):
        assert _score_industrial_rs(0.05) == 1.0

    def test_neutral(self):
        assert _score_industrial_rs(0.0) == 0.0

    def test_underperformance(self):
        assert _score_industrial_rs(-0.05) == -1.0


# ---------------------------------------------------------------------------
# Regime assessment
# ---------------------------------------------------------------------------

class TestAssessMacroRegime:
    def test_all_bullish_expansion(self):
        indicators = {
            'vix': 12, 'yield_curve_slope': 0.02,
            'credit_spread_3m': -0.03, 'spy_sma200_ratio': 1.08,
            'xli_rel_strength_3m': 0.05,
        }
        result = assess_macro_regime(indicators)
        assert result['regime'] == 'expansion'
        assert result['composite_score'] > 0.4

    def test_all_bearish_contraction(self):
        indicators = {
            'vix': 35, 'yield_curve_slope': -0.02,
            'credit_spread_3m': 0.03, 'spy_sma200_ratio': 0.92,
            'xli_rel_strength_3m': -0.05,
        }
        result = assess_macro_regime(indicators)
        assert result['regime'] == 'contraction'
        assert result['composite_score'] < -0.4

    def test_all_none_neutral(self):
        indicators = {
            'vix': None, 'yield_curve_slope': None,
            'credit_spread_3m': None, 'spy_sma200_ratio': None,
            'xli_rel_strength_3m': None,
        }
        result = assess_macro_regime(indicators)
        assert result['regime'] == 'neutral'
        assert result['composite_score'] == 0.0

    def test_mixed_mid_cycle(self):
        indicators = {
            'vix': 18, 'yield_curve_slope': 0.01,
            'credit_spread_3m': 0.0, 'spy_sma200_ratio': 1.03,
            'xli_rel_strength_3m': -0.02,
        }
        result = assess_macro_regime(indicators)
        assert result['regime'] in ('mid_cycle', 'neutral')

    def test_returns_raw_indicators(self):
        indicators = {'vix': 20, 'yield_curve_slope': 0.01,
                       'credit_spread_3m': 0.0, 'spy_sma200_ratio': 1.0,
                       'xli_rel_strength_3m': 0.0}
        result = assess_macro_regime(indicators)
        assert result['raw_indicators']['vix'] == 20
        assert 'indicator_scores' in result


# ---------------------------------------------------------------------------
# Parameter adjustments
# ---------------------------------------------------------------------------

class TestComputeAdjustments:
    def _regime(self, score):
        return {'composite_score': score, 'regime': 'test',
                'indicator_scores': {}, 'raw_indicators': {}}

    def test_neutral_zero_adjustments(self):
        adj = compute_macro_adjustments(self._regime(0.0))
        assert adj['erp_adjustment'] == 0.0
        assert adj['terminal_growth_adjustment'] == 0.0
        assert adj['wacc_sigma_adjustment'] == 0.0
        assert adj['growth_sigma_multiplier'] == 1.0
        assert adj['exit_mult_adjustment'] == 0.0
        assert adj['growth_weight_shift'] == 0.0

    def test_contraction_raises_erp(self):
        adj = compute_macro_adjustments(self._regime(-1.0))
        assert adj['erp_adjustment'] == pytest.approx(0.015)

    def test_expansion_reduces_erp_dampened(self):
        """Expansion ERP reduction is dampened (20% of full effect)."""
        adj = compute_macro_adjustments(self._regime(1.0))
        assert adj['erp_adjustment'] == pytest.approx(-0.003)

    def test_contraction_reduces_terminal_growth(self):
        adj = compute_macro_adjustments(self._regime(-1.0))
        assert adj['terminal_growth_adjustment'] == pytest.approx(-0.005)

    def test_contraction_widens_wacc_sigma(self):
        adj = compute_macro_adjustments(self._regime(-1.0))
        assert adj['wacc_sigma_adjustment'] == pytest.approx(0.005)

    def test_expansion_does_not_narrow_sigma(self):
        """One-sided: WACC sigma never decreases below base."""
        adj = compute_macro_adjustments(self._regime(1.0))
        assert adj['wacc_sigma_adjustment'] == 0.0
        assert adj['growth_sigma_multiplier'] == 1.0

    def test_linear_scaling(self):
        """Half-stress → half the adjustments."""
        adj = compute_macro_adjustments(self._regime(-0.5))
        assert adj['erp_adjustment'] == pytest.approx(0.0075)
        assert adj['terminal_growth_adjustment'] == pytest.approx(-0.0025)

    def test_exit_mult_contraction(self):
        adj = compute_macro_adjustments(self._regime(-1.0))
        assert adj['exit_mult_adjustment'] == pytest.approx(-2.0)

    def test_exit_mult_expansion_dampened(self):
        """Expansion exit multiple increase is dampened (20% of full effect)."""
        adj = compute_macro_adjustments(self._regime(1.0))
        assert adj['exit_mult_adjustment'] == pytest.approx(0.4)

    def test_growth_weight_shift_contraction(self):
        """In contraction, shift weight from analyst LT to fundamental."""
        adj = compute_macro_adjustments(self._regime(-1.0))
        assert adj['growth_weight_shift'] == pytest.approx(-0.05)

    def test_growth_sigma_multiplier_contraction(self):
        adj = compute_macro_adjustments(self._regime(-1.0))
        assert adj['growth_sigma_multiplier'] == pytest.approx(1.3)
