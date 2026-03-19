# tests/test_param_set.py
"""Tests for param_set module: creation, merging, and validation."""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.param_set import default_params, merge_params, validate_params


class TestDefaultParams:
    """Tests for default_params()."""

    def test_returns_dict(self):
        p = default_params()
        assert isinstance(p, dict)

    def test_all_expected_keys_present(self):
        p = default_params()
        required = [
            'erp', 'terminal_growth_rate', 'wacc_floor', 'wacc_cap',
            'score_weight_valuation', 'score_weight_quality',
            'score_weight_moat', 'score_weight_growth',
            'blend_dcf_weight', 'blend_mult_weight', 'blend_trigger',
            'growth_weight_analyst_lt', 'growth_weight_fundamental',
            'analyst_haircut', 'margin_trend_sensitivity',
            'dcf_years', 'dcf_stage1', 'mc_iterations',
            'ddm_blend_weight', 'dcf_blend_weight_with_ddm',
        ]
        for key in required:
            assert key in p, f"Missing expected key: {key}"

    def test_values_match_config(self):
        from scripts.config import ERP, TERMINAL_GROWTH_RATE, SCORE_WEIGHT_VALUATION
        p = default_params()
        assert p['erp'] == ERP
        assert p['terminal_growth_rate'] == TERMINAL_GROWTH_RATE
        assert p['score_weight_valuation'] == SCORE_WEIGHT_VALUATION

    def test_defaults_pass_validation(self):
        p = default_params()
        errors = validate_params(p)
        assert errors == [], f"Defaults should be valid, got: {errors}"


class TestMergeParams:
    """Tests for merge_params()."""

    def test_no_overrides_returns_defaults(self):
        p1 = default_params()
        p2 = merge_params()
        assert p1 == p2

    def test_override_single_key(self):
        p = merge_params({'erp': 0.06})
        assert p['erp'] == 0.06
        # Other keys unchanged
        assert p['terminal_growth_rate'] == default_params()['terminal_growth_rate']

    def test_override_multiple_keys(self):
        p = merge_params({
            'erp': 0.07,
            'score_weight_valuation': 0.40,
            'blend_trigger': 1.3,
        })
        assert p['erp'] == 0.07
        assert p['score_weight_valuation'] == 0.40
        assert p['blend_trigger'] == 1.3

    def test_rejects_unknown_key(self):
        with pytest.raises(ValueError, match="Unknown parameter"):
            merge_params({'not_a_real_param': 42})

    def test_empty_overrides(self):
        p = merge_params({})
        assert p == default_params()

    def test_none_overrides(self):
        p = merge_params(None)
        assert p == default_params()


class TestValidateParams:
    """Tests for validate_params()."""

    def test_valid_defaults(self):
        assert validate_params(default_params()) == []

    def test_bad_scoring_weight_sum(self):
        p = default_params()
        p['score_weight_valuation'] = 0.50  # Sum now 0.50+0.25+0.25+0.20=1.20
        errors = validate_params(p)
        assert any('Scoring weights' in e for e in errors)

    def test_scoring_weight_below_minimum(self):
        p = default_params()
        p['score_weight_growth'] = 0.01  # Below 0.05 floor
        errors = validate_params(p)
        assert any('below minimum' in e for e in errors)

    def test_bad_blend_weight_sum(self):
        p = default_params()
        p['blend_dcf_weight'] = 0.80
        # blend_mult_weight still 0.40, sum = 1.20
        errors = validate_params(p)
        assert any('Blend weights' in e for e in errors)

    def test_bad_ddm_blend_weight_sum(self):
        p = default_params()
        p['ddm_blend_weight'] = 0.50
        # dcf_blend_weight_with_ddm still 0.70, sum = 1.20
        errors = validate_params(p)
        assert any('DDM blend' in e for e in errors)

    def test_erp_too_low(self):
        p = default_params()
        p['erp'] = 0.01
        errors = validate_params(p)
        assert any('ERP' in e for e in errors)

    def test_erp_too_high(self):
        p = default_params()
        p['erp'] = 0.15
        errors = validate_params(p)
        assert any('ERP' in e for e in errors)

    def test_terminal_growth_too_high(self):
        p = default_params()
        p['terminal_growth_rate'] = 0.10
        errors = validate_params(p)
        assert any('Terminal growth' in e for e in errors)

    def test_dcf_stage1_exceeds_years(self):
        p = default_params()
        p['dcf_stage1'] = 15
        p['dcf_years'] = 10
        errors = validate_params(p)
        assert any('dcf_stage1' in e for e in errors)

    def test_mc_iterations_too_low(self):
        p = default_params()
        p['mc_iterations'] = 10
        errors = validate_params(p)
        assert any('mc_iterations' in e for e in errors)

    def test_valid_custom_params(self):
        """A valid non-default set should pass validation."""
        p = merge_params({
            'erp': 0.06,
            'score_weight_valuation': 0.35,
            'score_weight_quality': 0.25,
            'score_weight_moat': 0.20,
            'score_weight_growth': 0.20,
            'blend_dcf_weight': 0.55,
            'blend_mult_weight': 0.45,
        })
        assert validate_params(p) == []
