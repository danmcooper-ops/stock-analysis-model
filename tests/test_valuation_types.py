"""Tests for models/valuation_types.py — the validation substrate every
valuation model calls before dividing or discounting. It had no direct
coverage despite being the layer that turns bad inputs into loud errors.
"""

import numpy as np
import pytest


from models.valuation_types import Valuation, _validate_numeric, _validate_returns
from models.dcf import _validate_years


class TestValidateNumeric:
    def test_returns_float(self):
        assert _validate_numeric('x', 3) == 3.0
        assert isinstance(_validate_numeric('x', 3), float)
        assert _validate_numeric('x', '2.5') == 2.5

    def test_none_rejected(self):
        with pytest.raises(ValueError, match='x is None'):
            _validate_numeric('x', None)

    def test_non_numeric_rejected_with_name(self):
        with pytest.raises(ValueError, match="wacc is not numeric"):
            _validate_numeric('wacc', 'abc')
        with pytest.raises(ValueError, match='not numeric'):
            _validate_numeric('x', {'a': 1})

    def test_nan_and_inf_rejected(self):
        with pytest.raises(ValueError, match='not finite'):
            _validate_numeric('x', float('nan'))
        with pytest.raises(ValueError, match='not finite'):
            _validate_numeric('x', float('inf'))
        with pytest.raises(ValueError, match='not finite'):
            _validate_numeric('x', -np.inf)

    def test_finite_false_allows_inf(self):
        assert _validate_numeric('x', np.inf, finite=False) == np.inf

    def test_positive(self):
        assert _validate_numeric('x', 0.01, positive=True) == 0.01
        with pytest.raises(ValueError, match='must be > 0'):
            _validate_numeric('x', 0, positive=True)
        with pytest.raises(ValueError, match='must be > 0'):
            _validate_numeric('x', -1, positive=True)

    def test_allow_zero_false(self):
        with pytest.raises(ValueError, match='non-zero'):
            _validate_numeric('x', 0, allow_zero=False)
        assert _validate_numeric('x', -1, allow_zero=False) == -1.0

    def test_bounds_inclusive(self):
        assert _validate_numeric('x', 0.5, low=0.5, high=0.5) == 0.5
        with pytest.raises(ValueError, match='below'):
            _validate_numeric('x', 0.499, low=0.5)
        with pytest.raises(ValueError, match='above'):
            _validate_numeric('x', 0.501, high=0.5)

    def test_numpy_scalar_accepted(self):
        assert _validate_numeric('x', np.float64(1.5)) == 1.5


class TestValidateReturns:
    def test_clean_series_passes(self):
        arr = _validate_returns('r', [0.01] * 30)
        assert arr.shape == (30,)
        assert arr.dtype == float

    def test_min_obs_enforced(self):
        with pytest.raises(ValueError, match='need at least 24'):
            _validate_returns('r', [0.01] * 23)
        assert _validate_returns('r', [0.01] * 5, min_obs=5).size == 5

    def test_wrong_shape_rejected(self):
        with pytest.raises(ValueError, match='must be 1-D'):
            _validate_returns('r', [[0.01] * 30, [0.02] * 30])

    def test_non_finite_rejected(self):
        bad = [0.01] * 29 + [float('nan')]
        with pytest.raises(ValueError, match='non-finite'):
            _validate_returns('r', bad)


class TestValidateYears:
    def test_valid_combinations(self):
        _validate_years(10, 5)
        _validate_years(10, 10)   # no fade stage — legal
        _validate_years(10, 0)    # all fade — legal
        _validate_years(1, 1)

    def test_non_integer_rejected(self):
        with pytest.raises(ValueError, match='must be an integer'):
            _validate_years(10.0, 5)
        with pytest.raises(ValueError, match='must be an integer'):
            _validate_years(10, '5')
        with pytest.raises(ValueError, match='must be an integer'):
            _validate_years(True, 1)

    def test_numpy_integer_accepted(self):
        _validate_years(np.int64(10), np.int64(5))

    def test_bad_ranges_rejected(self):
        with pytest.raises(ValueError, match='total_years'):
            _validate_years(0, 0)
        with pytest.raises(ValueError, match='stage1_years'):
            _validate_years(10, 11)
        with pytest.raises(ValueError, match='stage1_years'):
            _validate_years(10, -1)


class TestValuationEnvelope:
    def test_truthiness_follows_value(self):
        assert Valuation(value=12.5, method='epv')
        assert not Valuation(value=None, method='epv')
        assert Valuation(value=0.0, method='epv')  # zero is a real number

    def test_float_conversion(self):
        assert float(Valuation(value=12.5, method='epv')) == 12.5
        assert np.isnan(float(Valuation(value=None, method='epv')))

    def test_invalid_constructor(self):
        v = Valuation.invalid('two_stage_ev', 'discount <= terminal growth',
                              inputs={'discount_rate': 0.02})
        assert v.value is None
        assert v.confidence == 0.0
        assert v.warnings == ('discount <= terminal growth',)
        assert v.inputs_used == {'discount_rate': 0.02}
        assert not v

    def test_frozen(self):
        import dataclasses
        v = Valuation(value=1.0, method='epv')
        with pytest.raises(dataclasses.FrozenInstanceError):
            v.value = 2.0

    def test_defaults(self):
        v = Valuation(value=1.0, method='epv')
        assert v.confidence == 1.0
        assert v.warnings == ()
        assert v.inputs_used == {}


class TestEnvelopePrimaries:
    """The *_valuation variants are the primary implementations; the legacy
    float|None functions must be exact .value views of them."""

    def test_two_stage_ev_envelope_matches_legacy(self):
        import warnings as w
        from models.dcf import two_stage_ev, two_stage_ev_valuation
        with w.catch_warnings():
            w.simplefilter('ignore', RuntimeWarning)
            v = two_stage_ev_valuation(100.0, 0.08, 0.10, 0.025)
            legacy = two_stage_ev(100.0, 0.08, 0.10, 0.025)
        assert v.value == legacy
        assert v.method == 'two_stage_ev_ggm'
        assert 0.4 <= v.confidence <= 1.0
        assert v.inputs_used['base_fcf'] == 100.0

    def test_two_stage_ev_invalid_carries_reason(self):
        import warnings as w
        from models.dcf import two_stage_ev_valuation
        with w.catch_warnings():
            w.simplefilter('ignore', RuntimeWarning)
            v = two_stage_ev_valuation(-5.0, 0.08, 0.10, 0.025)
        assert v.value is None
        assert v.confidence == 0.0
        assert any('invalid' in msg for msg in v.warnings)

    def test_soft_warnings_reduce_confidence(self):
        import warnings as w
        from models.dcf import two_stage_ev_valuation
        with w.catch_warnings():
            w.simplefilter('ignore', RuntimeWarning)
            calm = two_stage_ev_valuation(100.0, 0.05, 0.12, 0.02)
            aggressive = two_stage_ev_valuation(100.0, 0.40, 0.12, 0.02)
        assert aggressive.confidence < calm.confidence
        assert any('aggressive' in msg for msg in aggressive.warnings)

    def test_equity_models_envelopes_match_legacy(self):
        import warnings as w
        from models.ddm import (two_stage_ddm, two_stage_ddm_valuation,
                                ddm_h_model, ddm_h_model_valuation)
        from models.epv import earnings_power_value, earnings_power_value_valuation
        from models.rim import residual_income_model, residual_income_model_valuation
        with w.catch_warnings():
            w.simplefilter('ignore', RuntimeWarning)
            assert (two_stage_ddm_valuation(2.0, 0.06, 0.025, 0.09).value
                    == two_stage_ddm(2.0, 0.06, 0.025, 0.09))
            assert (ddm_h_model_valuation(2.0, 0.08, 0.03, 0.09).value
                    == ddm_h_model(2.0, 0.08, 0.03, 0.09))
            assert (earnings_power_value_valuation(500.0, 0.21, 0.09, 100.0).value
                    == earnings_power_value(500.0, 0.21, 0.09, 100.0))
            assert (residual_income_model_valuation(50.0, 0.15, 0.09).value
                    == residual_income_model(50.0, 0.15, 0.09))

    def test_rim_envelope_records_inferred_retention(self):
        import warnings as w
        from models.rim import residual_income_model_valuation
        with w.catch_warnings():
            w.simplefilter('ignore', RuntimeWarning)
            v = residual_income_model_valuation(50.0, 0.15, 0.09)
        assert 'retention_ratio' in v.inputs_used
        assert any('retention_ratio not provided' in msg for msg in v.warnings)
        assert v.confidence < 1.0
