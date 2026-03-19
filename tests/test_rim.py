# tests/test_rim.py
import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.rim import residual_income_model


class TestResidualIncomeModel:
    def test_basic_positive(self):
        """Standard inputs should produce positive value."""
        fv = residual_income_model(book_value_per_share=45.0, roe=0.15,
                                    cost_of_equity=0.10)
        assert fv is not None
        assert fv > 0

    def test_none_on_none_bv(self):
        """None book value -> None."""
        assert residual_income_model(None, 0.15, 0.10) is None

    def test_none_on_zero_bv(self):
        """Zero book value -> None."""
        assert residual_income_model(0, 0.15, 0.10) is None

    def test_none_on_zero_re(self):
        """Zero cost of equity -> None."""
        assert residual_income_model(45.0, 0.15, 0) is None

    def test_higher_roe_gives_higher_value(self):
        """Higher ROE -> higher intrinsic value."""
        fv_low = residual_income_model(45.0, roe=0.08, cost_of_equity=0.10)
        fv_high = residual_income_model(45.0, roe=0.20, cost_of_equity=0.10)
        assert fv_high > fv_low

    def test_roe_equals_re_gives_near_book_value(self):
        """When ROE = Re, residual income is zero, value ~ book value."""
        fv = residual_income_model(45.0, roe=0.10, cost_of_equity=0.10)
        # Should be approximately book value (RI is zero)
        assert fv is not None
        assert abs(fv - 45.0) < 5.0  # approximately BV

    def test_roe_below_re_gives_below_book(self):
        """ROE < Re -> negative residual income -> value below book."""
        fv = residual_income_model(45.0, roe=0.05, cost_of_equity=0.10)
        assert fv is not None
        assert fv < 45.0

    def test_none_when_re_below_g(self):
        """Re <= g -> terminal value undefined -> None."""
        assert residual_income_model(45.0, 0.15, 0.02, g=0.03) is None

    def test_value_above_book_when_roe_exceeds_re(self):
        """ROE > Re -> value should be above book value."""
        fv = residual_income_model(45.0, roe=0.18, cost_of_equity=0.10)
        assert fv > 45.0
