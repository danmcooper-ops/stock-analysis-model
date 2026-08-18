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


class TestRimTerminalValue:
    """Pin the Gordon terminal timing: TV_N = RI_{N+1}/(Re−g), no extra (1+g)."""

    def test_known_value_one_year(self):
        # BV0=100, ROE=15%, Re=10%, g=3%, retention=0.2 (so book grows 3%),
        # 1 explicit year.
        # Year 1: RI_1 = 100·(0.15−0.10) = 5; PV = 5/1.10 = 4.5454...
        # BV_1 = 100·(1 + 0.15·0.2) = 103
        # RI_terminal (=RI_2) = 103·0.05 = 5.15
        # TV_1 = 5.15/(0.10−0.03) = 73.5714...; PV = /1.10 = 66.8831...
        # intrinsic = 100 + 4.5454 + 66.8831 = 171.4286
        import pytest
        fv = residual_income_model(100.0, 0.15, 0.10, g=0.03, years=1,
                                   retention_ratio=0.2)
        assert fv == pytest.approx(171.4286, abs=1e-3)

    def test_below_cost_of_equity_can_destroy_value(self):
        """ROE < Re: negative terminal RI is kept (not zeroed), so a chronic
        under-earner is valued below book."""
        fv = residual_income_model(100.0, 0.06, 0.10, g=0.03, years=10,
                                   retention_ratio=0.5)
        assert fv is None or fv < 100.0
