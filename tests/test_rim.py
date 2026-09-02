# tests/test_rim.py
import warnings

import pytest


from models.rim import (residual_income_model, residual_income_model_valuation,
                        retention_from_shareholder_returns)


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
        fv = residual_income_model(100.0, 0.15, 0.10, g=0.03, years=1,
                                   retention_ratio=0.2, spread_persistence=1.0)
        assert fv == pytest.approx(171.4286, abs=1e-3)

    def test_below_cost_of_equity_can_destroy_value(self):
        """ROE < Re: negative terminal RI is kept (not zeroed), so a chronic
        under-earner is valued below book."""
        fv = residual_income_model(100.0, 0.06, 0.10, g=0.03, years=10,
                                   retention_ratio=0.5)
        assert fv is None or fv < 100.0


class TestRimRoeFade:
    """The ROE−Re spread fades linearly toward Re over the explicit horizon,
    keeping `spread_persistence` of it by year N and into the terminal."""

    def test_known_value_two_years_half_persistence(self):
        # BV0=100, ROE=15%, Re=10%, g=3%, retention=0.2, N=2, p=0.5.
        # spread=5%. ROE_1 = 10% + 5%·(1 − 0.5·1/2) = 13.75%
        #   RI_1 = 100·3.75% = 3.75; PV = 3.4091; BV_1 = 100·(1+.1375·.2) = 102.75
        # ROE_2 = 10% + 5%·(1 − 0.5) = 12.5%
        #   RI_2 = 102.75·2.5% = 2.56875; PV = 2.1229; BV_2 = 105.31875
        # RI_3 = 105.31875·2.5% = 2.63297; TV_2 = /0.07 = 37.6138; PV = 31.0858
        # intrinsic = 100 + 3.4091 + 2.1229 + 31.0858 = 136.6178
        fv = residual_income_model(100.0, 0.15, 0.10, g=0.03, years=2,
                                   retention_ratio=0.2, spread_persistence=0.5)
        assert fv == pytest.approx(136.6178, abs=1e-3)

    def test_full_persistence_matches_no_fade(self):
        """p=1.0 reproduces the flat-ROE result exactly."""
        fv = residual_income_model(100.0, 0.15, 0.10, g=0.03, years=1,
                                   retention_ratio=0.2, spread_persistence=1.0)
        assert fv == pytest.approx(171.4286, abs=1e-3)

    def test_zero_persistence_has_no_terminal_value(self):
        """p=0 fades ROE to Re by year N, so RI_{N+1} = 0 and value is
        book plus the PV of the fading explicit-period RI only."""
        v = residual_income_model_valuation(100.0, 0.15, 0.10, g=0.03, years=1,
                                            retention_ratio=0.2,
                                            spread_persistence=0.0)
        # ROE_1 = Re → RI_1 = 0 → value == book exactly.
        assert v.value == pytest.approx(100.0, abs=1e-9)

    def test_more_persistence_more_value_when_roe_above_re(self):
        low = residual_income_model(50.0, 0.20, 0.10, spread_persistence=0.25)
        mid = residual_income_model(50.0, 0.20, 0.10, spread_persistence=0.50)
        high = residual_income_model(50.0, 0.20, 0.10, spread_persistence=1.0)
        assert low < mid < high

    def test_under_earner_fades_upward(self):
        """ROE < Re mean-reverts too: fading shrinks the value destruction,
        but a chronic under-earner still sits below book."""
        faded = residual_income_model(100.0, 0.06, 0.10, retention_ratio=0.5,
                                      spread_persistence=0.5)
        flat = residual_income_model(100.0, 0.06, 0.10, retention_ratio=0.5,
                                     spread_persistence=1.0)
        assert faded is not None and flat is not None
        assert flat < faded < 100.0

    def test_default_fade_tames_high_roe_high_retention(self):
        """The failure mode that motivated the change: a 150% ROE with 85%
        retention (dividend-only payout on a buyback-heavy firm) compounded
        book past $100k/share. With the default fade and growth cap the
        value stays within an order of magnitude of book."""
        v = residual_income_model_valuation(4.0, 1.50, 0.09, g=0.03,
                                            retention_ratio=0.85)
        assert v.value is not None
        assert v.value < 4.0 * 100


class TestRimBookGrowthCap:
    def test_cap_applies_and_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            capped = residual_income_model_valuation(
                10.0, 1.0, 0.10, retention_ratio=1.0, spread_persistence=1.0,
                max_book_growth=0.25)
            uncapped = residual_income_model_valuation(
                10.0, 1.0, 0.10, retention_ratio=1.0, spread_persistence=1.0,
                max_book_growth=10.0)
        assert capped.value < uncapped.value
        assert any('capped' in msg for msg in capped.warnings)
        assert not any('capped' in msg for msg in uncapped.warnings)
        assert any('capped' in str(x.message) for x in w)
        assert capped.inputs_used['max_book_growth'] == 0.25

    def test_cap_exact_value_one_year(self):
        # BV0=10, ROE=100%, Re=10%, retention=1, p=1, cap 25%, N=1, g=3%.
        # RI_1 = 10·0.9 = 9; PV = 8.1818; BV_1 = 10·1.25 = 12.5 (not 20)
        # RI_2 = 12.5·0.9 = 11.25; TV = /0.07 = 160.714; PV = 146.104
        fv = residual_income_model(10.0, 1.0, 0.10, g=0.03, years=1,
                                   retention_ratio=1.0, spread_persistence=1.0,
                                   max_book_growth=0.25)
        assert fv == pytest.approx(10 + 8.1818 + 146.1039, abs=1e-3)

    def test_no_warning_when_growth_below_cap(self):
        v = residual_income_model_valuation(45.0, 0.15, 0.10, retention_ratio=0.5)
        assert not any('capped' in msg for msg in v.warnings)


class TestRetentionFromShareholderReturns:
    def test_dividends_plus_buybacks(self):
        # NI 100, dividends 15 + buybacks 85 → 0% retained.
        assert retention_from_shareholder_returns(100.0, 100.0) == 0.0
        assert retention_from_shareholder_returns(100.0, 40.0) == pytest.approx(0.6)

    def test_clamped(self):
        # Returning more than earned (debt-funded buybacks) → 0, not negative.
        assert retention_from_shareholder_returns(100.0, 130.0) == 0.0
        # Net issuance (negative total return) → 1, not > 1.
        assert retention_from_shareholder_returns(100.0, -20.0) == 1.0

    def test_none_when_undefined(self):
        assert retention_from_shareholder_returns(None, 10.0) is None
        assert retention_from_shareholder_returns(100.0, None) is None
        assert retention_from_shareholder_returns(0.0, 10.0) is None
        assert retention_from_shareholder_returns(-50.0, 10.0) is None
        assert retention_from_shareholder_returns(float('nan'), 10.0) is None
        assert retention_from_shareholder_returns('x', 10.0) is None
