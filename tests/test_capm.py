# tests/test_capm.py
import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.capm import (
    calculate_beta,
    r2_diagnostic, expected_return, ggm_implied_re, buildup_re,
)


# ---------------------------------------------------------------------------
# calculate_beta
# ---------------------------------------------------------------------------

class TestCalculateBeta:
    def test_known_beta(self, synthetic_returns):
        """With stock = 1.3 * market + noise, raw beta should be near 1.3."""
        stock, market = synthetic_returns
        result = calculate_beta(stock, market)
        assert result['raw_beta'] == pytest.approx(1.3, abs=0.15)

    def test_adjusted_beta(self, synthetic_returns):
        """Adjusted beta = (2/3)*raw + (1/3)*1.0."""
        stock, market = synthetic_returns
        result = calculate_beta(stock, market, adjust=True)
        expected_adj = (2 / 3) * result['raw_beta'] + (1 / 3) * 1.0
        assert result['adjusted_beta'] == pytest.approx(expected_adj)

    def test_no_adjustment(self, synthetic_returns):
        """With adjust=False, adjusted_beta == raw_beta."""
        stock, market = synthetic_returns
        result = calculate_beta(stock, market, adjust=False)
        assert result['adjusted_beta'] == pytest.approx(result['raw_beta'])

    def test_r_squared_range(self, synthetic_returns):
        """R² should be between 0 and 1."""
        stock, market = synthetic_returns
        result = calculate_beta(stock, market)
        assert 0 <= result['r_squared'] <= 1

    def test_r_squared_high_signal(self, synthetic_returns):
        """With strong linear relationship, R² should be reasonably high."""
        stock, market = synthetic_returns
        result = calculate_beta(stock, market)
        assert result['r_squared'] > 0.5

    def test_se_beta_positive(self, synthetic_returns):
        """Standard error of beta should be positive."""
        stock, market = synthetic_returns
        result = calculate_beta(stock, market)
        assert result['se_beta'] > 0

    def test_n_observations(self, synthetic_returns):
        """n_observations should match input length."""
        stock, market = synthetic_returns
        result = calculate_beta(stock, market)
        assert result['n_observations'] == len(stock)

    def test_beta_of_market_is_one(self):
        """The market regressed on itself should have beta=1."""
        np.random.seed(99)
        market = np.random.normal(0.0004, 0.01, 300)
        result = calculate_beta(market, market, adjust=False)
        assert result['raw_beta'] == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# r2_diagnostic
# ---------------------------------------------------------------------------

class TestR2Diagnostic:
    def test_reliable(self):
        classification, method = r2_diagnostic(0.70)
        assert classification == 'reliable'
        assert method == 'capm'

    def test_reliable_boundary(self):
        classification, method = r2_diagnostic(0.60)
        assert classification == 'reliable'
        assert method == 'capm'

    def test_directional(self):
        classification, method = r2_diagnostic(0.50)
        assert classification == 'directional'
        assert method == 'capm_plus_alternative'

    def test_directional_boundary(self):
        classification, method = r2_diagnostic(0.40)
        assert classification == 'directional'
        assert method == 'capm_plus_alternative'

    def test_unreliable(self):
        classification, method = r2_diagnostic(0.39)
        assert classification == 'unreliable'
        assert method == 'fundamental_only'

    def test_unreliable_zero(self):
        classification, method = r2_diagnostic(0.0)
        assert classification == 'unreliable'
        assert method == 'fundamental_only'


# ---------------------------------------------------------------------------
# expected_return
# ---------------------------------------------------------------------------

class TestExpectedReturn:
    def test_standard_capm(self):
        """Re = Rf + β(Rm - Rf) = 0.04 + 1.2*(0.10 - 0.04) = 0.112."""
        re = expected_return(0.04, 1.2, 0.10)
        assert re == pytest.approx(0.112)

    def test_beta_one(self):
        """With β=1, Re should equal Rm."""
        re = expected_return(0.04, 1.0, 0.10)
        assert re == pytest.approx(0.10)

    def test_beta_zero(self):
        """With β=0, Re should equal Rf."""
        re = expected_return(0.04, 0.0, 0.10)
        assert re == pytest.approx(0.04)


# ---------------------------------------------------------------------------
# ggm_implied_re
# ---------------------------------------------------------------------------

class TestGGMImpliedRe:
    def test_basic(self):
        """Re = div_yield*(1+g) + g."""
        re = ggm_implied_re(0.03, 0.03)
        expected = 0.03 * 1.03 + 0.03  # 0.0609
        assert re == pytest.approx(expected)

    def test_zero_yield_returns_none(self):
        assert ggm_implied_re(0, 0.03) is None

    def test_negative_yield_returns_none(self):
        assert ggm_implied_re(-0.01, 0.03) is None

    def test_none_yield_returns_none(self):
        assert ggm_implied_re(None, 0.03) is None


# ---------------------------------------------------------------------------
# buildup_re
# ---------------------------------------------------------------------------

class TestBuildupRe:
    def test_with_defaults(self):
        """Default: Rf + 0.055 + 0.02 + 0.01."""
        re = buildup_re(0.04)
        assert re == pytest.approx(0.04 + 0.055 + 0.02 + 0.01)

    def test_zero_premiums(self):
        """With zero premiums, Re = Rf + ERP = 0.04 + 0.055 = 0.095."""
        re = buildup_re(0.04, erp=0.055, size_premium=0, industry_premium=0)
        assert re == pytest.approx(0.095)

    def test_custom_premiums(self):
        re = buildup_re(0.04, erp=0.06, size_premium=0.03, industry_premium=0.02)
        assert re == pytest.approx(0.15)
