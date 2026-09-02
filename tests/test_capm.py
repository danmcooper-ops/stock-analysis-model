# tests/test_capm.py
import pytest
import numpy as np


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
        """Default: Rf + 0.045 + 0.02 + 0.01 (erp default mirrors config.ERP)."""
        from scripts.config import ERP
        re = buildup_re(0.04)
        assert re == pytest.approx(0.04 + ERP + 0.02 + 0.01)
        assert re == pytest.approx(0.04 + 0.045 + 0.02 + 0.01)

    def test_zero_premiums(self):
        """With zero premiums, Re = Rf + ERP = 0.04 + 0.045 = 0.085."""
        re = buildup_re(0.04, erp=0.045, size_premium=0, industry_premium=0)
        assert re == pytest.approx(0.085)

    def test_custom_premiums(self):
        re = buildup_re(0.04, erp=0.06, size_premium=0.03, industry_premium=0.02)
        assert re == pytest.approx(0.15)


class TestR2MethodSelection:
    """The R² tiers the pipeline now implements via select_cost_of_equity's
    build_re blend (unit-tested at the r2_diagnostic level)."""

    def test_r2_diagnostic_tiers(self):
        from models.capm import r2_diagnostic
        assert r2_diagnostic(0.70)[0] == 'reliable'
        assert r2_diagnostic(0.60)[0] == 'reliable'
        assert r2_diagnostic(0.50)[0] == 'directional'
        assert r2_diagnostic(0.40)[0] == 'directional'
        assert r2_diagnostic(0.39)[0] == 'unreliable'
        assert r2_diagnostic(0.0)[0] == 'unreliable'


class TestTzNormalizeJoin:
    def test_mixed_tz_series_join_after_normalize(self):
        import pandas as pd
        from scripts.analyze_stock import _to_tznaive
        aware = pd.Series([1.0, 2.0],
                          index=pd.to_datetime(['2024-01-01', '2024-01-02'], utc=True))
        naive = pd.Series([3.0, 4.0],
                          index=pd.to_datetime(['2024-01-01', '2024-01-02']))
        # Before normalizing, the join would misalign / raise; after, it aligns.
        combined = pd.DataFrame({'a': _to_tznaive(aware),
                                 'b': _to_tznaive(naive)}).dropna()
        assert len(combined) == 2

    def test_beta_rejects_flat_stock_series(self):
        import numpy as np
        import pytest
        from models.capm import calculate_beta
        with pytest.raises(ValueError):
            calculate_beta(np.zeros(50), np.random.RandomState(0).normal(0, 0.01, 50))


# ---------------------------------------------------------------------------
# shrink_beta (precision-weighted / Vasicek)
# ---------------------------------------------------------------------------

class TestShrinkBeta:
    def test_zero_se_keeps_raw(self):
        from models.capm import shrink_beta
        beta, w = shrink_beta(1.6, 0.0)
        assert w == pytest.approx(1.0)
        assert beta == pytest.approx(1.6)

    def test_huge_se_collapses_to_prior(self):
        from models.capm import shrink_beta
        beta, w = shrink_beta(1.6, 50.0, prior_mean=1.0)
        assert w < 0.001
        assert beta == pytest.approx(1.0, abs=0.01)

    def test_weight_monotone_in_se(self):
        from models.capm import shrink_beta
        weights = [shrink_beta(1.2, se)[1] for se in (0.0, 0.05, 0.1, 0.2, 0.4, 1.0)]
        assert weights == sorted(weights, reverse=True)
        assert all(0.0 < w <= 1.0 for w in weights)

    def test_calibration_matches_docstring(self):
        """prior_sd 0.20: w=0.80 at se=0.10, 0.64 at se=0.15, 0.50 at se=0.20."""
        from models.capm import shrink_beta
        assert shrink_beta(1.0, 0.10, prior_sd=0.20)[1] == pytest.approx(0.80)
        assert shrink_beta(1.0, 0.15, prior_sd=0.20)[1] == pytest.approx(0.64)
        assert shrink_beta(1.0, 0.20, prior_sd=0.20)[1] == pytest.approx(0.50)

    def test_prior_mean_respected(self):
        from models.capm import shrink_beta
        beta, w = shrink_beta(1.4, 0.2, prior_mean=0.9, prior_sd=0.2)
        assert w == pytest.approx(0.5)
        assert beta == pytest.approx(0.5 * 1.4 + 0.5 * 0.9)

    def test_zero_prior_sd_returns_prior(self):
        from models.capm import shrink_beta
        assert shrink_beta(2.0, 0.1, prior_mean=1.0, prior_sd=0.0) == (1.0, 0.0)

    def test_rejects_bad_se(self):
        from models.capm import shrink_beta
        for bad in (None, float('nan'), -0.1):
            with pytest.raises(ValueError):
                shrink_beta(1.0, bad)

    def test_calculate_beta_reports_shrunk(self, synthetic_returns):
        stock, market = synthetic_returns
        r = calculate_beta(stock, market)
        lo, hi = sorted([r['raw_beta'], 1.0])
        assert lo <= r['shrunk_beta'] <= hi
        assert 0.0 < r['shrink_weight'] <= 1.0
        # 500 clean observations: the regression dominates the prior.
        assert r['shrink_weight'] > 0.9
        # Blume adjustment is untouched by the new field.
        assert r['adjusted_beta'] == pytest.approx((2 / 3) * r['raw_beta'] + 1 / 3)

    def test_noisy_beta_warns_about_shrinkage(self):
        rng = np.random.default_rng(3)
        m = rng.normal(0.0, 0.02, 100)
        s = m + rng.normal(0.0, 0.08, 100)
        r = calculate_beta(s, m)
        assert r['shrink_weight'] < 0.5
        assert any('shrunk' in w for w in r['warnings'])


# ---------------------------------------------------------------------------
# weekly_returns / rolling_betas
# ---------------------------------------------------------------------------

def _daily(idx, seed=0, beta=1.2):
    rng = np.random.default_rng(seed)
    m = rng.normal(0.0004, 0.01, len(idx))
    s = beta * m + rng.normal(0, 0.006, len(idx))
    import pandas as pd
    return (pd.Series(50 * np.cumprod(1 + s), index=idx),
            pd.Series(100 * np.cumprod(1 + m), index=idx))


class TestWeeklyReturns:
    def test_full_weeks_give_one_return_per_week_boundary(self):
        import pandas as pd
        from models.capm import weekly_returns
        idx = pd.bdate_range('2024-01-01', periods=50)  # Mon .. 10 full weeks
        s, m = _daily(idx)
        s_ret, m_ret, wk = weekly_returns(s, m)
        assert len(s_ret) == len(m_ret) == 9
        assert all(d.dayofweek == 4 for d in wk)  # Friday-labelled bins
        # Weekly return = Friday-to-Friday close change.
        fridays = s[s.index.dayofweek == 4]
        assert s_ret[0] == pytest.approx(fridays.iloc[1] / fridays.iloc[0] - 1)

    def test_partial_final_week_is_dropped(self):
        import pandas as pd
        from models.capm import weekly_returns
        idx = pd.bdate_range('2024-01-01', periods=52)  # 10 weeks + Mon, Tue
        s, m = _daily(idx)
        s_ret, _, wk = weekly_returns(s, m)
        assert len(s_ret) == 9
        assert wk[-1] == pd.Timestamp('2024-03-08')

    def test_partial_first_week_is_dropped(self):
        import pandas as pd
        from models.capm import weekly_returns
        idx = pd.bdate_range('2024-01-04', periods=52)  # Thu, Fri + 10 weeks
        s, m = _daily(idx)
        s_ret, _, wk = weekly_returns(s, m)
        assert len(s_ret) == 9
        assert wk[0] == pd.Timestamp('2024-01-19')

    def test_holiday_shortened_week_is_kept(self):
        import pandas as pd
        from models.capm import weekly_returns
        idx = pd.bdate_range('2024-01-01', periods=50)
        idx = idx.drop(pd.Timestamp('2024-03-08'))  # last Friday is a holiday
        s, m = _daily(idx)
        s_ret, _, wk = weekly_returns(s, m)
        assert len(s_ret) == 9
        assert wk[-1] == pd.Timestamp('2024-03-08')  # bin label; close is Thursday's

    def test_aligns_on_shared_dates_and_dedupes(self):
        import pandas as pd
        from models.capm import weekly_returns
        idx = pd.bdate_range('2024-01-01', periods=50)
        s, m = _daily(idx)
        s = s.drop(idx[7])                       # stock halted one day
        m = pd.concat([m, m.iloc[[3]]]).sort_index()  # duplicated market row
        s_ret, m_ret, _ = weekly_returns(s, m)
        assert len(s_ret) == len(m_ret) == 9
        assert np.all(np.isfinite(s_ret)) and np.all(np.isfinite(m_ret))

    def test_no_overlap_returns_empty(self):
        import pandas as pd
        from models.capm import weekly_returns
        s, _ = _daily(pd.bdate_range('2024-01-01', periods=20))
        _, m = _daily(pd.bdate_range('2025-01-01', periods=20))
        s_ret, m_ret, wk = weekly_returns(s, m)
        assert len(s_ret) == len(m_ret) == len(wk) == 0


class TestRollingBetas:
    @staticmethod
    def _weekly(n, seed=1, beta=1.3):
        rng = np.random.default_rng(seed)
        m = rng.normal(0.001, 0.02, n)
        return beta * m + rng.normal(0, 0.015, n), m

    def test_windows_match_direct_regression(self):
        from models.capm import rolling_betas
        s, m = self._weekly(300)
        rb = rolling_betas(s, m)
        assert set(rb) == {'1y', '3y', '5y', 'stability'}
        for label, n in (('1y', 52), ('3y', 156), ('5y', 260)):
            direct = calculate_beta(s[-n:], m[-n:])
            assert rb[label]['beta'] == pytest.approx(direct['raw_beta'], abs=1e-4)
            assert rb[label]['shrunk'] == pytest.approx(direct['shrunk_beta'], abs=1e-4)
            assert rb[label]['n'] == n
        raws = [rb[k]['beta'] for k in ('1y', '3y', '5y')]
        assert rb['stability'] == pytest.approx(float(np.std(raws)), abs=1e-3)

    def test_short_history_only_reports_covered_windows(self):
        from models.capm import rolling_betas
        s, m = self._weekly(100)
        rb = rolling_betas(s, m)
        assert set(rb) == {'1y', 'stability'}
        assert rb['stability'] is None

    def test_too_short_returns_empty(self):
        from models.capm import rolling_betas
        s, m = self._weekly(20)
        assert rolling_betas(s, m) == {}


class TestGGMForward:
    def test_forward_yield_is_not_grown_again(self):
        assert ggm_implied_re(0.03, 0.03, forward=True) == pytest.approx(0.06)
        assert ggm_implied_re(0.03, 0.03, forward=False) == pytest.approx(0.03 * 1.03 + 0.03)

    def test_forward_none_for_nonpositive_yield(self):
        assert ggm_implied_re(0.0, 0.03, forward=True) is None
