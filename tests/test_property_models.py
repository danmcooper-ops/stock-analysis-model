# tests/test_property_models.py
"""Property-based tests for the valuation math (hypothesis).

Example tests pin specific input/output pairs; these sweep thousands of
generated inputs per run and assert the *invariants* a valuation model
must never violate: outputs are None or finite positive (never NaN/inf),
fair value moves monotonically with growth and discount rate, beta is
self-consistent, and Monte Carlo percentiles are ordered.

Input ranges mirror the bounds the models themselves enforce via
_dcf_validate_core / _validate_numeric, minus a margin so the guards
(e.g. two_stage_ev's min_spread shift) don't mask the property under test.
"""

import warnings

import numpy as np
import pytest
from hypothesis import assume, given, settings, strategies as st


from models.capm import calculate_beta
from models.dcf import two_stage_ev, fair_value_per_share, monte_carlo_dcf
from models.ddm import monte_carlo_ddm, two_stage_ddm, ddm_h_model
from models.epv import earnings_power_value
from models.rim import residual_income_model

# The model layer warns (RuntimeWarning) on aggressive-but-valid inputs;
# generated sweeps hit those constantly and the warnings are expected.
pytestmark = pytest.mark.filterwarnings('ignore::RuntimeWarning')

SETTINGS = settings(max_examples=75, deadline=None)

fcf = st.floats(min_value=1.0, max_value=1e12, allow_nan=False, allow_infinity=False)
growth = st.floats(min_value=-0.4, max_value=0.5, allow_nan=False, allow_infinity=False)
term_growth = st.floats(min_value=-0.05, max_value=0.05, allow_nan=False, allow_infinity=False)
discount = st.floats(min_value=0.06, max_value=0.30, allow_nan=False, allow_infinity=False)
shares = st.floats(min_value=1.0, max_value=1e11, allow_nan=False, allow_infinity=False)


def _finite_positive_or_none(v):
    assert v is None or (np.isfinite(v) and v > 0), f'invalid model output: {v!r}'


class TestTwoStageEv:
    @SETTINGS
    @given(base_fcf=fcf, g=growth, d=discount, tg=term_growth)
    def test_never_nan_or_inf(self, base_fcf, g, d, tg):
        _finite_positive_or_none(two_stage_ev(base_fcf, g, d, tg))

    @SETTINGS
    @given(base_fcf=fcf, g1=growth, g2=growth, d=discount, tg=term_growth)
    def test_monotone_in_growth(self, base_fcf, g1, g2, d, tg):
        """More growth can never be worth less, all else equal."""
        assume(d >= tg + 0.03)  # keep both runs clear of the min-spread shift
        g_lo, g_hi = sorted((g1, g2))
        ev_lo = two_stage_ev(base_fcf, g_lo, d, tg)
        ev_hi = two_stage_ev(base_fcf, g_hi, d, tg)
        assume(ev_lo is not None and ev_hi is not None)
        assert ev_hi >= ev_lo * (1 - 1e-9)

    @SETTINGS
    @given(base_fcf=fcf, g=growth, d1=discount, d2=discount, tg=term_growth)
    def test_antitone_in_discount_rate(self, base_fcf, g, d1, d2, tg):
        """A higher discount rate can never raise the value, all else equal."""
        d_lo, d_hi = sorted((d1, d2))
        assume(d_lo >= tg + 0.03)
        ev_lo_rate = two_stage_ev(base_fcf, g, d_lo, tg)
        ev_hi_rate = two_stage_ev(base_fcf, g, d_hi, tg)
        assume(ev_lo_rate is not None and ev_hi_rate is not None)
        assert ev_hi_rate <= ev_lo_rate * (1 + 1e-9)

    @SETTINGS
    @given(base_fcf=fcf, g=growth, d=discount, tg=term_growth,
           scale=st.floats(min_value=1.5, max_value=100.0))
    def test_linear_in_base_fcf(self, base_fcf, g, d, tg, scale):
        """Doubling every cash flow doubles the enterprise value."""
        assume(d >= tg + 0.03)
        ev = two_stage_ev(base_fcf, g, d, tg)
        assume(ev is not None)
        ev_scaled = two_stage_ev(base_fcf * scale, g, d, tg)
        assume(ev_scaled is not None)
        assert ev_scaled == pytest.approx(ev * scale, rel=1e-6)


class TestFairValuePerShare:
    @SETTINGS
    @given(ev=st.floats(min_value=1.0, max_value=1e13),
           net_debt=st.floats(min_value=-1e12, max_value=1e12),
           n=shares)
    def test_never_nan_and_none_when_insolvent(self, ev, net_debt, n):
        fv = fair_value_per_share(ev, net_debt, n)
        if ev - net_debt <= 0:
            assert fv is None
        else:
            assert np.isfinite(fv) and fv > 0
            assert fv == pytest.approx((ev - net_debt) / n)


class TestCalculateBeta:
    returns = st.lists(
        st.floats(min_value=-0.2, max_value=0.2, allow_nan=False, allow_infinity=False),
        min_size=30, max_size=80)

    @SETTINGS
    @given(r=returns)
    def test_self_beta_is_one(self, r):
        arr = np.asarray(r)
        assume(float(np.var(arr, ddof=1)) > 1e-12)
        result = calculate_beta(arr, arr)
        assert result['raw_beta'] == pytest.approx(1.0)
        assert result['r_squared'] == pytest.approx(1.0)

    @SETTINGS
    @given(r=returns, c=st.floats(min_value=0.5, max_value=3.0))
    def test_scaling_stock_scales_beta(self, r, c):
        m = np.asarray(r)
        assume(float(np.var(m, ddof=1)) > 1e-12)
        result = calculate_beta(m * c, m)
        assert result['raw_beta'] == pytest.approx(c, rel=1e-6)

    @SETTINGS
    @given(r=returns, r2=returns)
    def test_output_never_nan(self, r, r2):
        s = np.asarray(r)
        m = np.asarray(r2[:len(r)])
        assume(len(m) == len(s))
        assume(float(np.var(s, ddof=1)) > 1e-12)
        assume(float(np.var(m, ddof=1)) > 1e-12)
        result = calculate_beta(s, m)
        for key in ('raw_beta', 'adjusted_beta', 'r_squared', 'se_beta'):
            assert result[key] is None or np.isfinite(result[key]), key


class TestEquityFlowModels:
    @SETTINGS
    @given(dps=st.floats(min_value=0.01, max_value=100.0),
           high_g=st.floats(min_value=-0.3, max_value=0.4),
           tg=term_growth, re=discount)
    def test_two_stage_ddm_finite(self, dps, high_g, tg, re):
        _finite_positive_or_none(two_stage_ddm(dps, high_g, tg, re))

    @SETTINGS
    @given(dps=st.floats(min_value=0.01, max_value=100.0),
           short_g=st.floats(min_value=-0.3, max_value=0.4),
           tg=term_growth, re=discount)
    def test_h_model_finite(self, dps, short_g, tg, re):
        _finite_positive_or_none(ddm_h_model(dps, short_g, tg, re))

    @SETTINGS
    @given(ebit=st.floats(min_value=1.0, max_value=1e12),
           tax=st.floats(min_value=0.0, max_value=0.5),
           coc=st.floats(min_value=0.05, max_value=0.30),
           n=shares,
           cash=st.floats(min_value=0.0, max_value=1e11),
           debt=st.floats(min_value=0.0, max_value=1e12))
    def test_epv_finite(self, ebit, tax, coc, n, cash, debt):
        _finite_positive_or_none(
            earnings_power_value(ebit, tax, coc, n, excess_cash=cash, total_debt=debt))

    @SETTINGS
    @given(bvps=st.floats(min_value=0.01, max_value=1e4),
           roe=st.floats(min_value=-0.5, max_value=1.0),
           re=discount)
    def test_rim_finite(self, bvps, roe, re):
        _finite_positive_or_none(residual_income_model(bvps, roe, re))


class TestMonteCarloDcf:
    @settings(max_examples=25, deadline=None)
    @given(base_fcf=st.floats(min_value=1e6, max_value=1e11),
           g=st.floats(min_value=-0.1, max_value=0.3),
           d=st.floats(min_value=0.07, max_value=0.20),
           tg=st.floats(min_value=0.0, max_value=0.04),
           n=st.floats(min_value=1e6, max_value=1e10))
    def test_percentiles_ordered_and_finite(self, base_fcf, g, d, tg, n):
        assume(d >= tg + 0.04)
        result = monte_carlo_dcf(base_fcf, g, d, tg,
                                 net_debt=0, shares_outstanding=n,
                                 n_iterations=300)
        assume(result is not None)
        assert result['p10_fv'] <= result['median_fv'] <= result['p90_fv']
        for key in ('median_fv', 'mean_fv', 'p10_fv', 'p90_fv', 'std_fv'):
            assert np.isfinite(result[key]), key

    @settings(max_examples=25, deadline=None)
    @given(base_fcf=st.floats(min_value=1e6, max_value=1e11),
           g=st.floats(min_value=-0.1, max_value=0.3),
           d=st.floats(min_value=0.07, max_value=0.20),
           tg=st.floats(min_value=0.0, max_value=0.04),
           ebitda_mult=st.floats(min_value=0.5, max_value=5.0),
           exit_mult=st.floats(min_value=3.0, max_value=25.0))
    def test_exit_multiple_leg_ordered_finite_and_bookkept(self, base_fcf, g, d, tg,
                                                           ebitda_mult, exit_mult):
        assume(d >= tg + 0.04)
        result = monte_carlo_dcf(base_fcf, g, d, tg, net_debt=0, shares_outstanding=1e8,
                                 base_ebitda=base_fcf * ebitda_mult, exit_multiple=exit_mult,
                                 n_iterations=300)
        assume(result is not None)
        assert 0 <= result['p10_fv'] <= result['median_fv'] <= result['p90_fv']
        assert 0.0 <= result['clip_rate'] <= 1.0
        assert 0.0 <= result['invalid_rate'] <= 0.9
        assert result['n_valid'] == pytest.approx(300 * (1 - result['invalid_rate']), abs=1e-6)
        for key in ('median_fv', 'mean_fv', 'p10_fv', 'p90_fv', 'std_fv'):
            assert np.isfinite(result[key]), key

    @settings(max_examples=25, deadline=None)
    @given(base_fcf=st.floats(min_value=1e6, max_value=1e11),
           g=st.floats(min_value=-0.1, max_value=0.3),
           d=st.floats(min_value=0.07, max_value=0.20),
           tg=st.floats(min_value=0.0, max_value=0.04),
           n=st.floats(min_value=1e6, max_value=1e10))
    def test_tiny_sigma_median_matches_point_estimate(self, base_fcf, g, d, tg, n):
        """Collapse the sampling noise: the simulation must reproduce the
        deterministic two-stage value, whatever the (valid) inputs."""
        assume(d >= tg + 0.04)
        result = monte_carlo_dcf(base_fcf, g, d, tg, net_debt=0, shares_outstanding=n,
                                 n_iterations=400, growth_sigma=1e-9, wacc_sigma=1e-9,
                                 tg_sigma=1e-9)
        point = fair_value_per_share(two_stage_ev(base_fcf, g, d, tg), 0, n)
        assert result is not None and point is not None
        assert result['median_fv'] == pytest.approx(point, rel=2e-2)

    @settings(max_examples=25, deadline=None)
    @given(base_fcf=st.floats(min_value=1e6, max_value=1e11),
           g=st.floats(min_value=-0.1, max_value=0.3),
           d=st.floats(min_value=0.07, max_value=0.20),
           tg=st.floats(min_value=0.0, max_value=0.04),
           scale=st.floats(min_value=0.1, max_value=10.0))
    def test_homogeneous_in_base_fcf(self, base_fcf, g, d, tg, scale):
        """Same seed → same draws: scaling the cash flow scales every
        per-share statistic exactly and leaves the CV unchanged."""
        assume(d >= tg + 0.04)
        a = monte_carlo_dcf(base_fcf, g, d, tg, net_debt=0, shares_outstanding=1e8,
                            n_iterations=200)
        b = monte_carlo_dcf(base_fcf * scale, g, d, tg, net_debt=0, shares_outstanding=1e8,
                            n_iterations=200)
        assume(a is not None and b is not None)
        for key in ('median_fv', 'mean_fv', 'p10_fv', 'p90_fv', 'std_fv'):
            assert b[key] == pytest.approx(a[key] * scale, rel=1e-9), key
        assert b['cv'] == pytest.approx(a['cv'], rel=1e-9)


class TestMonteCarloDdm:
    @settings(max_examples=25, deadline=None)
    @given(dps=st.floats(min_value=0.01, max_value=50.0),
           g=st.floats(min_value=-0.1, max_value=0.25),
           re=st.floats(min_value=0.05, max_value=0.20),
           tg=st.floats(min_value=0.0, max_value=0.04))
    def test_percentiles_ordered_finite_and_bookkept(self, dps, g, re, tg):
        assume(re >= tg + 0.03)
        result = monte_carlo_ddm(dps, g, re, tg, n=300)
        assume(result is not None)
        assert 0 <= result['p10_fv'] <= result['median_fv'] <= result['p90_fv']
        assert 0.0 <= result['clip_rate'] <= 1.0
        assert result['n_valid'] == pytest.approx(300 * (1 - result['invalid_rate']), abs=1e-6)
        for key in ('median_fv', 'mean_fv', 'p10_fv', 'p90_fv', 'std_fv'):
            assert np.isfinite(result[key]), key

    @settings(max_examples=25, deadline=None)
    @given(dps=st.floats(min_value=0.01, max_value=50.0),
           g=st.floats(min_value=-0.1, max_value=0.25),
           re=st.floats(min_value=0.05, max_value=0.20),
           tg=st.floats(min_value=0.0, max_value=0.04))
    def test_tiny_sigma_median_matches_point_estimate(self, dps, g, re, tg):
        """Holds INSIDE the 2% minimum-spread wall too: both sides must
        substitute the same effective terminal growth."""
        assume(re > tg)
        result = monte_carlo_ddm(dps, g, re, tg, n=400,
                                 g_sigma=1e-9, re_sigma=1e-9, tg_sigma=1e-9)
        point = two_stage_ddm(dps, g, tg, re)
        assert result is not None and point is not None
        assert result['median_fv'] == pytest.approx(point, rel=2e-2)

    @settings(max_examples=25, deadline=None)
    @given(dps=st.floats(min_value=0.01, max_value=50.0),
           g=st.floats(min_value=-0.1, max_value=0.25),
           re=st.floats(min_value=0.05, max_value=0.20),
           tg=st.floats(min_value=0.0, max_value=0.04))
    def test_defined_exactly_when_point_estimate_is_defined(self, dps, g, re, tg):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            result = monte_carlo_ddm(dps, g, re, tg, n=200)
        assert (result is None) == (two_stage_ddm(dps, g, tg, re) is None)
