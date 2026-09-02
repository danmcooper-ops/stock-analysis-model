# tests/test_ddm.py
import pytest
import numpy as np


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


class TestAnnualiseDividends:
    """Partial-year drop + gap-year reindexing (Phase 1.1)."""

    def _series(self, pairs):
        import pandas as pd
        idx = pd.to_datetime([p[0] for p in pairs])
        return pd.Series([p[1] for p in pairs], index=idx)

    def test_partial_current_year_dropped(self):
        from scripts.analyze_stock import _annualise_dividends
        s = self._series([('2023-03', .25), ('2023-06', .25), ('2023-09', .25),
                          ('2023-12', .25), ('2024-03', .25), ('2024-06', .25),
                          ('2024-09', .25), ('2024-12', .25),
                          ('2025-03', .30), ('2025-06', .30)])  # 2025 partial
        # as_of 2025 → drop 2025, keep full 2023,2024
        assert _annualise_dividends(s, as_of_year=2025) == [1.0, 1.0]

    def test_suspension_year_filled_with_zero(self):
        from scripts.analyze_stock import _annualise_dividends
        s = self._series([('2021-06', 1.0), ('2022-06', 1.1),
                          ('2024-06', 1.3), ('2025-06', 1.4)])  # 2023 missing
        assert _annualise_dividends(s, as_of_year=2026) == [1.0, 1.1, 0.0, 1.3, 1.4]

    def test_single_partial_year_kept(self):
        """A freshly-initiated payer with only the current year still surfaces."""
        from scripts.analyze_stock import _annualise_dividends
        s = self._series([('2026-03', .20), ('2026-06', .20)])
        assert _annualise_dividends(s, as_of_year=2026) == [0.40]

    def test_empty(self):
        from scripts.analyze_stock import _annualise_dividends
        import pandas as pd
        assert _annualise_dividends(pd.Series(dtype=float)) == []


class TestDivCagrOverGaps:
    def test_span_uses_positions_not_filtered_length(self):
        from models.ddm import estimate_ddm_growth
        # [1.0, 1.1, 0, 1.3, 1.4]: first/last positive span = 4 years
        g = estimate_ddm_growth([1.0, 1.1, 0.0, 1.3, 1.4],
                                payout=0.5, roe=None, analyst_ltg=None)
        import pytest
        assert g['div_cagr'] == pytest.approx((1.4 / 1.0) ** (1 / 4) - 1, abs=1e-4)


# ---------------------------------------------------------------------------
# Forward-rate DPS (yfinance dividendRate is D1, not D0)
# ---------------------------------------------------------------------------

class TestForwardDps:
    """dps_is_forward=True pins the year-1 dividend at `dps` instead of
    compounding a forward rate a second time."""

    D, G, TG, RE = 2.0, 0.07, 0.03, 0.10

    def test_two_stage_forward_equals_trailing_backed_out(self):
        fwd = two_stage_ddm(self.D, self.G, self.TG, self.RE, dps_is_forward=True)
        trailing = two_stage_ddm(self.D / (1 + self.G), self.G, self.TG, self.RE)
        assert fwd == pytest.approx(trailing)

    def test_two_stage_forward_lower_by_one_year_of_growth(self):
        fwd = two_stage_ddm(self.D, self.G, self.TG, self.RE, dps_is_forward=True)
        d0 = two_stage_ddm(self.D, self.G, self.TG, self.RE)
        assert d0 / fwd == pytest.approx(1 + self.G)

    def test_two_stage_year1_dividend_is_dps(self):
        """Hand-built PV with D1 = dps must match the forward-mode value."""
        pv = sum(self.D * (1 + self.G) ** (t - 1) / (1 + self.RE) ** t
                 for t in range(1, 6))
        d5 = self.D * (1 + self.G) ** 4
        tv = d5 * (1 + self.TG) / (self.RE - self.TG) / (1 + self.RE) ** 5
        fwd = two_stage_ddm(self.D, self.G, self.TG, self.RE, dps_is_forward=True)
        assert fwd == pytest.approx(pv + tv)

    def test_two_stage_zero_growth_unchanged(self):
        """With g = 0 the forward and trailing readings coincide."""
        assert two_stage_ddm(self.D, 0.0, self.TG, self.RE, dps_is_forward=True) == \
            pytest.approx(two_stage_ddm(self.D, 0.0, self.TG, self.RE))

    def test_h_model_forward_equals_trailing_backed_out(self):
        fwd = ddm_h_model(self.D, self.G, self.TG, self.RE, half_life=2.5,
                          dps_is_forward=True)
        trailing = ddm_h_model(self.D / (1 + self.G), self.G, self.TG, self.RE,
                               half_life=2.5)
        assert fwd == pytest.approx(trailing)

    def test_envelope_records_d0(self):
        from models.ddm import two_stage_ddm_valuation
        v = two_stage_ddm_valuation(self.D, self.G, self.TG, self.RE, dps_is_forward=True)
        assert v.inputs_used['dps_is_forward'] is True
        assert v.inputs_used['d0'] == pytest.approx(self.D / (1 + self.G))
        v0 = two_stage_ddm_valuation(self.D, self.G, self.TG, self.RE)
        assert v0.inputs_used['dps_is_forward'] is False
        assert v0.inputs_used['d0'] == pytest.approx(self.D)

    def test_monte_carlo_forward_median_tracks_deterministic(self):
        mc = monte_carlo_ddm(self.D, self.G, self.RE, self.TG, n=2000, dps_is_forward=True)
        det = two_stage_ddm(self.D, self.G, self.TG, self.RE, dps_is_forward=True)
        assert mc['median_fv'] == pytest.approx(det, rel=0.05)
        mc0 = monte_carlo_ddm(self.D, self.G, self.RE, self.TG, n=2000)
        assert mc0['median_fv'] > mc['median_fv']

    def test_monte_carlo_forward_survives_extreme_growth_draws(self):
        """A negative growth draw below −100% must not flip or blow up D0."""
        mc = monte_carlo_ddm(self.D, -0.10, self.RE, self.TG, n=2000,
                             g_sigma=0.60, dps_is_forward=True)
        assert mc is not None
        assert np.isfinite(mc['median_fv']) and mc['median_fv'] > 0

    def test_default_is_trailing(self):
        """Existing callers passing D0 see no change."""
        assert two_stage_ddm(self.D, self.G, self.TG, self.RE) == \
            pytest.approx(two_stage_ddm(self.D, self.G, self.TG, self.RE,
                                        dps_is_forward=False))


class TestAnnualiseDividendsPartialFirstYear:
    """A mid-year initiation (or a feed starting mid-year) must not enter the
    CAGR span as a full-year DPS."""

    def _series(self, pairs):
        import pandas as pd
        idx = pd.to_datetime([p[0] for p in pairs])
        return pd.Series([p[1] for p in pairs], index=idx)

    def _quarterly(self, start_year, end_year, start_q=0.25, growth=0.05):
        rows, q = [], start_q
        for y in range(start_year, end_year + 1):
            for m in (3, 6, 9, 12):
                rows.append((f'{y}-{m:02d}-01', q))
            q *= 1 + growth
        return rows

    def test_partial_initiation_year_dropped(self):
        from scripts.analyze_stock import _annualise_dividends
        rows = [('2021-12-01', 0.25)] + self._quarterly(2022, 2025, 0.2625)
        ann = _annualise_dividends(self._series(rows), as_of_year=2026)
        assert len(ann) == 4
        assert ann[0] == pytest.approx(1.05)

    def test_partial_first_year_no_longer_inflates_cagr(self):
        from scripts.analyze_stock import _annualise_dividends
        rows = [('2016-12-01', 0.25)] + self._quarterly(2017, 2025, 0.2625)
        ann = _annualise_dividends(self._series(rows), as_of_year=2026)
        g = estimate_ddm_growth(ann, None, None, None)
        assert g['div_cagr'] == pytest.approx(0.05, abs=1e-6)

    def test_full_first_year_kept(self):
        from scripts.analyze_stock import _annualise_dividends
        rows = self._quarterly(2022, 2025)
        ann = _annualise_dividends(self._series(rows), as_of_year=2026)
        assert len(ann) == 4
        assert ann[0] == pytest.approx(1.0)

    def test_annual_payer_first_year_kept(self):
        """One payment a year is the modal count, so nothing is dropped."""
        from scripts.analyze_stock import _annualise_dividends
        s = self._series([('2021-06', 1.0), ('2022-06', 1.1),
                          ('2023-06', 1.2), ('2024-06', 1.3)])
        assert _annualise_dividends(s, as_of_year=2025) == [1.0, 1.1, 1.2, 1.3]

    def test_partial_first_and_partial_current_year(self):
        from scripts.analyze_stock import _annualise_dividends
        rows = ([('2022-12-01', 0.25)] + self._quarterly(2023, 2025, 0.25, 0.0)
                + [('2026-03-01', 0.25)])
        assert _annualise_dividends(self._series(rows), as_of_year=2026) == \
            pytest.approx([1.0, 1.0, 1.0])

    def test_two_partial_years_keeps_the_older_one(self):
        """Partial first year + partial current year: current is dropped and
        the remaining sole year is kept so a new payer still surfaces."""
        from scripts.analyze_stock import _annualise_dividends
        s = self._series([('2025-12-01', 0.25), ('2026-03-01', 0.25)])
        assert _annualise_dividends(s, as_of_year=2026) == [0.25]

    def test_special_dividend_year_does_not_shift_modal_count(self):
        """A year with an extra special payment doesn't make regular years
        look partial."""
        from scripts.analyze_stock import _annualise_dividends
        rows = self._quarterly(2022, 2025, 0.25, 0.0) + [('2024-07-15', 1.0)]
        ann = _annualise_dividends(self._series(rows), as_of_year=2026)
        assert len(ann) == 4
        assert ann[2] == pytest.approx(2.0)
