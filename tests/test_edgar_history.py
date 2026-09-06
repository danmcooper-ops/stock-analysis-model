# tests/test_edgar_history.py

import pytest


from scripts.analyze_stock import _flow_to_annual, _stock_to_annual


class TestFlowToAnnual:
    def test_int_keyed_passthrough(self):
        """New EDGAR client format: int year keys, one value per FY."""
        history = {2020: 100.0, 2021: 110.0, 2022: 121.0}
        assert _flow_to_annual(history) == {2020: 100.0, 2021: 110.0, 2022: 121.0}

    def test_year_string_keyed_passthrough(self):
        """JSON round-trip turns int year keys into 4-char digit strings."""
        history = {'2020': 100.0, '2021': 110.0, '2022': 121.0}
        assert _flow_to_annual(history) == {2020: 100.0, 2021: 110.0, 2022: 121.0}

    def test_legacy_full_year_only(self):
        """Legacy date-keyed format: one annual value per year passes through."""
        history = {'2020-12-31': 100.0, '2021-12-31': 110.0}
        assert _flow_to_annual(history) == {2020: 100.0, 2021: 110.0}

    def test_legacy_four_quarters_summed(self):
        """Legacy date-keyed format: four quarterly entries sum to annual."""
        history = {
            '2020-03-31': 25.0,
            '2020-06-30': 25.0,
            '2020-09-30': 25.0,
            '2020-12-31': 25.0,
        }
        assert _flow_to_annual(history) == {2020: 100.0}

    def test_legacy_mixed_period_year_dropped(self):
        """Legacy date-keyed format: 2 or 3 entries per year are dropped (the
        ambiguous mixed-period case from the old quarterly extractor)."""
        history = {
            '2020-03-31': 25.0,
            '2020-12-31': 100.0,  # FY + Q1 → ambiguous, drop
            '2021-12-31': 110.0,
        }
        assert _flow_to_annual(history) == {2021: 110.0}

    def test_empty(self):
        assert _flow_to_annual({}) == {}
        assert _flow_to_annual(None) == {}

    def test_none_values_skipped(self):
        history = {2020: 100.0, 2021: None, 2022: 121.0}
        assert _flow_to_annual(history) == {2020: 100.0, 2022: 121.0}


class TestStockToAnnual:
    def test_latest_per_year(self):
        """Point-in-time series: keep latest observation per calendar year."""
        history = {
            '2020-03-31': 100.0,
            '2020-12-31': 95.0,   # later in 2020 → keep
            '2021-06-30': 90.0,
        }
        assert _stock_to_annual(history) == {2020: 95.0, 2021: 90.0}

    def test_empty(self):
        assert _stock_to_annual({}) == {}
        assert _stock_to_annual(None) == {}


# ---------------------------------------------------------------------------
# derive_edgar_metrics — through-cycle operating margin (2026-07 rebalance)
# ---------------------------------------------------------------------------

class TestOpMarginHistory:
    def _hist(self, years):
        """edgar_history with flat revenue 1000 and op income by year."""
        return {
            'revenue_history': {y: 1000.0 for y in years},
            'operating_income_history': {y: 150.0 + (y % 2) * 50.0
                                         for y in years},
        }

    def test_ten_year_average(self):
        from scripts.analyze_stock import derive_edgar_metrics
        years = list(range(2015, 2025))  # 10 years
        m = derive_edgar_metrics(self._hist(years))
        assert m['op_margin_hist_years'] == 10
        # margins alternate 0.15 / 0.20 → mean 0.175
        assert m['op_margin_avg_10y'] == pytest.approx(0.175)

    def test_window_capped_at_ten_years(self):
        from scripts.analyze_stock import derive_edgar_metrics
        years = list(range(2009, 2025))  # 16 years
        m = derive_edgar_metrics(self._hist(years))
        assert m['op_margin_hist_years'] == 10

    def test_missing_series_yields_zero_years(self):
        from scripts.analyze_stock import derive_edgar_metrics
        m = derive_edgar_metrics({'revenue_history': {2024: 1000.0}})
        assert m['op_margin_avg_10y'] is None
        assert m['op_margin_hist_years'] == 0

    def test_zero_revenue_years_excluded(self):
        from scripts.analyze_stock import derive_edgar_metrics
        hist = {
            'revenue_history': {2022: 0.0, 2023: 1000.0, 2024: 1000.0},
            'operating_income_history': {2022: 100.0, 2023: 200.0, 2024: 200.0},
        }
        m = derive_edgar_metrics(hist)
        assert m['op_margin_hist_years'] == 2
        assert m['op_margin_avg_10y'] == pytest.approx(0.20)


class TestIntCovEdgar:
    """EBIT / interest expense derived from EDGAR — the fallback for rows
    where yfinance surfaces no income statement."""

    def test_uses_latest_common_year(self):
        from scripts.analyze_stock import derive_edgar_metrics
        hist = {
            'operating_income_history': {2023: 8000.0, 2024: 8699.0, 2025: 8127.0},
            'interest_expense_history': {2023: 900.0, 2024: 1058.0, 2025: 1344.0},
        }
        m = derive_edgar_metrics(hist)
        # latest common year 2025: 8127 / 1344
        assert m['int_cov_edgar'] == pytest.approx(6.046875)

    def test_intersects_years_across_the_two_series(self):
        """The newest op-income year with no matching interest year is skipped
        rather than pairing mismatched periods."""
        from scripts.analyze_stock import derive_edgar_metrics
        hist = {
            'operating_income_history': {2023: 1000.0, 2024: 2000.0},
            'interest_expense_history': {2023: 100.0},
        }
        m = derive_edgar_metrics(hist)
        assert m['int_cov_edgar'] == pytest.approx(10.0)

    def test_negative_ebit_propagates(self):
        from scripts.analyze_stock import derive_edgar_metrics
        hist = {
            'operating_income_history': {2024: -500.0},
            'interest_expense_history': {2024: 200.0},
        }
        m = derive_edgar_metrics(hist)
        assert m['int_cov_edgar'] == pytest.approx(-2.5)

    def test_zero_or_missing_interest_yields_none(self):
        from scripts.analyze_stock import derive_edgar_metrics
        assert derive_edgar_metrics({
            'operating_income_history': {2024: 500.0},
            'interest_expense_history': {2024: 0.0},
        })['int_cov_edgar'] is None
        assert derive_edgar_metrics({
            'operating_income_history': {2024: 500.0},
        })['int_cov_edgar'] is None

    def test_absent_history_yields_none(self):
        from scripts.analyze_stock import derive_edgar_metrics
        assert derive_edgar_metrics({})['int_cov_edgar'] is None
        assert derive_edgar_metrics(None)['int_cov_edgar'] is None


# derive_edgar_metrics — value-trap decline detectors (2026-08)
class TestTrapDetectors:
    def test_consecutive_revenue_declines(self):
        from scripts.analyze_stock import derive_edgar_metrics
        m = derive_edgar_metrics({'revenue_history': {
            2020: 100.0, 2021: 95.0, 2022: 90.0, 2023: 85.0}})
        assert m['rev_down_years'] == 3

    def test_rebound_resets_streak(self):
        from scripts.analyze_stock import derive_edgar_metrics
        # Down, then up, then down again: only the trailing decline counts.
        m = derive_edgar_metrics({'revenue_history': {
            2020: 100.0, 2021: 90.0, 2022: 95.0, 2023: 92.0}})
        assert m['rev_down_years'] == 1

    def test_growing_revenue_zero_streak(self):
        from scripts.analyze_stock import derive_edgar_metrics
        m = derive_edgar_metrics({'revenue_history': {
            2021: 100.0, 2022: 110.0, 2023: 121.0}})
        assert m['rev_down_years'] == 0

    def test_year_gap_breaks_streak(self):
        from scripts.analyze_stock import derive_edgar_metrics
        # 2021 missing: the 2022-vs-2020 comparison must not bridge the gap.
        m = derive_edgar_metrics({'revenue_history': {
            2019: 120.0, 2020: 110.0, 2022: 100.0, 2023: 90.0}})
        assert m['rev_down_years'] == 1

    def test_thin_revenue_history_none(self):
        from scripts.analyze_stock import derive_edgar_metrics
        m = derive_edgar_metrics({'revenue_history': {2022: 100.0, 2023: 90.0}})
        assert m['rev_down_years'] is None

    def test_fcf_neg_years(self):
        from scripts.analyze_stock import derive_edgar_metrics
        m = derive_edgar_metrics({
            'operating_cf_history': {2019: 10.0, 2020: 5.0, 2021: 8.0, 2022: 4.0, 2023: 6.0},
            'capex_history': {2019: 6.0, 2020: 9.0, 2021: 5.0, 2022: 9.0, 2023: 4.0},
        })
        # FCF: +4, -4, +3, -5, +2 → two negative years of five
        assert m['fcf_neg_years_5y'] == 2

    def test_fcf_neg_years_thin_none(self):
        from scripts.analyze_stock import derive_edgar_metrics
        m = derive_edgar_metrics({
            'operating_cf_history': {2022: 10.0, 2023: 5.0},
            'capex_history': {2022: 2.0, 2023: 2.0},
        })
        assert m['fcf_neg_years_5y'] is None

    def test_div_fcf_ratio_payer(self):
        from scripts.analyze_stock import derive_edgar_metrics
        m = derive_edgar_metrics({
            'operating_cf_history': {2021: 100.0, 2022: 100.0, 2023: 100.0},
            'capex_history': {2021: 20.0, 2022: 20.0, 2023: 20.0},
            # dividends_paid is a cash outflow (negative in filings)
            'dividends_paid_history': {2021: -60.0, 2022: -60.0, 2023: -60.0},
        })
        assert m['div_fcf_ratio_3y'] == pytest.approx(180.0 / 240.0)

    def test_div_fcf_ratio_payer_negative_fcf_capped(self):
        from scripts.analyze_stock import derive_edgar_metrics
        m = derive_edgar_metrics({
            'operating_cf_history': {2021: 10.0, 2022: 10.0, 2023: 10.0},
            'capex_history': {2021: 20.0, 2022: 20.0, 2023: 20.0},
            'dividends_paid_history': {2021: -5.0, 2022: -5.0, 2023: -5.0},
        })
        assert m['div_fcf_ratio_3y'] == 9.99

    def test_div_fcf_ratio_explicit_nonpayer_zero(self):
        from scripts.analyze_stock import derive_edgar_metrics
        m = derive_edgar_metrics({
            'operating_cf_history': {2021: 100.0, 2022: 100.0, 2023: 100.0},
            'capex_history': {2021: 20.0, 2022: 20.0, 2023: 20.0},
            'dividends_paid_history': {2021: 0.0, 2022: 0.0, 2023: 0.0},
        })
        assert m['div_fcf_ratio_3y'] == 0.0

    def test_div_fcf_ratio_absent_history_none(self):
        from scripts.analyze_stock import derive_edgar_metrics
        m = derive_edgar_metrics({
            'operating_cf_history': {2021: 100.0, 2022: 100.0, 2023: 100.0},
            'capex_history': {2021: 20.0, 2022: 20.0, 2023: 20.0},
        })
        assert m['div_fcf_ratio_3y'] is None

    def test_net_debt_slope_rising(self):
        from scripts.analyze_stock import derive_edgar_metrics
        m = derive_edgar_metrics({
            'revenue_history': {2020: 900.0, 2021: 950.0, 2022: 980.0, 2023: 1000.0},
            'total_debt_history': {2020: 100.0, 2021: 150.0, 2022: 200.0, 2023: 250.0},
            'cash_history': {2020: 50.0, 2021: 50.0, 2022: 50.0, 2023: 50.0},
        })
        # net debt 50 → 200 over 3 years on 1000 revenue → +0.05/yr
        assert m['net_debt_slope_3y'] == pytest.approx(0.05)

    def test_net_debt_slope_crossing_zero(self):
        from scripts.analyze_stock import derive_edgar_metrics
        # Net cash → net debt: a CAGR would be undefined; the slope is not.
        m = derive_edgar_metrics({
            'revenue_history': {2020: 1000.0, 2021: 1000.0, 2022: 1000.0, 2023: 1000.0},
            'total_debt_history': {2020: 0.0, 2021: 40.0, 2022: 80.0, 2023: 120.0},
            'cash_history': {2020: 60.0, 2021: 40.0, 2022: 20.0, 2023: 0.0},
        })
        # net debt −60 → +120 over 3 years → 180/(3·1000) = +0.06/yr
        assert m['net_debt_slope_3y'] == pytest.approx(0.06)

    def test_net_debt_slope_thin_none(self):
        from scripts.analyze_stock import derive_edgar_metrics
        m = derive_edgar_metrics({
            'revenue_history': {2022: 1000.0, 2023: 1000.0},
            'total_debt_history': {2022: 100.0, 2023: 150.0},
            'cash_history': {2022: 50.0, 2023: 50.0},
        })
        assert m['net_debt_slope_3y'] is None

    def test_all_detectors_none_on_empty(self):
        from scripts.analyze_stock import derive_edgar_metrics
        m = derive_edgar_metrics({})
        for k in ('rev_down_years', 'net_debt_slope_3y',
                  'div_fcf_ratio_3y', 'fcf_neg_years_5y'):
            assert m[k] is None


class TestSharesCagrFallback:
    """Ownership: Share Shrink from weighted-average counts when the
    period-end share series is too short (AOS: period-end shares tagged for
    2014-2015 only against 14 years of weighted-average counts)."""

    def _hist(self, **series):
        from scripts.analyze_stock import derive_edgar_metrics
        base = {'revenue_history': {y: 100.0 for y in range(2015, 2026)}}
        base.update(series)
        return derive_edgar_metrics(base)

    def test_period_end_series_preferred_when_long_enough(self):
        out = self._hist(
            shares_history={f'{y}-12-31': 100.0 - (y - 2019) for y in range(2019, 2026)},
            wavg_basic_history={y: 500.0 for y in range(2015, 2026)})
        assert out['shares_cagr_5y'] == pytest.approx((94.0 / 99.0) ** 0.2 - 1)

    def test_sparse_period_end_falls_back_to_weighted_average(self):
        out = self._hist(
            shares_history={'2014-12-31': 100.0, '2015-12-31': 99.0},
            wavg_basic_history={y: 200.0 * (0.98 ** (y - 2015)) for y in range(2015, 2026)})
        assert out['shares_cagr_5y'] == pytest.approx(0.98 - 1)

    def test_diluted_used_when_basic_missing(self):
        out = self._hist(
            wavg_diluted_history={y: 300.0 * (1.03 ** (y - 2015)) for y in range(2015, 2026)})
        assert out['shares_cagr_5y'] == pytest.approx(0.03)

    def test_hole_at_year_five_stays_none(self):
        """Year-keyed: a gap in the filing history must not stretch the
        window over more than five years."""
        w = {y: 200.0 for y in range(2015, 2026)}
        del w[2020]
        out = self._hist(wavg_basic_history=w)
        assert out['shares_cagr_5y'] is None

    def test_no_series_at_all_stays_none(self):
        assert self._hist()['shares_cagr_5y'] is None
