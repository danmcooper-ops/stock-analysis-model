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


class TestFallbackTags:
    """Tags appended as near-substitutes (InterestPaidNet for interest expense,
    the allocated/stock-issued SBC tags, net/software capex) fill a year only
    when no primary tag covers it — merging is latest-filed-wins across tags,
    so without the guard a later 10-K's comparative could displace a primary."""

    @staticmethod
    def _entries(years_values, filed_year_offset=1, fp='FY'):
        return [{'form': '10-K', 'fy': fy, 'fp': fp, 'val': val,
                 'filed': f'{fy + filed_year_offset}-02-15',
                 'start': f'{fy}-01-01', 'end': f'{fy}-12-31'}
                for fy, val in years_values.items()]

    def _client(self, monkeypatch, tags):
        from data.sec_xbrl_client import SECXBRLClient
        c = SECXBRLClient(cik_map={'TEST': '0000000001'}, name_map={},
                          email='t@e.com', request_delay=0)
        facts = {'facts': {'us-gaap': {
            'Revenues': {'units': {'USD': self._entries({2022: 1e9, 2023: 1.1e9, 2024: 1.2e9})}},
            'OperatingIncomeLoss': {'units': {'USD': self._entries(
                {2022: 100e6, 2023: 120e6, 2024: 150e6})}},
            **{t: {'units': {'USD': e}} for t, e in tags.items()},
        }}}
        monkeypatch.setattr(c, 'fetch_company_facts', lambda tk: facts)
        return c

    def test_interest_paid_only_resolves_history_and_int_cov(self, monkeypatch):
        from scripts.analyze_stock import derive_edgar_metrics
        c = self._client(monkeypatch, {
            'InterestPaidNet': self._entries({2022: 10e6, 2023: 12e6, 2024: 15e6})})
        h = c.fetch_historical_financials('TEST')
        assert h['interest_expense_history'] == {2022: 10e6, 2023: 12e6, 2024: 15e6}
        assert derive_edgar_metrics(h)['int_cov_edgar'] == pytest.approx(10.0)

    def test_expense_tag_wins_where_both_exist(self, monkeypatch):
        c = self._client(monkeypatch, {
            'InterestExpense': self._entries({2022: 20e6, 2023: 22e6}),
            # a later filing's cash-paid comparative for 2022 must not win
            'InterestPaidNet': self._entries({2022: 9e6, 2023: 11e6, 2024: 13e6},
                                             filed_year_offset=3),
        })
        h = c.fetch_historical_financials('TEST')
        # 2024 has no expense tag, so the fallback fills that year only
        assert h['interest_expense_history'] == {2022: 20e6, 2023: 22e6, 2024: 13e6}

    def test_sbc_fallback_order(self, monkeypatch):
        c = self._client(monkeypatch, {
            'AllocatedShareBasedCompensationExpense': self._entries({2023: 5e6}),
            'StockIssuedDuringPeriodValueShareBasedCompensation': self._entries(
                {2023: 99e6, 2024: 6e6}),
        })
        h = c.fetch_historical_financials('TEST')
        assert h['sbc_cf_history'] == {2023: 5e6, 2024: 6e6}
        c2 = self._client(monkeypatch, {
            'ShareBasedCompensation': self._entries({2023: 7e6}),
            'AllocatedShareBasedCompensationExpense': self._entries({2023: 5e6}),
        })
        assert c2.fetch_historical_financials('TEST')['sbc_cf_history'] == {2023: 7e6}

    def test_capex_fallbacks(self, monkeypatch):
        c = self._client(monkeypatch, {
            'PaymentsToAcquirePropertyPlantAndEquipment': self._entries({2024: 40e6}),
            'PaymentsToDevelopSoftware': self._entries({2023: 8e6, 2024: 9e6}),
            'PaymentsForProceedsFromProductiveAssets': self._entries({2022: 30e6}),
        })
        h = c.fetch_historical_financials('TEST')
        assert h['capex_history'] == {2022: 30e6, 2023: 8e6, 2024: 40e6}

    def test_periodic_uses_fallback_only_without_any_primary(self):
        from data.sec_xbrl_client import SECXBRLClient
        c = SECXBRLClient(cik_map={}, name_map={}, email='t@e.com', request_delay=0)
        tags = SECXBRLClient._XBRL_TAG_MAP['interest_expense']
        only_paid = {'facts': {'us-gaap': {
            'InterestPaidNet': {'units': {'USD': self._entries({2023: 12e6})}}}}}
        assert c._extract_periodic_values(only_paid, tags) == {'2023-12-31': 12e6}
        both = {'facts': {'us-gaap': {
            'InterestExpense': {'units': {'USD': self._entries({2023: 20e6})}},
            'InterestPaidNet': {'units': {'USD': self._entries({2023: 12e6, 2024: 13e6},
                                                               filed_year_offset=2)}}}}}
        assert c._extract_periodic_values(both, tags) == {'2023-12-31': 20e6}

    def test_fallback_tags_sit_at_the_tail_of_their_lists(self):
        from data.sec_xbrl_client import SECXBRLClient
        fb = SECXBRLClient._FALLBACK_TAGS
        seen = set()
        for tags in SECXBRLClient._XBRL_TAG_MAP.values():
            flags = [t in fb for t in tags]
            if any(flags):
                first = flags.index(True)
                assert all(flags[first:]), tags
                seen |= {t for t in tags if t in fb}
            assert len(tags) == len(set(tags)), f'duplicate tag in {tags}'
        assert seen == fb

    def test_mislabelled_fallback_period_cannot_displace_primary_label(self):
        """Shopify: a 2023 InterestPaidNet fact filed with fy=2022 shares the
        2022 label with the expense tag's 2022 period."""
        from data.sec_xbrl_client import SECXBRLClient
        c = SECXBRLClient(cik_map={}, name_map={}, email='t@e.com', request_delay=0)
        facts = {'facts': {'us-gaap': {
            'InterestExpense': {'units': {'USD': self._entries({2022: 3_499_000})}},
            'InterestPaidNet': {'units': {'USD': [
                {'form': '40-F', 'fy': 2022, 'fp': 'FY', 'val': 1_000_000,
                 'filed': '2024-02-13', 'start': '2023-01-01', 'end': '2023-12-31'}]}},
        }}}
        tags = SECXBRLClient._XBRL_TAG_MAP['interest_expense']
        assert c._extract_annual_values(facts, tags) == {2022: 3_499_000}

    def test_stale_fallback_series_is_dropped(self, monkeypatch):
        """SO: AllocatedShareBasedCompensationExpense last tagged 2016 must not
        become the current SBC; a current fallback series keeps its history."""
        c = self._client(monkeypatch, {
            'AllocatedShareBasedCompensationExpense': self._entries({2015: 2e6, 2016: 3e6}),
            'InterestPaidNet': self._entries({2016: 8e6, 2023: 9e6}),
        })
        h = c.fetch_historical_financials('TEST')        # revenue runs to 2024
        assert h['sbc_cf_history'] == {}
        assert h['interest_expense_history'] == {2016: 8e6, 2023: 9e6}

    def test_stale_rule_never_touches_primary_tags(self, monkeypatch):
        c = self._client(monkeypatch, {
            'ShareBasedCompensation': self._entries({2016: 3e6}),
            'AllocatedShareBasedCompensationExpense': self._entries({2015: 2e6}),
        })
        # the primary 2016 stays; the stale fallback-only 2015 goes
        assert c.fetch_historical_financials('TEST')['sbc_cf_history'] == {2016: 3e6}
        facts = c.fetch_company_facts('TEST')
        from data.sec_xbrl_client import SECXBRLClient
        tags = SECXBRLClient._XBRL_TAG_MAP['sbc_cf']
        assert c._extract_annual_values(facts, tags, fallback_min_year=2023) == {2016: 3e6}
        assert c._extract_annual_values(facts, tags) == {2015: 2e6, 2016: 3e6}

    def test_enrich_sbc_uses_only_current_fallbacks(self):
        from data.sec_xbrl_client import SECXBRLClient
        from scripts.enrich_xbrl import _compute_one
        c = SECXBRLClient(cik_map={}, name_map={}, email='t@e.com', request_delay=0)
        rec = {'edgar_history': {'revenue_history': {'2023': 1e9, '2024': 1.2e9}}}
        stale = {'facts': {'us-gaap': {'AllocatedShareBasedCompensationExpense': {
            'units': {'USD': self._entries({2016: 3e6})}}}}}
        _compute_one(rec, stale, c)
        assert rec.get('sbc_pct_rev_xbrl') is None
        current = {'facts': {'us-gaap': {'AllocatedShareBasedCompensationExpense': {
            'units': {'USD': self._entries({2024: 12e6})}}}}}
        _compute_one(rec, current, c)
        assert rec['sbc_pct_rev_xbrl'] == pytest.approx(0.01)

    def test_negative_fallback_values_are_ignored(self, monkeypatch):
        c = self._client(monkeypatch, {
            'StockIssuedDuringPeriodValueShareBasedCompensation': self._entries(
                {2023: 4e6, 2024: -90e6}),
            'PaymentsForProceedsFromProductiveAssets': self._entries({2024: -5e6}),
        })
        h = c.fetch_historical_financials('TEST')
        assert h['sbc_cf_history'] == {2023: 4e6}
        assert h['capex_history'] == {}


class TestYearsAvailable:
    """years_available counts fiscal years; points_available keeps the old
    longest-series point count."""

    def test_quarterly_share_points_do_not_count_as_years(self, monkeypatch):
        from data.sec_xbrl_client import SECXBRLClient
        c = SECXBRLClient(cik_map={}, name_map={}, email='t@e.com', request_delay=0)
        rev = [{'form': '10-K', 'fy': y, 'fp': 'FY', 'val': 1e9, 'filed': f'{y + 1}-02-15',
                'start': f'{y}-01-01', 'end': f'{y}-12-31'} for y in (2023, 2024, 2025)]
        shares = [{'form': '10-Q' if q < 4 else '10-K', 'fy': y, 'fp': f'Q{q}' if q < 4 else 'FY',
                   'val': 1e8, 'filed': f'{y}-{3 * q:02d}-28', 'end': f'{y}-{3 * q:02d}-28'}
                  for y in (2023, 2024, 2025) for q in (1, 2, 3, 4)]
        facts = {'facts': {'us-gaap': {'Revenues': {'units': {'USD': rev}}},
                           'dei': {'EntityCommonStockSharesOutstanding': {'units': {'shares': shares}}}}}
        monkeypatch.setattr(c, 'fetch_company_facts', lambda tk: facts)
        h = c.fetch_historical_financials('TEST')
        assert len(h['shares_history']) == 12
        assert h['years_available'] == 3
        assert h['points_available'] == 12

    def test_helper(self):
        from data.sec_xbrl_client import edgar_years_available
        assert edgar_years_available(None) == 0
        assert edgar_years_available({}) == 0
        assert edgar_years_available({'revenue_history': {'2023': 1, '2024': 2}}) == 2
        assert edgar_years_available({'revenue_history': {2023: 1, 2024: 2, 2025: 3}}) == 3
        # longest series by distinct year, not revenue alone; quarterly points
        # collapse to their years
        assert edgar_years_available({'revenue_history': {'2021': 1},
                                      'earnings_history': {2020: 1, 2021: 1},
                                      'operating_cf_history': {'2019': 1, '2020': 1, '2021': 1},
                                      'shares_history': {f'2025-{m:02d}-28': 1 for m in range(1, 13)}}) == 3
        assert edgar_years_available({'shares_history': {
            f'{y}-{m:02d}-28': 1 for y in (2024, 2025) for m in (3, 6, 9, 12)}}) == 2
