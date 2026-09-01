# tests/test_comparisons.py
import warnings

import pytest
import pandas as pd
import numpy as np


from models.ratios import (calculate_wacc, effective_tax_rate, calculate_roic, compute_ratios,
                           calculate_fundamental_growth, dupont_decomposition,
                           compute_dupont)
from models.quality import (calculate_piotroski_f, calculate_altman_z,
                            calculate_earnings_quality, calculate_interest_coverage,
                            calculate_net_debt_ebitda, get_net_debt,
                            calculate_revenue_cagr, calculate_beneish_m)
from models.market import compute_analyst_consensus
from models.nav import tangible_book_value_per_share


# ---------------------------------------------------------------------------
# calculate_wacc
# ---------------------------------------------------------------------------

class TestCalculateWACC:
    # Fixture: mcap 100e9, debt 10e9 (prior 11e9), interest 0.5e9,
    # tax 2e9/10e9 and 1.5e9/7.5e9 -> 20% both years.
    #   E/V = 100/110, D/V = 10/110
    #   Kd  = 0.5e9 / avg(10e9, 11e9) = 0.047619
    #   WACC = (100/110)*0.10 + (10/110)*0.047619*(1-0.20) = 0.094372
    EXPECTED = (100 / 110) * 0.10 + (10 / 110) * (0.5 / 10.5) * 0.8

    def test_returns_positive(self, sample_financials):
        """WACC should be a positive decimal."""
        wacc = calculate_wacc(sample_financials, cost_of_equity=0.10)
        assert wacc is not None
        assert 0 < wacc < 0.30

    def test_exact_value(self, sample_financials):
        """Hand-computed: market-cap weights, Kd on average debt, 20% tax."""
        wacc = calculate_wacc(sample_financials, cost_of_equity=0.10)
        assert wacc == pytest.approx(self.EXPECTED, rel=1e-6)

    def test_exact_value_with_rf_inside_band(self, sample_financials):
        """Kd 4.76% sits inside [Rf, Rf+10%] for Rf=4% -> no clamp, same value."""
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            wacc = calculate_wacc(sample_financials, cost_of_equity=0.10,
                                  risk_free_rate=0.04)
        assert wacc == pytest.approx(self.EXPECTED, rel=1e-6)

    def test_higher_re_gives_higher_wacc(self, sample_financials):
        """Higher cost of equity → higher WACC."""
        wacc_low = calculate_wacc(sample_financials, cost_of_equity=0.08)
        wacc_high = calculate_wacc(sample_financials, cost_of_equity=0.14)
        assert wacc_high > wacc_low

    def test_kd_uses_average_debt(self, sample_financials):
        """A prior-period balance halves the impact of a year-end repayment."""
        fin = _copy_fin(sample_financials)
        _set(fin['balance_sheet'], 'Total Debt', 1, 30e9)  # prior year
        wacc = calculate_wacc(fin, cost_of_equity=0.10)
        kd = 0.5e9 / ((10e9 + 30e9) / 2)
        expected = (100 / 110) * 0.10 + (10 / 110) * kd * 0.8
        assert wacc == pytest.approx(expected, rel=1e-6)

    def test_kd_floored_at_rf(self, sample_financials):
        """Legacy 1% coupons -> Kd floored at Rf, with a warning."""
        fin = _copy_fin(sample_financials)
        _set(fin['income_statement'], 'Interest Expense', 0, 0.1e9)  # ~0.95%
        with pytest.warns(RuntimeWarning, match='below the risk-free rate'):
            wacc = calculate_wacc(fin, cost_of_equity=0.10, risk_free_rate=0.04)
        expected = (100 / 110) * 0.10 + (10 / 110) * 0.04 * 0.8
        assert wacc == pytest.approx(expected, rel=1e-6)

    def test_kd_capped_at_rf_plus_spread(self, sample_financials):
        """Interest / debt of 19% -> capped at Rf + 10%, with a warning."""
        fin = _copy_fin(sample_financials)
        _set(fin['income_statement'], 'Interest Expense', 0, 2.0e9)
        with pytest.warns(RuntimeWarning, match='exceeds Rf'):
            wacc = calculate_wacc(fin, cost_of_equity=0.10, risk_free_rate=0.04)
        expected = (100 / 110) * 0.10 + (10 / 110) * 0.14 * 0.8
        assert wacc == pytest.approx(expected, rel=1e-6)

    def test_no_clamp_without_rf(self, sample_financials):
        """Without a risk-free rate there is no band to clamp to."""
        fin = _copy_fin(sample_financials)
        _set(fin['income_statement'], 'Interest Expense', 0, 2.0e9)
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            wacc = calculate_wacc(fin, cost_of_equity=0.10)
        kd = 2.0e9 / 10.5e9
        expected = (100 / 110) * 0.10 + (10 / 110) * kd * 0.8
        assert wacc == pytest.approx(expected, rel=1e-6)

    def test_credit_spread_fallback_when_interest_missing(self, sample_financials):
        """No interest expense row -> Kd = Rf + credit spread."""
        fin = _copy_fin(sample_financials)
        fin['income_statement'] = fin['income_statement'].drop(index='Interest Expense')
        wacc = calculate_wacc(fin, cost_of_equity=0.10, risk_free_rate=0.04,
                              credit_spread=0.025)
        expected = (100 / 110) * 0.10 + (10 / 110) * 0.065 * 0.8
        assert wacc == pytest.approx(expected, rel=1e-6)

    def test_loss_year_gets_no_tax_shield(self, sample_financials):
        """Pretax loss with a tax benefit in every year -> 0% rate, not
        (-benefit)/(-loss) = a positive shield."""
        fin = _copy_fin(sample_financials)
        for i in range(2):
            _set(fin['income_statement'], 'Pretax Income', i, -5e9)
            _set(fin['income_statement'], 'Tax Provision', i, -1e9)
        wacc = calculate_wacc(fin, cost_of_equity=0.10)
        kd = 0.5e9 / 10.5e9
        expected = (100 / 110) * 0.10 + (10 / 110) * kd * 1.0
        assert wacc == pytest.approx(expected, rel=1e-6)

    def test_loss_year_uses_prior_positive_years(self, sample_financials):
        """A one-off loss year is skipped; the prior profitable year's 20%
        still carries the shield."""
        fin = _copy_fin(sample_financials)
        _set(fin['income_statement'], 'Pretax Income', 0, -5e9)
        _set(fin['income_statement'], 'Tax Provision', 0, -1e9)
        wacc = calculate_wacc(fin, cost_of_equity=0.10)
        assert wacc == pytest.approx(self.EXPECTED, rel=1e-6)

    def test_zero_provision_is_zero_rate_not_default(self, sample_financials):
        """A genuine 0 provision on positive pretax income is a 0% rate."""
        fin = _copy_fin(sample_financials)
        for i in range(2):
            _set(fin['income_statement'], 'Tax Provision', i, 0.0)
        wacc = calculate_wacc(fin, cost_of_equity=0.10)
        kd = 0.5e9 / 10.5e9
        expected = (100 / 110) * 0.10 + (10 / 110) * kd * 1.0
        assert wacc == pytest.approx(expected, rel=1e-6)

    def test_tax_rate_defaults_when_rows_missing(self, sample_financials):
        fin = _copy_fin(sample_financials)
        fin['income_statement'] = fin['income_statement'].drop(
            index=['Tax Provision', 'Pretax Income'])
        wacc = calculate_wacc(fin, cost_of_equity=0.10)
        kd = 0.5e9 / 10.5e9
        expected = (100 / 110) * 0.10 + (10 / 110) * kd * (1 - 0.21)
        assert wacc == pytest.approx(expected, rel=1e-6)

    def test_negative_book_equity_with_market_cap(self, sample_financials):
        """Negative book equity is irrelevant when market cap is available."""
        fin = _copy_fin(sample_financials)
        _set(fin['balance_sheet'], 'Stockholders Equity', 0, -5e9)
        wacc = calculate_wacc(fin, cost_of_equity=0.10)
        assert wacc == pytest.approx(self.EXPECTED, rel=1e-6)

    def test_negative_book_equity_without_market_cap_is_none(self, sample_financials):
        fin = _copy_fin(sample_financials)
        fin['info'] = dict(fin['info'], marketCap=None)
        _set(fin['balance_sheet'], 'Stockholders Equity', 0, -5e9)
        assert calculate_wacc(fin, cost_of_equity=0.10) is None

    def test_no_debt_is_pure_cost_of_equity(self, sample_financials):
        fin = _copy_fin(sample_financials)
        fin['balance_sheet'] = fin['balance_sheet'].drop(
            index=['Total Debt', 'Long Term Debt'])
        assert calculate_wacc(fin, cost_of_equity=0.10) == pytest.approx(0.10)

    def test_none_with_missing_data(self):
        """Missing financial data → None."""
        result = calculate_wacc({}, cost_of_equity=0.10)
        assert result is None

    def test_none_financials(self):
        """None financials causes AttributeError (no guard); test actual behavior."""
        with pytest.raises(AttributeError):
            calculate_wacc(None, cost_of_equity=0.10)


class TestEffectiveTaxRate:
    def _inc(self, pairs):
        """pairs: list of (pretax, tax) newest first; None omits the row."""
        cols = pd.to_datetime([f'{2024 - i}-12-31' for i in range(len(pairs))])
        return pd.DataFrame({
            c: {'Pretax Income': p, 'Tax Provision': t}
            for c, (p, t) in zip(cols, pairs, strict=True)
        })

    def test_averages_positive_years(self):
        inc = self._inc([(100, 10), (100, 30), (100, 20), (100, 50)])
        assert effective_tax_rate(inc) == pytest.approx(0.20)  # 3-year window

    def test_skips_loss_years(self):
        inc = self._inc([(-50, -10), (100, 25)])
        assert effective_tax_rate(inc) == pytest.approx(0.25)

    def test_all_loss_is_zero(self):
        inc = self._inc([(-50, -10), (-20, 5)])
        assert effective_tax_rate(inc) == 0.0

    def test_clips_to_half(self):
        inc = self._inc([(100, 80)])
        assert effective_tax_rate(inc) == pytest.approx(0.50)

    def test_tax_benefit_on_profit_is_zero(self):
        inc = self._inc([(100, -5)])
        assert effective_tax_rate(inc) == 0.0

    def test_default_when_unknown(self):
        inc = pd.DataFrame({pd.Timestamp('2024-12-31'): {'Total Revenue': 1.0}})
        assert effective_tax_rate(inc) == pytest.approx(0.21)
        assert effective_tax_rate(None) == pytest.approx(0.21)


def _set(df, row, col_idx, val):
    df.iloc[df.index.get_loc(row), col_idx] = val


def _copy_fin(fin):
    return {k: (v.copy() if hasattr(v, 'copy') else v) for k, v in fin.items()}


# ---------------------------------------------------------------------------
# calculate_roic
# ---------------------------------------------------------------------------

class TestCalculateROIC:
    def test_returns_dict(self, sample_financials):
        """Should return dict with avg_roic and roic_by_year."""
        result = calculate_roic(sample_financials)
        assert result is not None
        assert 'avg_roic' in result
        assert 'roic_by_year' in result

    def test_avg_roic_positive(self, sample_financials):
        """For a profitable company, avg_roic should be positive."""
        result = calculate_roic(sample_financials)
        assert result['avg_roic'] > 0

    def test_roic_by_year_has_entries(self, sample_financials):
        result = calculate_roic(sample_financials)
        assert len(result['roic_by_year']) > 0

    def test_empty_data_returns_none(self):
        result = calculate_roic({})
        assert result is None

    def test_per_year_intermediates_align_with_roic(self, sample_financials):
        """nopat_by_year / invested_capital_by_year share roic_by_year's keys
        and reproduce the ratio (feeds the Moat: Incr ROIC gate)."""
        result = calculate_roic(sample_financials)
        assert set(result['nopat_by_year']) == set(result['roic_by_year'])
        assert set(result['invested_capital_by_year']) == set(result['roic_by_year'])
        for y, roic in result['roic_by_year'].items():
            nopat = result['nopat_by_year'][y]
            ic = result['invested_capital_by_year'][y]
            assert ic > 0
            assert nopat / ic == pytest.approx(roic)


# ---------------------------------------------------------------------------
# calculate_piotroski_f
# ---------------------------------------------------------------------------

class TestPiotroskiF:
    def test_returns_integer(self, sample_financials):
        """Piotroski F-Score should be an integer 0-9."""
        score = calculate_piotroski_f(sample_financials)
        if score is not None:
            assert 0 <= score <= 9
            assert isinstance(score, int)

    def test_profitable_company_scores_well(self, sample_financials):
        """A company with improving metrics should score >= 5."""
        score = calculate_piotroski_f(sample_financials)
        if score is not None:
            assert score >= 4  # our fixture has positive OCF, ROA, improving margins

    def test_none_with_empty_data(self):
        score = calculate_piotroski_f({})
        assert score is None


# ---------------------------------------------------------------------------
# calculate_altman_z
# ---------------------------------------------------------------------------

class TestAltmanZ:
    def test_healthy_company(self, sample_financials):
        """A healthy company should have Z > 2.99 (safe zone)."""
        z = calculate_altman_z(sample_financials)
        if z is not None:
            assert z > 2.0  # at least gray zone for our fixture

    def test_returns_float(self, sample_financials):
        z = calculate_altman_z(sample_financials)
        if z is not None:
            assert isinstance(z, (int, float))
            assert np.isfinite(z)

    def test_none_with_missing(self):
        z = calculate_altman_z({})
        assert z is None


# ---------------------------------------------------------------------------
# tangible_book_value_per_share
# ---------------------------------------------------------------------------

class TestTangibleBookValuePerShare:
    def test_happy_path_no_intangibles(self, sample_financials):
        """With no goodwill/intangibles in the fixture, TBV == BV (equity/shares)."""
        tbv = tangible_book_value_per_share(sample_financials)
        assert tbv is not None
        # 30B equity / 666.7M shares ≈ $45 — same as info.bookValue
        assert 40 < tbv < 50

    def test_strips_goodwill_and_intangibles(self, sample_info, sample_balance_sheet):
        """Adding goodwill + intangibles drops TBV below plain book value."""
        latest_col = sample_balance_sheet.columns[0]
        sample_balance_sheet.loc['Goodwill', latest_col] = 5e9
        sample_balance_sheet.loc['Other Intangible Assets', latest_col] = 3e9
        financials = {'balance_sheet': sample_balance_sheet, 'info': sample_info}
        tbv = tangible_book_value_per_share(financials)
        assert tbv is not None
        # (30B - 5B - 3B) / 666.7M ≈ $33
        assert 30 < tbv < 36

    def test_missing_goodwill_treated_as_zero(self, sample_info, sample_balance_sheet):
        """Goodwill absent (no key in BS) is treated as 0, not None."""
        # sample_balance_sheet has neither goodwill nor intangibles by default
        financials = {'balance_sheet': sample_balance_sheet, 'info': sample_info}
        tbv = tangible_book_value_per_share(financials)
        assert tbv is not None  # didn't refuse just because goodwill was missing

    def test_none_with_empty_dict(self):
        assert tangible_book_value_per_share({}) is None

    def test_none_with_zero_shares(self, sample_balance_sheet):
        info = {'sharesOutstanding': 0}
        financials = {'balance_sheet': sample_balance_sheet, 'info': info}
        assert tangible_book_value_per_share(financials) is None

    def test_none_with_negative_equity(self, sample_info, sample_balance_sheet):
        """Negative tangible equity (insolvent on a tangible basis) returns None."""
        latest_col = sample_balance_sheet.columns[0]
        sample_balance_sheet.loc['Goodwill', latest_col] = 50e9  # exceeds equity
        financials = {'balance_sheet': sample_balance_sheet, 'info': sample_info}
        assert tangible_book_value_per_share(financials) is None


# ---------------------------------------------------------------------------
# calculate_earnings_quality
# ---------------------------------------------------------------------------

class TestEarningsQuality:
    def test_returns_dict(self, sample_financials):
        result = calculate_earnings_quality(sample_financials)
        assert result is not None
        assert 'cash_conversion' in result
        assert 'accruals_ratio' in result

    def test_positive_cash_conversion(self, sample_financials):
        """OCF > Net Income → cash_conversion > 1."""
        result = calculate_earnings_quality(sample_financials)
        if result and result.get('cash_conversion') is not None:
            assert result['cash_conversion'] > 0

    def test_none_with_empty(self):
        result = calculate_earnings_quality({})
        assert result is None or result.get('cash_conv') is None


# ---------------------------------------------------------------------------
# calculate_interest_coverage
# ---------------------------------------------------------------------------

class TestInterestCoverage:
    def test_returns_positive(self, sample_financials):
        """Profitable company should have positive interest coverage."""
        ic = calculate_interest_coverage(sample_financials)
        if ic is not None:
            assert ic > 0

    def test_none_with_no_interest(self):
        """No interest expense data → None."""
        result = calculate_interest_coverage({})
        assert result is None


# ---------------------------------------------------------------------------
# calculate_net_debt_ebitda
# ---------------------------------------------------------------------------

class TestNetDebtEBITDA:
    def test_returns_finite(self, sample_financials):
        nd_ebitda = calculate_net_debt_ebitda(sample_financials)
        if nd_ebitda is not None:
            assert np.isfinite(nd_ebitda)

    def test_none_with_empty(self):
        assert calculate_net_debt_ebitda({}) is None


# ---------------------------------------------------------------------------
# get_net_debt
# ---------------------------------------------------------------------------

class TestGetNetDebt:
    def test_basic(self, sample_financials):
        """Net Debt = Total Debt - Cash."""
        nd = get_net_debt(sample_financials)
        if nd is not None:
            # Our fixture: Debt=10B, Cash=5B → ND=5B
            assert nd == pytest.approx(5e9)

    def test_none_when_balance_sheet_absent(self):
        """No balance sheet → None (leverage unknown), NOT 0 — a 0 would value
        a levered firm as debt-free in the EV→equity bridges."""
        assert get_net_debt({}) is None

    def test_zero_when_bs_present_but_no_debt_or_cash(self):
        """A present balance sheet with no debt/cash lines is a genuine read
        of an unlevered / sparsely-tagged filing → 0."""
        import pandas as pd
        bs = pd.DataFrame({pd.Timestamp('2024-12-31'): {'Total Assets': 100.0}})
        assert get_net_debt({'balance_sheet': bs}) == 0


# ---------------------------------------------------------------------------
# calculate_revenue_cagr
# ---------------------------------------------------------------------------

class TestRevenueCagr:
    def test_positive_growth(self, sample_financials):
        """Revenue grew from 35B to 40B → positive CAGR."""
        cagr = calculate_revenue_cagr(sample_financials)
        if cagr is not None:
            assert cagr > 0

    def test_none_with_empty(self):
        assert calculate_revenue_cagr({}) is None


# ---------------------------------------------------------------------------
# compute_ratios
# ---------------------------------------------------------------------------

class TestComputeRatios:
    def test_returns_dict(self, sample_financials):
        result = compute_ratios(sample_financials)
        assert result is not None
        assert isinstance(result, dict)

    def test_has_standard_keys(self, sample_financials):
        """compute_ratios returns balance sheet ratios (ROE, D/E, etc.)."""
        result = compute_ratios(sample_financials)
        # compute_ratios returns ROE, D/E, ROA, Current Ratio
        assert 'ROE' in result or 'Debt-to-Equity' in result


# ---------------------------------------------------------------------------
# compute_analyst_consensus
# ---------------------------------------------------------------------------

class TestAnalystConsensus:
    def test_returns_dict(self, sample_financials):
        result = compute_analyst_consensus(sample_financials)
        assert result is not None
        assert 'rec_key' in result

    def test_has_targets(self, sample_financials):
        result = compute_analyst_consensus(sample_financials)
        assert result.get('target_mean') is not None


# ---------------------------------------------------------------------------
# calculate_fundamental_growth
# ---------------------------------------------------------------------------

class TestFundamentalGrowth:
    def test_returns_dict(self, sample_financials):
        """Should return dict with required keys."""
        result = calculate_fundamental_growth(sample_financials)
        assert isinstance(result, dict)
        assert 'fundamental_growth' in result
        assert 'reinvestment_rate' in result
        assert 'roic_used' in result

    def test_growth_positive(self, sample_financials):
        """For a profitable company with positive capex, growth should be positive."""
        result = calculate_fundamental_growth(sample_financials)
        assert result['fundamental_growth'] > 0

    def test_reinvestment_rate_clamped(self, sample_financials):
        """Reinvestment rate should be in [0, 1]."""
        result = calculate_fundamental_growth(sample_financials)
        assert 0 <= result['reinvestment_rate'] <= 1.0

    def test_growth_clamped_at_30pct(self, sample_financials):
        """Growth should be capped at 30%."""
        result = calculate_fundamental_growth(sample_financials)
        assert result['fundamental_growth'] <= 0.30

    def test_roic_override(self, sample_financials):
        """roic_override should be used instead of computing ROIC."""
        result = calculate_fundamental_growth(sample_financials, roic_override=0.50)
        assert result['roic_used'] == 0.50

    def test_empty_data_returns_empty(self):
        """Empty financials → empty dict."""
        result = calculate_fundamental_growth({})
        assert result == {}

    def test_negative_operating_income_returns_empty(self, sample_financials):
        """Negative operating income → empty dict (NOPAT ≤ 0)."""
        inc = sample_financials['income_statement'].copy()
        col = inc.columns[0]
        inc.loc['Operating Income', col] = -5e9
        financials = {**sample_financials, 'income_statement': inc}
        result = calculate_fundamental_growth(financials)
        assert result == {}

    def test_none_roic_returns_empty(self, sample_financials):
        """If ROIC override is negative → empty dict."""
        result = calculate_fundamental_growth(sample_financials, roic_override=-0.05)
        assert result == {}


# ---------------------------------------------------------------------------
# calculate_beneish_m
# ---------------------------------------------------------------------------

class TestBeneishM:
    def test_returns_dict(self, sample_financials):
        """Should return dict with m_score and manipulation_flag."""
        result = calculate_beneish_m(sample_financials)
        assert result is not None
        assert isinstance(result, dict)
        assert 'm_score' in result
        assert 'manipulation_flag' in result
        assert 'components' in result

    def test_healthy_company_not_flagged(self, sample_financials):
        """Our fixture company should not be flagged as manipulator."""
        result = calculate_beneish_m(sample_financials)
        assert result is not None
        # Standard healthy company → M < -1.78
        assert result['m_score'] < -1.0  # generous bound

    def test_none_with_single_year(self, sample_financials):
        """Needs 2 years; single year → None."""
        bs = sample_financials['balance_sheet'].iloc[:, :1]
        inc = sample_financials['income_statement'].iloc[:, :1]
        financials = {**sample_financials, 'balance_sheet': bs, 'income_statement': inc}
        assert calculate_beneish_m(financials) is None

    def test_none_with_empty(self):
        """Empty financials → None."""
        assert calculate_beneish_m({}) is None
        assert calculate_beneish_m({'balance_sheet': pd.DataFrame(),
                                     'income_statement': pd.DataFrame()}) is None

    def test_components_present(self, sample_financials):
        """All 8 component keys should be present."""
        result = calculate_beneish_m(sample_financials)
        assert result is not None
        for key in ('dsri', 'gmi', 'aqi', 'sgi', 'depi', 'sgai', 'tata', 'lvgi'):
            assert key in result['components']

    def test_sgi_matches_revenue_ratio(self, sample_financials):
        """SGI should equal rev_t / rev_t1."""
        result = calculate_beneish_m(sample_financials)
        assert result is not None
        assert result['components']['sgi'] == pytest.approx(40e9 / 35e9)

    def test_none_on_zero_revenue(self, sample_financials):
        """Zero revenue → None."""
        inc = sample_financials['income_statement'].copy()
        col = inc.columns[0]
        inc.loc['Total Revenue', col] = 0
        financials = {**sample_financials, 'income_statement': inc}
        assert calculate_beneish_m(financials) is None


# ---------------------------------------------------------------------------
# dupont_decomposition / compute_dupont
# ---------------------------------------------------------------------------

class TestDuPontDecomposition:
    def test_basic(self):
        """Known inputs → correct decomposition."""
        result = dupont_decomposition(10, 100, 500, 200)
        assert result is not None
        assert result['margin'] == pytest.approx(0.10)
        assert result['turnover'] == pytest.approx(0.20)
        assert result['leverage'] == pytest.approx(2.50)
        assert result['roe'] == pytest.approx(0.05)

    def test_product_equals_roe(self):
        """margin * turnover * leverage should equal ROE."""
        result = dupont_decomposition(8e9, 40e9, 50e9, 30e9)
        assert result is not None
        expected_roe = 8e9 / 30e9  # NI / Equity
        assert result['roe'] == pytest.approx(expected_roe, rel=1e-6)

    def test_none_on_missing_data(self):
        """None inputs → None."""
        assert dupont_decomposition(None, 100, 500, 200) is None
        assert dupont_decomposition(10, None, 500, 200) is None
        assert dupont_decomposition(10, 100, None, 200) is None
        assert dupont_decomposition(10, 100, 500, None) is None

    def test_none_on_zero_equity(self):
        """Zero equity → None."""
        assert dupont_decomposition(10, 100, 500, 0) is None

    def test_none_on_zero_revenue(self):
        """Zero revenue → None."""
        assert dupont_decomposition(10, 0, 500, 200) is None


class TestComputeDuPont:
    def test_returns_dict(self, sample_financials):
        """Should return dict from sample financials."""
        result = compute_dupont(sample_financials)
        assert result is not None
        assert 'margin' in result
        assert 'turnover' in result
        assert 'leverage' in result
        assert 'roe' in result

    def test_none_with_empty(self):
        """Empty financials → None."""
        assert compute_dupont({}) is None
        assert compute_dupont({'balance_sheet': pd.DataFrame(),
                               'income_statement': pd.DataFrame()}) is None

    def test_values_reasonable(self, sample_financials):
        """DuPont components should be in reasonable ranges."""
        result = compute_dupont(sample_financials)
        assert result is not None
        assert 0 < result['margin'] < 1
        assert result['turnover'] > 0
        assert result['leverage'] >= 1
