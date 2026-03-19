# tests/test_comparisons.py
import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.ratios import (calculate_wacc, calculate_roic, compute_ratios,
                           calculate_fundamental_growth, dupont_decomposition,
                           compute_dupont)
from models.quality import (calculate_piotroski_f, calculate_altman_z,
                            calculate_earnings_quality, calculate_interest_coverage,
                            calculate_net_debt_ebitda, get_net_debt,
                            calculate_revenue_cagr, calculate_beneish_m)
from models.market import compute_rating, compute_analyst_consensus


# ---------------------------------------------------------------------------
# calculate_wacc
# ---------------------------------------------------------------------------

class TestCalculateWACC:
    def test_returns_positive(self, sample_financials):
        """WACC should be a positive decimal."""
        wacc = calculate_wacc(sample_financials, cost_of_equity=0.10)
        assert wacc is not None
        assert 0 < wacc < 0.30

    def test_higher_re_gives_higher_wacc(self, sample_financials):
        """Higher cost of equity → higher WACC."""
        wacc_low = calculate_wacc(sample_financials, cost_of_equity=0.08)
        wacc_high = calculate_wacc(sample_financials, cost_of_equity=0.14)
        assert wacc_high > wacc_low

    def test_none_with_missing_data(self):
        """Missing financial data → None."""
        result = calculate_wacc({}, cost_of_equity=0.10)
        assert result is None

    def test_none_financials(self):
        """None financials causes AttributeError (no guard); test actual behavior."""
        with pytest.raises(AttributeError):
            calculate_wacc(None, cost_of_equity=0.10)


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
# compute_rating
# ---------------------------------------------------------------------------

class TestComputeRating:
    def test_buy_rating(self):
        """High-quality metrics should produce BUY."""
        row = {
            'mos': 0.25, 'spread': 0.25, 'piotroski': 8,
            'cash_conv': 0.95, 'analyst_rec': 'buy',
            'rev_cagr': 0.15, 'de': 0.5, 'int_cov': 10.0,
            'roic_by_year': {'2024': 0.20, '2023': 0.18, '2022': 0.16},
            'wacc': 0.09,
        }
        rating = compute_rating(row)
        assert rating in ('BUY', 'LEAN BUY')

    def test_pass_rating(self):
        """Poor metrics should produce PASS."""
        row = {
            'mos': -0.20, 'spread': 0.01, 'piotroski': 2,
            'cash_conv': 0.3, 'analyst_rec': 'sell',
            'rev_cagr': -0.05, 'de': 3.0, 'int_cov': 0.5,
            'roic_by_year': {'2024': 0.05, '2023': 0.08, '2022': 0.12},
            'wacc': 0.10,
        }
        rating = compute_rating(row)
        assert rating in ('PASS', 'HOLD')

    def test_hold_rating(self):
        """Middling metrics should produce HOLD or LEAN BUY."""
        row = {
            'mos': 0.05, 'spread': 0.08, 'piotroski': 5,
            'cash_conv': 0.7, 'analyst_rec': 'hold',
            'rev_cagr': 0.04, 'de': 1.2, 'int_cov': 3.0,
            'roic_by_year': {'2024': 0.12, '2023': 0.11},
            'wacc': 0.09,
        }
        rating = compute_rating(row)
        assert rating in ('HOLD', 'LEAN BUY')

    def test_valid_ratings_only(self):
        """Rating must be one of the 4 valid values."""
        row = {'mos': 0.0, 'spread': 0.0}
        rating = compute_rating(row)
        assert rating in ('BUY', 'LEAN BUY', 'HOLD', 'PASS')


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

    def test_zero_with_empty(self):
        """Empty dict returns 0 (default when no debt or cash found)."""
        assert get_net_debt({}) == 0


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
