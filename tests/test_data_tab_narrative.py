# tests/test_data_tab_narrative.py
"""Tests for the popup Data sub-tab summaries (models/data_tab_narrative.py)."""
import math
import re
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from models.data_tab_narrative import TAB_KEYS, MAX_SENTENCES, generate_data_tab_summaries


def _row(**over):
    """A healthy large-cap technology row that fills every tab."""
    row = {
        'ticker': 'TST', 'sector': 'Technology',
        # Sector
        'operating_margin': 0.30, '_sector_median_opm': 0.15,
        'pp_margin_advantage': 0.15, 'pp_multiple': 1.8,
        'pp_sector_hhi': 0.05, 'pp_sector_count': 290,
        # Market
        'momentum_12_1': 0.25, 'range_52w_position': 90, 'pct_from_52w_high': -0.03,
        'realized_vol': 0.25, 'beta_adjusted': 1.1, 'avg_dollar_volume_3m': 5e8,
        # Valuation
        'mos': 0.25, '_fv_effective': 130.0, '_fv_source': 'dcf',
        '_gate_fv_dispersion': 0.10, 'pe': 15.0, 'ev_ebitda': 10.0, 'pfcf': 12.0,
        'implied_growth': 0.04, 'implied_vs_estimated': -0.03,
        # Profitability
        'roic': 0.25, 'wacc': 0.09, 'spread': 0.16, 'roic_cv': 0.12,
        'gross_margin': 0.60, 'fcf_margin': 0.22, 'fcf_margin_ex_sbc': 0.20,
        'piotroski': 8, 'cash_conv': 1.1, 'accruals': 0.01,
        # Health
        'net_debt': 2e9, 'nd_ebitda': 0.8, 'int_cov': 20.0, 'cr': 1.6,
        'altman_z': 6.0, 'altman_z_zone': 'safe', 'beneish_flag': False,
        # Growth
        'rev_cagr': 0.15, 'rev_cagr_5y': 0.10, 'rev_cagr_10y': 0.08,
        'fundamental_growth': 0.08, 'reinvestment_rate': 0.35, 'margin_trend': 0.02,
        # Ownership
        'insider_pct': 0.08, 'inst_pct': 0.7, 'shareholder_yield': 0.03,
        'share_buyback_rate': 0.02, 'div_yield': 0.01, 'payout_ratio': 0.3,
        'ddm_consecutive_years': 12,
        # People
        'revenue_per_emp': 900_000, 'rpe_cagr': 0.06, 'ceo_total_pay': 2e7,
        'compensation_risk': 2,
    }
    row.update(over)
    return row


_STATS = {'pe': 25.0, 'ev_ebitda': 16.0, 'pfcf': 24.0, 'revenue_per_emp': 450_000}


def _text(summ, key):
    return ' '.join(summ[key])


def test_rich_row_fills_every_tab():
    summ = generate_data_tab_summaries(_row(), _STATS)
    assert set(summ) == set(TAB_KEYS)
    for key in TAB_KEYS:
        assert summ[key], key
        assert len(summ[key]) <= MAX_SENTENCES
        assert all(isinstance(s, str) and s.endswith('.') for s in summ[key])


def test_empty_row_says_nothing():
    summ = generate_data_tab_summaries({'ticker': 'X'})
    assert summ == {k: [] for k in TAB_KEYS}


def test_non_finite_and_text_inputs_are_skipped():
    row = {'ticker': 'X', 'mos': math.nan, 'pe': math.inf, 'roic': 'Infinity',
           'spread': None, 'insider_pct': True}
    assert generate_data_tab_summaries(row) == {k: [] for k in TAB_KEYS}


def test_mos_wording_flips_at_thresholds():
    wide = _text(generate_data_tab_summaries(_row(mos=0.16)), 'val')
    thin = _text(generate_data_tab_summaries(_row(mos=0.05)), 'val')
    over = _text(generate_data_tab_summaries(_row(mos=-0.05)), 'val')
    deep = _text(generate_data_tab_summaries(_row(mos=-0.25)), 'val')
    assert 'meaningful margin of safety' in wide
    assert 'thin margin of safety' in thin
    assert 'no valuation cushion' in over and 'PASS' not in over
    assert 'cap the rating at PASS' in deep


def test_fair_value_is_a_per_share_price():
    txt = _text(generate_data_tab_summaries(_row(_fv_effective=31.4)), 'val')
    assert '$31.40' in txt


def test_leverage_follows_gate_flag_not_recomputation():
    # The gate flag wins over the raw value, so the summary can't contradict
    # the scorecard even if a threshold drifts.
    passed = _text(generate_data_tab_summaries(_row(nd_ebitda=2.0, _gp_net_debt_ebitda=True)), 'hlth')
    failed = _text(generate_data_tab_summaries(_row(nd_ebitda=2.0, _gp_net_debt_ebitda=False)), 'hlth')
    assert 'conservative' in passed
    assert 'moderate' in failed
    heavy = _text(generate_data_tab_summaries(_row(nd_ebitda=4.0)), 'hlth')
    assert 'heavy' in heavy


def test_dispersion_follows_gate_flag():
    txt = _text(generate_data_tab_summaries(_row(_gate_fv_dispersion=0.10,
                                                 _gp_fv_dispersion=False)), 'val')
    assert 'disagree' in txt


def test_net_cash_reads_as_net_cash():
    txt = _text(generate_data_tab_summaries(_row(net_debt=-3e9, net_cash_to_mcap=0.06)), 'hlth')
    assert 'net cash of $3.0B (6% of market cap)' in txt


def test_distress_flags_name_the_cap():
    one = _text(generate_data_tab_summaries(_row(altman_z=1.1, altman_z_zone='distress')), 'hlth')
    both = _text(generate_data_tab_summaries(_row(altman_z=1.1, altman_z_zone='distress',
                                                  beneish_flag=True)), 'hlth')
    assert 'which caps the rating at HOLD' in one
    assert 'either one caps the rating at HOLD' in both


def test_financials_skip_industrial_health_metrics():
    row = _row(sector='Financial Services', altman_z=0.4, altman_z_zone='distress',
               cet1_ratio=0.15, npl_ratio=0.008, nim=0.029, efficiency_ratio=0.54)
    summ = generate_data_tab_summaries(row)
    hlth = _text(summ, 'hlth')
    assert 'Altman' not in hlth and 'net debt to EBITDA' not in hlth
    assert 'CET1 capital of 15.0%' in hlth
    assert 'net interest margin' in _text(summ, 'prof')
    assert 'Piotroski' not in _text(summ, 'prof')


def test_sector_comparisons_need_stats():
    without = _text(generate_data_tab_summaries(_row()), 'val')
    with_stats = _text(generate_data_tab_summaries(_row(), _STATS), 'val')
    assert 'peers' not in without
    assert 'looks cheap against Technology peers' in with_stats
    rich = _text(generate_data_tab_summaries(_row(pe=40.0, ev_ebitda=30.0, pfcf=40.0), _STATS), 'val')
    assert 'looks expensive' in rich


def test_implausible_values_are_not_repeated():
    txt = _text(generate_data_tab_summaries(_row(rev_cagr=11.08, rev_cagr_5y=None,
                                                 rev_cagr_10y=None, margin_trend=17.9)),
                'growth')
    assert '1108' not in txt and '1790' not in txt


def test_shrinking_revenue_reads_as_easing_decline():
    txt = _text(generate_data_tab_summaries(_row(rev_cagr=-0.04, rev_cagr_5y=-0.07)), 'growth')
    assert 'decline is easing' in txt


def test_institutional_over_100_pct_is_explained():
    txt = _text(generate_data_tab_summaries(_row(inst_pct=1.2)), 'own')
    assert '120%' not in txt and 'exceed 100%' in txt


def test_zero_shareholder_yield_is_not_called_dilution():
    txt = _text(generate_data_tab_summaries(_row(shareholder_yield=0.0)), 'own')
    assert 'dilut' not in txt


def test_thin_liquidity_mentions_buy_floor():
    txt = _text(generate_data_tab_summaries(_row(avg_dollar_volume_3m=400_000)), 'mkt')
    assert '$400K a day' in txt and 'BUY' in txt


def test_market_reversal_and_drawdowns():
    txt = _text(generate_data_tab_summaries(_row(momentum_12_1=0.5, momentum_3m=-0.16,
                                                 drawdown_2020=-0.31, drawdown_2022=-0.3)), 'mkt')
    assert 'pulled back 16% over the last three months' in txt
    assert 'fell 31% in the 2020 crash and 30% in the 2022 bear market' in txt


def test_mult_vs_hist_follows_gate():
    dear = _text(generate_data_tab_summaries(_row(_gate_mult_vs_hist=0.86,
                                                  _gp_mult_vs_hist=False)), 'val')
    cheap = _text(generate_data_tab_summaries(_row(_gate_mult_vs_hist=-0.26,
                                                   _gp_mult_vs_hist=True)), 'val')
    assert 'dear: the EBIT multiple sits 86% above' in dear
    assert 'cheap: the EBIT multiple sits 26% below' in cheap


def test_monte_carlo_range_rides_the_dispersion_sentence():
    txt = _text(generate_data_tab_summaries(_row(mc_p10_fv=102.4, mc_p90_fv=432.9)), 'val')
    assert 'Monte Carlo range runs $102 to $433 (P10-P90)' in txt


def test_street_target_mentions_upside():
    txt = _text(generate_data_tab_summaries(_row(price=100.0, target_mean=120.0,
                                                 num_analysts=12)), 'val')
    assert 'The 12 covering analysts target $120 on average, 20% above the price' in txt


def test_dupont_caveat_only_when_leverage_lifts_roe():
    lev = _text(generate_data_tab_summaries(_row(roe=0.6, dupont_leverage=4.5)), 'prof')
    weak = _text(generate_data_tab_summaries(_row(roe=0.01, dupont_leverage=4.5)), 'prof')
    bank = _text(generate_data_tab_summaries(_row(sector='Financial Services', roe=0.16,
                                                  dupont_leverage=12.0)), 'prof')
    assert 'amplified by 4.5x balance-sheet leverage' in lev
    assert 'amplified' not in weak
    assert 'return on equity' in bank and 'amplified' not in bank


def test_sbc_above_gate_is_flagged():
    txt = _text(generate_data_tab_summaries(_row(sbc_pct_rev_xbrl=0.031, _gp_sbc_dilution=False)),
                'prof')
    assert '3.1% of revenue on stock compensation (above the 2% gate)' in txt


def test_health_warnings_survive_the_sentence_cap():
    row = _row(altman_z=1.1, altman_z_zone='distress', trap_score=72, cr=0.8,
               working_capital_days=-60, debt_maturity_wall_yrs=1.5, goodwill_pct=0.5,
               beneish_m=-2.0, edgar_fields_flagged=2)
    hlth = generate_data_tab_summaries(row)['hlth']
    assert len(hlth) == MAX_SENTENCES
    txt = ' '.join(hlth)
    assert 'caps the rating at HOLD' in txt and 'value-trap profile is high' in txt
    assert '2 reported figures differ' in txt


def test_beneish_grey_zone():
    txt = _text(generate_data_tab_summaries(_row(beneish_m=-2.0)), 'hlth')
    assert 'just under the manipulation line' in txt
    clean = _text(generate_data_tab_summaries(_row(beneish_m=-2.6)), 'hlth')
    assert 'manipulation line' not in clean


def test_growth_forward_indicators_and_capex():
    txt = _text(generate_data_tab_summaries(_row(book_to_bill_proxy=1.22, backlog_to_revenue=0.45,
                                                 capex_to_dd_ratio=0.5, capex_intensity=0.02)),
                'growth')
    assert 'book-to-bill of 1.22 (orders outpacing sales)' in txt
    assert 'backlog worth 45% of annual revenue' in txt
    assert 'Capex runs only 0.5x depreciation (2.0% of sales)' in txt


def test_share_count_and_issuance_netting():
    shrink = _text(generate_data_tab_summaries(_row(_gate_share_shrink=-0.022)), 'own')
    assert 'shrunk 2.2% a year over five years' in shrink
    issue = _text(generate_data_tab_summaries(_row(shareholder_yield=0.045, div_yield=0.065,
                                                   share_buyback_rate=-0.02)), 'own')
    assert '6.5% in dividends, less 2.0% of net share issuance' in issue


def test_people_founder_and_negative_money():
    txt = _text(generate_data_tab_summaries(_row(founder_led=True, ceo='Mr. Ada  Founder',
                                                 fcf_per_emp=-65_000)), 'people')
    assert 'founder-led' in txt and 'Ada' not in txt
    assert '-$65K per employee' in txt


_VAL = st.one_of(st.none(), st.booleans(), st.text(max_size=4),
                 st.floats(allow_nan=True, allow_infinity=True),
                 st.integers(min_value=-10**12, max_value=10**12))
_KEYS = list(_row()) + ['cet1_ratio', 'npl_ratio', 'nim', 'efficiency_ratio',
                        'combined_ratio', 'affo_margin', 'trap_score', 'goodwill_pct',
                        'glassdoor_rating', 'rule_of_40', 'surprise_avg', 'short_pct_float',
                        'insider_buy_count_365d', 'insider_sell_count_365d',
                        'momentum_3m', 'drawdown_2020', 'drawdown_2022', 'drawdown_2008',
                        'price_data_stale', 'mc_p10_fv', 'mc_p90_fv', 'target_mean', 'price',
                        'num_analysts', '_gate_mult_vs_hist', '_gp_mult_vs_hist',
                        '_gate_fcf_yield', '_gp_fcf_yield', '_gate_ebit_ev', 'roe',
                        'dupont_leverage', 'rd_intensity_xbrl', 'sbc_pct_rev_xbrl',
                        'sga_yoy_change', '_gate_margin_vs_hist', '_gp_margin_vs_hist',
                        'working_capital_days', 'debt_maturity_wall_yrs', 'beneish_m',
                        'edgar_fields_flagged', 'cash', 'total_debt', 'book_to_bill_proxy',
                        'backlog_to_revenue', 'deferred_rev_growth', 'ffo_growth_5y',
                        'fda_pipeline_count', 'capex_to_dd_ratio', 'capex_intensity',
                        'analyst_ltg', '_gate_fcf_durability', '_gate_rev_volatility',
                        '_gate_share_shrink', 'dividend_cagr_5y', 'insider_net_value',
                        'employees', 'fcf_per_emp', 'sbc_per_emp', 'founder_led', 'ceo',
                        'glassdoor_rec_pct', 'glassdoor_ceo_pct', 'pp_revenue_share',
                        'pp_profit_share', '_gate_pool_share', 'pp_sector_cr4']


@settings(max_examples=200, deadline=None)
@given(st.dictionaries(st.sampled_from(_KEYS), _VAL),
       st.sampled_from([None, 'Technology', 'Financial Services', 'Real Estate']))
def test_never_raises_on_arbitrary_rows(row, sector):
    row['sector'] = sector
    summ = generate_data_tab_summaries(row, _STATS)
    assert set(summ) == set(TAB_KEYS)
    for v in summ.values():
        assert isinstance(v, list) and len(v) <= MAX_SENTENCES
        assert all(isinstance(s, str) and s for s in v)
        assert not any(re.search(r'\bnan\b|\binf\b', s, re.I) for s in v)


def test_template_renders_summary_escaped():
    src = (Path(__file__).resolve().parents[1] / 'templates' / 'report.html').read_text(
        encoding='utf-8')
    body = src[src.index('function _renderDetDataTab('):]
    body = body[:body.index('\nfunction ', 1)]
    assert 'd.data_summaries' in body
    assert 'esc(x)' in body
