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


_VAL = st.one_of(st.none(), st.booleans(), st.text(max_size=4),
                 st.floats(allow_nan=True, allow_infinity=True),
                 st.integers(min_value=-10**12, max_value=10**12))
_KEYS = list(_row()) + ['cet1_ratio', 'npl_ratio', 'nim', 'efficiency_ratio',
                        'combined_ratio', 'affo_margin', 'trap_score', 'goodwill_pct',
                        'glassdoor_rating', 'rule_of_40', 'surprise_avg', 'short_pct_float',
                        'insider_buy_count_365d', 'insider_sell_count_365d']


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
