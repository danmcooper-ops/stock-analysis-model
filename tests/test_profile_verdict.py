# tests/test_profile_verdict.py
"""The Profile tab's invest / watch / avoid verdict (models/profile_verdict.py).

Pins the rule table (each red flag vetoes, each INVEST condition is
required, missing inputs are N/A rather than failures) and, by property
test, that the function never raises and never improves as the margin of
safety shrinks.
"""
import math

from hypothesis import given, settings
from hypothesis import strategies as st

from models.profile_verdict import (
    AVOID,
    INVEST,
    MIN_MOS,
    NO_DATA,
    VERDICTS,
    WATCH,
    profile_verdict,
)

_RANK = {AVOID: 0, NO_DATA: 0, WATCH: 1, INVEST: 2}


def _good(**over):
    """A BUY-rated, cheap, profitable, unlevered company: an INVEST."""
    row = {
        'ticker': 'GOOD', 'sector': 'Technology', 'price': 100.0,
        '_fv_effective': 130.0, '_fv_source': 'dcf', 'mos': 0.23,
        'rating': 'BUY', 'rating_raw': 'BUY', '_composite_score': 64.0,
        '_gates_passed': '9/12', 'roic': 0.25, 'wacc': 0.09, 'spread': 0.16,
        'fcf_margin': 0.22, 'rev_cagr_5y': 0.11, 'net_debt': -5e9,
        'piotroski': 7, 'altman_z_zone': 'safe', 'altman_z': 4.1,
        'beneish_m': -2.6, 'int_cov': 40.0, 'mc_confidence': 'MEDIUM (CV 28%)',
        'target_mean': 125.0, 'num_analysts': 30, 'pe': 20.0,
    }
    row.update(over)
    return row


def test_clean_buy_with_margin_of_safety_is_invest():
    v = profile_verdict(_good(), {'pe': 28.0})
    assert v['v'] == INVEST
    assert v['flags'] == [] and v['need'] == []
    assert v['conv'] >= 80
    assert v['head'].startswith('Invest: ')
    assert 'Model rating BUY (composite 64, gates 9/12)' in v['pros']
    assert v['buy_below'] == round(130.0 * (1 - MIN_MOS), 2)
    assert v['med'] == {'pe': 28.0}


def test_good_business_without_margin_of_safety_is_watch_with_a_buy_price():
    v = profile_verdict(_good(price=125.0, mos=0.04))
    assert v['v'] == WATCH
    assert 'only 4% below fair value' in v['need']
    assert v['head'] == ('Watch: BUY-rated, but only 4% below fair value; '
                         'it becomes a buy below $110.')


def test_hold_rating_is_watch_and_names_the_rating():
    v = profile_verdict(_good(rating='HOLD'))
    assert v['v'] == WATCH
    assert v['need'][0] == 'the model rates it HOLD'


def test_low_confidence_fair_value_blocks_invest():
    v = profile_verdict(_good(mc_confidence='LOW (CV 55%)'))
    assert v['v'] == WATCH
    assert 'wide Monte Carlo spread' in v['head']


def test_negative_spread_blocks_invest():
    v = profile_verdict(_good(spread=-0.02, roic=0.07))
    assert v['v'] == WATCH
    assert 'it earns less than its cost of capital' in v['need']


def test_each_red_flag_forces_avoid():
    cases = {
        'rating PASS': dict(rating='PASS'),
        'mos <= -20%': dict(mos=-0.25, price=162.5),
        'beneish': dict(beneish_m=-1.2),
        'beneish flag': dict(beneish_flag=True),
        'altman distress, unhealthy': dict(altman_z_zone='distress', fcf_margin=0.01),
        'value-destroying cash burner': dict(spread=-0.03, fcf_margin=-0.1),
        'interest barely covered': dict(int_cov=1.2, net_debt=2e9),
    }
    for name, over in cases.items():
        v = profile_verdict(_good(**over))
        assert v['v'] == AVOID, name
        assert v['flags'], name
        assert v['head'].startswith('Avoid: '), name


def test_avoid_headline_prefers_the_specific_flag_over_the_label():
    v = profile_verdict(_good(rating='PASS', mos=-0.636, price=338.56, _fv_effective=206.91))
    assert v['flags'] == ['Priced 64% above its fair value of $207', 'The model rates it PASS']
    assert v['head'] == 'Avoid: priced 64% above its fair value of $207.'


def test_altman_distress_is_not_a_veto_where_it_does_not_apply():
    # Utilities and REITs sit in Altman's distress zone by construction.
    for sector in ('Utilities', 'Real Estate', 'Financial Services'):
        v = profile_verdict(_good(sector=sector, altman_z_zone='distress', fcf_margin=0.01))
        assert v['v'] != AVOID, sector
        assert not any('Altman' in t for t in v['cons'] + v['flags']), sector
    # And a healthy business in the zone gets a con, not a veto.
    v = profile_verdict(_good(altman_z_zone='distress', altman_z=1.5))
    assert v['v'] != AVOID
    assert 'Altman Z in the distress zone (1.50)' in v['cons']


def test_financials_are_judged_on_capital_not_leverage():
    v = profile_verdict(_good(sector='Financial Services', int_cov=0.5, net_debt=9e11,
                              nd_ebitda=12.0, roe=0.16, er=0.095, cet1_ratio=0.15))
    assert v['v'] == INVEST
    assert 'CET1 capital ratio 15.0%' in v['pros']
    assert 'ROE 16.0% vs cost of equity 9.5%' in v['pros']
    assert not any('Net debt' in t or 'Interest' in t for t in v['cons'])


def test_missing_price_or_fair_value_is_insufficient_data():
    for over in (dict(price=None), dict(_fv_effective=None), dict(_fv_effective=-3.0)):
        v = profile_verdict(_good(**over))
        assert v['v'] == NO_DATA
        assert v['pros'] == [] and v['cons'] == [] and v['flags'] == []
    assert profile_verdict({})['v'] == NO_DATA


def test_multiples_compare_against_the_sector_median():
    base = {'price': 100.0, '_fv_effective': 110.0, 'mos': 0.09, 'rating': 'HOLD'}
    v = profile_verdict(dict(base, pe=20.0, ev_ebitda=30.0, pfcf=-4.0),
                        {'pe': 28.0, 'ev_ebitda': 15.0, 'pfcf': 20.0})
    assert 'P/E 20.0x vs sector 28.0x' in v['pros']
    assert 'EV/EBITDA 30.0x vs sector 15.0x' in v['cons']
    assert not any('P/FCF' in t for t in v['pros'] + v['cons'])  # negative: not a multiple
    assert profile_verdict(dict(base, pe=20.0))['cons'] == []   # no medians, no comparison


def test_missing_inputs_are_na_not_failures():
    sparse = {'price': 100.0, '_fv_effective': 130.0, 'mos': 0.23, 'rating': 'BUY'}
    v = profile_verdict(sparse)
    assert v['cons'] == []
    assert v['v'] == INVEST


def test_rating_cap_reasons_already_covered_are_not_repeated():
    v = profile_verdict(_good(rating='HOLD', _rating_cap_reasons=[
        'non-positive margin of safety', 'Altman Z distress zone', 'thin EDGAR history (3y)']))
    caps = [t for t in v['cons'] if t.startswith('Rating capped')]
    assert caps == ['Rating capped: thin EDGAR history (3y)']


def test_acronyms_keep_their_case_mid_sentence():
    v = profile_verdict(_good(mos=0.2, rev_cagr_5y=None, fcf_margin=None))
    assert 'ROIC 25.0% vs WACC 9.0%' in v['head']


_num = st.one_of(st.none(), st.floats(allow_nan=True, allow_infinity=True),
                 st.integers(-10**6, 10**6), st.text(max_size=3), st.booleans())
_FIELDS = ('price', '_fv_effective', 'mos', 'spread', 'roic', 'wacc', 'fcf_margin',
           'rev_cagr', 'rev_cagr_5y', 'net_debt', 'nd_ebitda', 'int_cov', 'piotroski',
           'altman_z', 'beneish_m', 'trap_score', 'pe', 'ev_ebitda', 'pfcf',
           'target_mean', 'num_analysts', 'insider_net_value', 'mcap',
           'insider_buy_count_365d', 'momentum_12_1', 'margin_trend',
           'shareholder_yield', 'avg_dollar_volume_3m', 'roe', 'er', 'cet1_ratio',
           'npl_ratio', '_composite_score')


@settings(max_examples=300, deadline=None)
@given(st.fixed_dictionaries({}, optional={f: _num for f in _FIELDS}),
       st.sampled_from([None, 'BUY', 'LEAN BUY', 'HOLD', 'PASS', 3]),
       st.sampled_from([None, 'Technology', 'Financial Services', 'Utilities', 7]),
       st.one_of(st.none(), st.lists(st.one_of(st.text(max_size=8), st.none()), max_size=3)),
       st.sampled_from([None, 'safe', 'grey', 'distress']),
       st.sampled_from([None, 'LOW (CV 60%)', 'HIGH', 5]))
def test_never_raises_on_arbitrary_rows(row, rating, sector, reasons, zone, mc):
    row = dict(row, rating=rating, sector=sector, _rating_cap_reasons=reasons,
               trap_reasons=reasons, altman_z_zone=zone, mc_confidence=mc,
               trap_flag=bool(reasons))
    v = profile_verdict(row, {'pe': 20.0, 'pb': float('nan')})
    assert v['v'] in VERDICTS
    assert v['conv'] is None or 0 <= v['conv'] <= 100
    assert isinstance(v['head'], str) and v['head']
    assert all(isinstance(t, str) for t in v['pros'] + v['cons'] + v['flags'] + v['need'])
    assert all(math.isfinite(x) for x in v['med'].values())


@settings(max_examples=200, deadline=None)
@given(st.floats(-1.0, 0.95), st.floats(-1.0, 0.95),
       st.sampled_from(['BUY', 'LEAN BUY', 'HOLD']),
       st.floats(-0.1, 0.3), st.floats(-0.2, 0.4))
def test_verdict_never_improves_as_the_margin_of_safety_shrinks(m1, m2, rating, spread, fcfm):
    hi, lo = max(m1, m2), min(m1, m2)
    fv = 100.0
    def at(mos):
        return profile_verdict(_good(mos=mos, price=fv * (1 - mos), _fv_effective=fv,
                                     rating=rating, spread=spread, fcf_margin=fcfm))['v']
    assert _RANK[at(lo)] <= _RANK[at(hi)]
