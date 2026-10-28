"""Pillar metric functions."""

import pytest

from models import carry, external, momentum, stability, valuation


# ---------------------------------------------------------------- valuation

def _reer(values):
    return [('%04d-%02d' % (1990 + i // 12, i % 12 + 1), v)
            for i, v in enumerate(values)]


def test_cheap_currency_has_negative_z_and_deviation():
    """A currency in long-term real decline — the Japan shape."""
    series = _reer([160.0 - i * 0.25 for i in range(300)])
    out = valuation.compute(series)
    assert out['reer_z20'] < 0
    assert out['reer_dev10'] < 0
    assert out['reer_level'] == series[-1][1]
    assert out['reer_months'] == 300


def test_expensive_currency_has_positive_z():
    series = _reer([80.0 + i * 0.2 for i in range(300)])
    assert valuation.compute(series)['reer_z20'] > 0


def test_valuation_on_empty_series_is_all_none():
    out = valuation.compute([])
    assert out['reer_z20'] is None
    assert out['reer_months'] == 0
    assert not valuation.has_anchor(out)


def test_has_anchor_threshold():
    assert valuation.has_anchor({'reer_months': 60})
    assert not valuation.has_anchor({'reer_months': 59})


# --------------------------------------------------------------------- carry

def test_real_rate_subtracts_inflation():
    assert carry.real_rate(5.0, 2.0) == 3.0
    assert carry.real_rate(None, 2.0) is None
    assert carry.real_rate(5.0, None) is None


def test_high_nominal_carry_can_be_negative_in_real_terms():
    """The carry trap: 37% nominal against 40% inflation is not a 33% pickup."""
    out = carry.compute(policy_rate_pct=37.0, inflation_pct=40.0,
                        base_policy_rate_pct=4.0, base_inflation_pct=3.0,
                        vol_1y=0.20)
    assert out['nominal_carry'] == 33.0
    assert out['real_policy_rate'] == -3.0
    assert out['real_carry'] == -4.0     # -3 versus the base's +1
    assert out['real_carry'] < 0 < out['nominal_carry']


def test_carry_to_vol_ratio():
    out = carry.compute(8.0, 2.0, 4.0, 2.0, vol_1y=0.10)
    assert out['carry_to_vol'] == pytest.approx(0.4)   # 4pp / 10% vol


def test_carry_to_vol_guards_near_zero_volatility():
    """A peg posts almost no realized volatility; without the guard a modest
    carry would divide into an enormous ratio and top the book."""
    out = carry.compute(8.0, 2.0, 4.0, 2.0, vol_1y=0.0001)
    assert out['carry_to_vol'] is None


def test_carry_none_when_base_rate_missing():
    out = carry.compute(5.0, 2.0, None, 2.0, vol_1y=0.1)
    assert out['nominal_carry'] is None
    assert out['real_carry'] is None
    assert out['real_policy_rate'] == 3.0   # still computable on its own leg


# ------------------------------------------------------------------ momentum

def test_momentum_signs_follow_appreciation():
    spot = [('d%04d' % i, 1.0 + i * 0.0005) for i in range(400)]
    reer = _reer([100.0 + i for i in range(12)])
    out = momentum.compute(spot, reer)
    assert out['ret_3m'] > 0
    assert out['ret_12m_1m'] > 0
    assert out['spot_vs_ma200'] > 0
    assert out['reer_trend_6m'] > 0
    assert out['spot'] == spot[-1][1]


def test_momentum_short_history_is_none_not_zero():
    out = momentum.compute([('d1', 1.0), ('d2', 1.1)], [])
    assert out['ret_12m_1m'] is None
    assert out['spot_vs_ma200'] is None
    assert out['spot'] == 1.1


# ------------------------------------------------------------------ external

def test_inflation_gap_is_zero_inside_the_band():
    assert external.inflation_gap(2.0) == 0.0
    assert external.inflation_gap(1.0) == 0.0
    assert external.inflation_gap(3.0) == 0.0


def test_inflation_gap_penalizes_deflation_and_inflation_alike():
    # Deflation is a monetary problem too; scoring raw inflation would rank a
    # deflating economy as maximally price-stable.
    assert external.inflation_gap(-4.0) == 5.0
    assert external.inflation_gap(8.0) == 5.0
    assert external.inflation_gap(None) is None


def test_external_passes_through_and_handles_empty():
    out = external.compute({'current_account_pct_gdp': 3.0,
                            'inflation_pct': 2.0})
    assert out['current_account_pct_gdp'] == 3.0
    assert out['inflation_gap'] == 0.0
    assert out['reserves_months_imports'] is None
    assert external.compute(None)['inflation_pct'] is None


def test_inflation_volatility_from_series():
    macro = {'inflation_pct_series': [(2020, 1.0), (2021, 5.0), (2022, 3.0)]}
    assert external.inflation_volatility(macro) == pytest.approx(2.0)
    assert external.inflation_volatility({}) is None


# ----------------------------------------------------------------- stability

def test_stability_metrics_present_with_enough_history():
    spot = []
    v = 1.0
    for i in range(800):
        spot.append(('d%04d' % i, v))
        v *= 1.002 if i % 3 else 0.997
    out = stability.compute(spot, macro_inflation_vol=1.5)
    assert out['vol_1y'] > 0
    assert out['max_dd_3y'] >= 0
    assert out['vol_ratio'] is not None
    assert out['inflation_vol'] == 1.5


def test_stability_short_history_is_none():
    out = stability.compute([('d1', 1.0), ('d2', 1.01)])
    assert out['vol_1y'] is None
    assert out['max_dd_3y'] is None
    assert out['vol_ratio'] is None
