"""DCF growth model v2 (models/growth_v2.py, config.DCF_GROWTH_MODEL): the
stage-1 path and base-FCF uncertainty in the DCF and Monte Carlo, the model
functions, the wiring behind the flag, and the fit that sets its parameters."""
import math

import numpy as np
import pandas as pd
import pytest

from models import growth_v2 as g2
from models.dcf import monte_carlo_dcf, reverse_dcf, two_stage_ev, two_stage_ev_exit_multiple

P = {  # a small, explicit parameter set (the JSON's shape)
    'history_weights': {'fcf_cagr': 0.0, 'rev_cagr': 0.1, 'fundamental': 0.2},
    'base_rates': {'_all': 0.05, 'Utilities': 0.0},
    'shrink_lambda': 0.5,
    'growth_sigma': {'beta': [-2.2, 0.6, -0.05, 3.6], 'k': 1.9, 'floor': 0.02, 'cap': 0.25,
                     'vol_default': 0.09, 'logrev_default': 21.5},
    'base_sigma': {'a': 0.15, 'b': 0.63, 'floor': 0.10, 'cap': 0.60, 'vol_default': 0.43},
}


# --- DCF: stage-1 path ---------------------------------------------------

def _manual_ev(base, path, rate_after, wacc, tg, years=10, s1=5):
    """Stage 1 follows `path`; stage 2 fades rate_after -> tg; GGM terminal."""
    v, pv = base, 0.0
    for yr in range(1, years + 1):
        g = path[yr - 1] if yr <= s1 else rate_after + (tg - rate_after) * (yr - s1) / (years - s1)
        v *= 1 + g
        pv += v / (1 + wacc) ** yr
    return pv + v * (1 + tg) / (wacc - tg) / (1 + wacc) ** years


def test_zero_offsets_equal_constant_rate():
    assert two_stage_ev(100, 0.06, 0.09, 0.025, stage1_offsets=(0,) * 5) == \
        pytest.approx(two_stage_ev(100, 0.06, 0.09, 0.025))


def test_path_projection_matches_manual():
    path = g2.stage1_path(0.15, 0.05)
    off = g2.path_offsets(path, 0.05)
    assert two_stage_ev(100, 0.05, 0.09, 0.025, stage1_offsets=off) == \
        pytest.approx(_manual_ev(100, path, 0.05, 0.09, 0.025))
    # The exit leg's EBITDA follows the same path.
    ev = two_stage_ev_exit_multiple(100, 0.05, 0.09, 0.025, 150, 10, stage1_offsets=off)
    flat = two_stage_ev_exit_multiple(100, 0.05, 0.09, 0.025, 150, 10)
    assert ev > flat


def test_bad_offsets_are_rejected():
    with pytest.warns(RuntimeWarning):
        assert two_stage_ev(100, 0.05, 0.09, 0.025, stage1_offsets=(0.1,) * 3) is None


def test_reverse_dcf_recovers_long_run_rate_with_path():
    off = g2.path_offsets(g2.stage1_path(0.12, 0.04), 0.04)
    ev = two_stage_ev(100, 0.04, 0.09, 0.025, stage1_offsets=off)
    price = (ev - 200) / 50
    r = reverse_dcf(price, 100, 0.09, 50, 200, terminal_g=0.025, stage1_offsets=off)
    assert r['converged'] and r['implied_growth'] == pytest.approx(0.04, abs=1e-5)


# --- Monte Carlo: path and base-FCF uncertainty --------------------------

MC = dict(base_fcf=100, growth_rate=0.05, discount_rate=0.09, terminal_growth=0.025,
          net_debt=200, shares_outstanding=50, n_iterations=1024, growth_sigma=0.03,
          seed=7)


def test_v1_draws_unchanged_without_v2_inputs():
    a = monte_carlo_dcf(**MC)
    b = monte_carlo_dcf(**MC, stage1_offsets=None, base_sigma=None)
    assert a['quantiles'] == b['quantiles'] and b['base_sigma'] is None


def test_base_sigma_widens_and_is_attributed():
    plain = monte_carlo_dcf(**MC, attribution=True)
    noisy = monte_carlo_dcf(**MC, base_sigma=0.4, attribution=True)
    assert noisy['cv'] > 2 * plain['cv']
    assert noisy['drivers']['base_fcf'] > 0.5 and 'base_fcf' not in plain['drivers']
    # Median-preserving: the centre stays near the plain run's.
    assert noisy['median_fv'] == pytest.approx(plain['median_fv'], rel=0.15)


def test_path_moves_the_mc_centre_like_the_point_value():
    off = g2.path_offsets(g2.stage1_path(0.15, 0.05), 0.05)
    flat = monte_carlo_dcf(**{**MC, 'growth_sigma': 0.001})
    pathd = monte_carlo_dcf(**{**MC, 'growth_sigma': 0.001}, stage1_offsets=off)
    point = (two_stage_ev(100, 0.05, 0.09, 0.025, stage1_offsets=off) - 200) / 50
    assert pathd['median_fv'] > flat['median_fv']
    assert pathd['median_fv'] == pytest.approx(point, rel=0.03)


# --- Model functions -----------------------------------------------------

def test_path_holds_then_fades_and_equivalent_rate():
    path = g2.stage1_path(0.20, 0.05)
    assert path[:2] == [0.20, 0.20] and path[-1] == pytest.approx(0.05)
    assert path[2] > path[3] > path[4]
    eq = g2.equivalent_rate(path)
    assert np.prod([1 + g for g in path]) == pytest.approx((1 + eq) ** 5)


def test_long_run_rate_shrinks_toward_base_rate():
    L, h, b = g2.long_run_rate({'rev_cagr': 0.15, 'fundamental': 0.15, 'fcf_cagr': 0.9},
                               'Technology', 0.15, P)
    assert h == pytest.approx(0.15) and b == 0.05     # FCF CAGR weight is 0
    assert L == pytest.approx(0.5 * 0.15 + 0.5 * 0.05)
    assert g2.long_run_rate({}, 'Utilities', 0.15, P)[0] == 0.0      # no history: base
    assert g2.long_run_rate({'rev_cagr': -0.9}, None, 0.15, P)[0] == g2.GROWTH_FLOOR


def test_near_term_rate_uses_analysts_else_long_run():
    w = {'st': 0.15, 'lt': 0.30, 'earnings': 0.15}
    g0, a = g2.near_term_rate({'st': 0.10, 'lt': 0.20, 'earnings': None}, w, 0.01, 0.05, 0.25)
    assert a == pytest.approx((0.10 * 0.15 + 0.20 * 0.30) / 0.45)
    assert g0 == pytest.approx(a + 0.01)
    assert g2.near_term_rate({}, w, 0.01, 0.05, 0.15) == (pytest.approx(0.06), None)
    assert g2.near_term_rate({'lt': 0.9}, w, 0.0, 0.05, 0.15)[0] == 0.15     # capped


def test_sigmas_are_monotone_and_bounded():
    lo = g2.growth_sigma(0.02, 1e10, 0.03, P)
    hi = g2.growth_sigma(0.60, 1e10, 0.03, P)
    assert P['growth_sigma']['floor'] <= lo < hi <= P['growth_sigma']['cap']
    assert g2.growth_sigma(None, None, 0.03, P) > 0          # defaults fill gaps
    steady = g2.base_sigma([100, 101, 102, 103, 104, 105], P)
    lumpy = g2.base_sigma([100, -40, 160, 20, 140, 60], P)
    assert P['base_sigma']['floor'] <= steady < lumpy <= P['base_sigma']['cap']
    assert g2.fcf_volatility([1, 2]) is None
    assert g2.fcf_volatility([100, -40, 160, 20]) > 0


def test_shipped_params_load_and_fit_the_contract():
    p = g2.load_params()
    assert p['history_weights']['fcf_cagr'] == 0.0
    lo, hi = g2.BASE_RATE_BOUNDS
    assert all(lo <= v <= hi for v in p['base_rates'].values())
    assert 0 < p['shrink_lambda'] < 1 and len(p['growth_sigma']['beta']) == 4


# --- Wiring behind the flag ---------------------------------------------

def _yf():
    from tests.test_monte_carlo import _dcf_yf_data
    yf = _dcf_yf_data()
    yf['growth_estimates'] = pd.DataFrame.from_dict({'+1y': {'stockTrend': 0.20}}, orient='index')
    yf['info'] = {**yf['info'], 'revenueGrowth': 0.10, 'earningsGrowth': 0.15}
    return yf


def test_run_forward_dcf_under_v2(monkeypatch):
    from scripts import analyze_stock as az
    _, _, g1, d1, mc1 = az.run_forward_dcf(_yf(), wacc=0.10, rev_growth_vol=0.2)
    monkeypatch.setattr(az, 'DCF_GROWTH_MODEL', 'v2')
    fv, sens, g, d, mc = az.run_forward_dcf(_yf(), wacc=0.10, rev_growth_vol=0.2)
    v2 = d['dcf_inputs']['v2']
    assert d['dcf_inputs']['model'] == 'v2' and d1['dcf_inputs']['model'] == 'v1'
    assert v2['v1_growth'] == pytest.approx(g1)
    assert g == pytest.approx(g2.equivalent_rate(v2['path']))
    assert d['dcf_growth_rate'] == v2['long_run'] and len(d['stage1_offsets']) == 5
    assert mc['base_sigma'] == v2['base_sigma'] > 0 and 'base_fcf' in mc['drivers']
    assert mc['cv'] > mc1['cv'] and sens == (mc['p10_fv'], mc['p90_fv'])


def test_fingerprint_changes_only_under_v2(monkeypatch):
    from scripts import param_set
    base = param_set.default_params()
    assert 'dcf_growth_model' not in base
    monkeypatch.setattr(param_set, 'DCF_GROWTH_MODEL', 'v2')
    v2 = param_set.default_params()
    assert v2['dcf_growth_model'] == 'v2' and len(v2['dcf_growth_v2_params']) == 12
    assert param_set.scoring_params_hash(v2) != param_set.scoring_params_hash(base)


def test_labels_and_spread_follow_the_model(monkeypatch):
    from models import profile_verdict as pv
    from scripts import scoring
    assert scoring._mc_confidence_label(0.5).startswith('LOW')
    monkeypatch.setattr(scoring, 'DCF_GROWTH_MODEL', 'v2')
    assert scoring._mc_confidence_label(0.5).startswith('HIGH')
    assert pv._spread_wide({}) == 3.0
    assert pv._spread_wide({'dcf_model': 'v2'}) == pv.MC_SPREAD_WIDE_BY_MODEL['v2']


# --- The fit ---------------------------------------------------------------

def _rows(n=70, years=range(2006, 2026), seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        g, vol = rng.normal(0.06, 0.04), 0.05 + 0.4 * (i % 5) / 4
        rev = op = None
        fcf, rv, opi = {}, {}, {}
        for k, y in enumerate(years):
            rev = 1000 * (1 + g) ** k * math.exp(rng.normal(0, 0.03))
            op = 0.15 * rev
            fcf[str(y)] = 0.10 * rev * math.exp(rng.normal(0, vol))
            rv[str(y)], opi[str(y)] = rev, op
        rows.append({'ticker': f'T{i}', 'sector': 'Industrials', 'edgar_history': {
            'operating_cf_history': {y: v + 10 for y, v in fcf.items()},
            'capex_history': {y: 10 for y in fcf}, 'revenue_history': rv,
            'operating_income_history': opi}})
    return rows


def test_fit_v2_produces_a_usable_parameter_set():
    from scripts import growth_hindcast as gh
    p = gh.fit_v2(_rows())
    lo, hi = g2.BASE_RATE_BOUNDS
    assert lo <= p['base_rates']['Industrials'] <= hi
    assert p['growth_sigma']['k'] > 0 and p['base_sigma']['b'] > 0   # noisier FCF, wider
    assert set(p['fit']['walk_forward_lambda']) <= {str(x) for x in gh.LAMBDA_GRID}
    assert g2.growth_sigma(0.1, 1e9, 0.05, p) > 0 and g2.base_sigma([1, 2, 3, 4], p) > 0
