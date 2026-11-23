"""Growth hindcast (scripts/growth_hindcast.py): point-in-time estimates,
realised growth, CRPS and walk-forward fitting without look-ahead."""
import math

import numpy as np
import pytest
from scipy import integrate
from scipy.stats import norm

from scripts import growth_hindcast as gh


def _eh(fcf, rev=None, start=2006):
    """edgar_history with OCF = fcf + 10, capex = 10 (so owner FCF = fcf)."""
    years = range(start, start + len(fcf))
    rev = rev or [1000 * 1.05 ** i for i in range(len(fcf))]
    return {
        'operating_cf_history': {str(y): f + 10 for y, f in zip(years, fcf, strict=True)},
        'capex_history': {str(y): 10 for y in years},
        'revenue_history': {str(y): r for y, r in zip(years, rev, strict=True)},
    }


def test_realised_growth_inverts_projection():
    base, g = 100.0, 0.07
    future = [base * (1 + g) ** k for k in range(1, 6)]
    y, censored = gh.realised_growth(base, future)
    assert y == pytest.approx(g, abs=1e-9) and not censored


def test_realised_growth_handles_a_negative_year_and_censors():
    y, censored = gh.realised_growth(100.0, [120, -50, 110, 120, 130])
    assert not censored and -0.5 < y < 0.0
    assert gh.realised_growth(100.0, [-10, -10, -10, -10, -10]) == (gh.G_MIN, True)


def test_estimate_reads_nothing_after_origin():
    fcf = [100 * 1.1 ** i for i in range(12)]
    a = _eh(fcf)
    b = _eh(fcf[:8] + [1, 5000, -300, 2])          # future rewritten
    t = 2006 + 7
    assert gh.estimate(a, t) == gh.estimate(b, t)
    assert gh.rev_growth_vol(gh.annual(a['revenue_history']), t) == \
        gh.rev_growth_vol(gh.annual(b['revenue_history']), t)


def test_estimate_blends_fcf_and_revenue_cagr_and_clamps():
    eh = _eh([100 * 1.10 ** i for i in range(8)],
             rev=[1000 * 1.04 ** i for i in range(8)])
    g, sig, wts = gh.estimate(eh, 2006 + 7, sector='Technology')
    assert sig == pytest.approx([0.10, 0.04]) and wts == [0.10, 0.10]
    assert g == pytest.approx(0.07)
    hot = _eh([100 * 1.6 ** i for i in range(8)], rev=[1000 * 1.6 ** i for i in range(8)])
    assert gh.estimate(hot, 2006 + 7, sector='Technology')[0] == 0.15   # sector cap


def test_build_cases_needs_lookback_and_five_outcome_years():
    cases = gh.build_cases([{'ticker': 'A', 'edgar_history': _eh([100] * 12)}])
    origins = [c['origin'] for c in cases]
    # 3-year look-back for the CAGRs, five years of outcome after.
    assert origins == list(range(2006 + 3, 2006 + 12 - 5))
    assert all(c['realised'] == pytest.approx(0.0, abs=1e-9) for c in cases)


def test_revenue_target():
    cases = gh.build_cases([{'ticker': 'A', 'edgar_history': _eh(
        [100] * 12, rev=[1000 * 1.08 ** i for i in range(12)])}], target='revenue')
    assert cases and all(c['realised'] == pytest.approx(0.08) for c in cases)
    with pytest.raises(ValueError):
        gh.build_cases([], target='price')


@pytest.mark.parametrize('y,mu,s', [(0.1, 0.0, 0.05), (-0.3, 0.02, 0.2), (0.0, 0.0, 1.0)])
def test_crps_matches_numeric_integral(y, mu, s):
    f = lambda x: (norm.cdf((x - mu) / s) - (x >= y)) ** 2   # noqa: E731
    num = sum(integrate.quad(f, a, b, limit=200)[0]
              for a, b in ((mu - 12 * s, y), (y, mu + 12 * s)))
    assert float(gh.crps_normal(y, mu, s)) == pytest.approx(num, rel=1e-5)


def _synthetic(n_tickers=80, years=range(2005, 2025), seed=1):
    """Cases whose errors scale with rev_vol: the fitted rule should learn it."""
    rng = np.random.default_rng(seed)
    cases = []
    for i in range(n_tickers):
        vol = 0.05 + 0.3 * (i % 4) / 3
        for t in years:
            g = rng.normal(0.05, 0.03)
            cases.append({'ticker': f'T{i}', 'origin': t, 'estimate': g,
                          'realised': g + 0.02 + rng.normal(0, vol),
                          'signals': [g, g + 0.01], 'weights': [0.1, 0.1],
                          'rev_vol': vol, 'log_rev': 20.0 + rng.normal(),
                          'censored': False})
    return cases


def test_fitted_rule_uses_only_resolved_outcomes(monkeypatch):
    cases = _synthetic()
    newest_trained = []
    real_fit = gh.fit_scale

    def spy(train):
        newest_trained.append(max(c['origin'] for c in train))
        return real_fit(train)

    monkeypatch.setattr(gh, 'fit_scale', spy)
    gh.forecasts(cases)
    # forecasts fits once per origin, in order, whenever training data exists.
    origins = sorted({c['origin'] for c in cases})
    fitted_for = [t for t in origins if t - gh.HORIZON >= origins[0]]
    assert len(newest_trained) == len(fitted_for)
    assert all(last == t - gh.HORIZON for t, last in zip(fitted_for, newest_trained, strict=True))


def test_fitted_rule_beats_fixed_sigma_on_heteroskedastic_errors():
    cases = _synthetic()
    gh.forecasts(cases)
    res = gh.score(cases)
    rules = res['rules']
    assert rules['fitted']['crps'] < rules['ratio']['crps']
    assert rules['fitted_bias']['crps'] <= rules['fitted']['crps'] + 1e-3
    # The fitted band is roughly calibrated; the 2pp ratio rule is far too narrow.
    assert 0.35 < rules['fitted']['in_50'] < 0.65
    assert rules['ratio']['in_90'] < 0.4
    assert all(r['n'] > 0 for r in rules['ratio']['per_origin'])
    assert math.isfinite(res['estimate_rank_ic'])


def test_score_compares_rules_on_common_rows():
    cases = _synthetic(n_tickers=40)
    gh.forecasts(cases)
    early = gh.score(cases, ('ratio', 'evidence'))
    full = gh.score(cases)
    assert full['n_obs'] < early['n_obs']           # fitted starts later
    assert min(full['origins']) >= min(early['origins']) + gh.HORIZON
