"""Growth hindcast: test the DCF Monte Carlo's growth sigma against what
companies' free cash flow actually did.

The DCF simulation holds one growth draw for the 5-year explicit stage, and
that draw carries most of the fair-value spread (growth is the top Sobol
driver on ~79% of DCF rows, median share ~0.62 on 2026-11-20). Its sigma is
a rule of thumb (max(0.30·|g|, 2pp)) or the evidence sigma that ships OFF.
Neither has been checked against outcomes, and the price-based band check
(backtest.py measure, mc_band_coverage) cannot judge fair value until 365d+.

Growth itself is observable, and every snapshot row carries 10-16 years of
EDGAR history. So, for each company and each past fiscal year t:

  1. Slice the history to years <= t (point in time) and rebuild the growth
     estimate from the signals that history can supply: trailing FCF and
     revenue CAGRs from the oldest to the newest year on file (as the live
     DCF reads them off XBRL statements spanning 10-17 years) and fundamental
     growth (reinvestment rate x ROIC), at their production weights,
     renormalised, clamped to [-15%, sector cap].
  2. Realised growth: the constant rate g that makes the DCF's own stage-1
     projection, base x (1+g)^k for k = 1..5, sum to the FCF the company
     actually produced in t+1..t+5 (SBC-adjusted like the live base). That is
     the growth the stage-1 value depends on, and it stays defined through a
     negative year as long as the 5-year total is positive (otherwise it is
     censored at G_MIN).
  3. Score each sigma rule as a forecast distribution N(estimate, sigma) of
     realised growth: CRPS (lower is better; it penalises both misses and
     needless width, unlike band coverage), PIT coverage of the middle 50% /
     90%, mean PIT (0.5 = unbiased) and mean error.

Sigma rules compared (all on the same rows):

  ratio      the live rule, growth_sigma(g)
  evidence   evidence_growth_sigma over the signals rebuilt here
  fitted     a scale model, log|error| ~ revenue-growth volatility, size and
             |estimate|, fitted only on outcomes RESOLVED by year t (origins
             <= t - 5), so no look-ahead; 'fitted_bias' also learns the
             estimate's mean error (a + b·estimate), i.e. how much the
             estimate should shrink

Limits, read before acting on a result:

  - The live blend also uses three analyst signals (60% of nominal weight)
    plus margin-trend and surprise adjustments. Point-in-time analyst data
    is not stored, so this tests the history-based estimator, not the live
    estimate. Its error scale is the question; the live estimate's errors
    may be smaller or larger.
  - Survivorship: histories exist only for companies listed today, so
    failures are missing and the downside tail is understated.
  - XBRL values may carry later restatements (mild look-ahead in inputs).
  - Origins one year apart share four outcome years: n_obs overstates the
    independent sample. Results are averaged per origin year first.

Usage:
  python scripts/growth_hindcast.py                       # newest snapshot
  python scripts/growth_hindcast.py --snapshot output/results_2026-11-20.json
  python scripts/growth_hindcast.py --target revenue      # revenue CAGR target
  python scripts/growth_hindcast.py --all-rows            # not just DCF rows
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import norm, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import growth_v2  # noqa: E402
from models.growth_v2 import fcf_volatility  # noqa: E402
from models.montecarlo import evidence_growth_sigma, growth_sigma  # noqa: E402
from scripts.config import (  # noqa: E402
    GROWTH_WEIGHT_FCF,
    GROWTH_WEIGHT_FUNDAMENTAL,
    GROWTH_WEIGHT_REV,
    MC_GROWTH_SIGMA_CAP,
    MC_GROWTH_SIGMA_FLOOR,
    MC_GROWTH_SIGMA_HISTORY_YEARS,
    MC_GROWTH_SIGMA_RATIO,
    _get_sector_config,
)

HORIZON = 5            # DCF_STAGE1: years the single growth draw covers
CAGR_SPAN = 3          # minimum look-back before an origin is scored
G_MIN, G_MAX = -0.95, 3.0
MIN_REV_YEARS = 4      # derive_edgar_metrics' rev_growth_vol minimum
MIN_TRAIN = 200        # fitted rule: resolved outcomes needed to fit
BANDS = (('50', 0.25, 0.75), ('90', 0.05, 0.95))
DEFAULT_TAX = 0.21
K_GRID = tuple(float(x) for x in np.geomspace(0.25, 4.0, 49))


# ---------------------------------------------------------------------------
# History helpers
# ---------------------------------------------------------------------------

def annual(history):
    """{int year: float} from an edgar_history series (year-keyed dict)."""
    out = {}
    for k, v in (history or {}).items():
        try:
            yr = int(str(k)[:4])
        except ValueError:
            continue
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
            out[yr] = float(v)
    return out


def owner_fcf(eh):
    """{year: OCF - |capex| - SBC}, SBC floored like the live base
    (never below 25% of pre-SBC FCF when that is positive)."""
    ocf, cap, sbc = (annual(eh.get(k)) for k in
                     ('operating_cf_history', 'capex_history', 'sbc_cf_history'))
    out = {}
    for y in ocf:
        if y not in cap:
            continue
        f = ocf[y] - abs(cap[y])
        s = abs(sbc.get(y, 0.0))
        out[y] = max(f - s, 0.25 * f) if f > 0 else f
    return out


def full_cagr(series, t):
    """Live DCF definition: oldest to newest point at or before t, both > 0,
    over their span in years (_run_forward_dcf's fcf_cagr / rev_cagr)."""
    ys = sorted(y for y in series if y <= t)
    if len(ys) < 2 or ys[-1] != t:
        return None
    a, b = series[ys[0]], series[t]
    if a <= 0 or b <= 0:
        return None
    return (b / a) ** (1 / (t - ys[0])) - 1


def cagr(series, t, span=CAGR_SPAN):
    a, b = series.get(t - span), series.get(t)
    if a is None or b is None or a <= 0 or b <= 0:
        return None
    return (b / a) ** (1 / span) - 1


def rev_growth_vol(rev, t):
    """Population std of YoY revenue growth over years <= t (as live)."""
    vals = [rev[y] for y in sorted(rev) if y <= t and rev[y] > 0]
    if len(vals) < MIN_REV_YEARS:
        return None
    gr = [(vals[i] - vals[i - 1]) / vals[i - 1] for i in range(1, len(vals))]
    m = sum(gr) / len(gr)
    return (sum((x - m) ** 2 for x in gr) / len(gr)) ** 0.5


def fundamental_growth(eh, t):
    """Reinvestment rate x ROIC at year t, from EDGAR series (approximates
    models.ratios.calculate_fundamental_growth, which reads statements).
    None on an operating loss or missing inputs."""
    g = {k: annual(eh.get(k)) for k in (
        'operating_income_history', 'pretax_income_history',
        'tax_provision_history', 'capex_history', 'd_and_a_history',
        'current_assets_history', 'current_liabilities_history',
        'cash_history', 'st_investments_history', 'total_debt_history',
        'total_equity_history')}
    op = g['operating_income_history'].get(t)
    capex = g['capex_history'].get(t)
    da = g['d_and_a_history'].get(t)
    eq = g['total_equity_history'].get(t)
    if op is None or op <= 0 or capex is None or da is None or eq is None:
        return None
    pt, tx = g['pretax_income_history'].get(t), g['tax_provision_history'].get(t)
    tax = tx / pt if (pt and pt > 0 and tx is not None) else DEFAULT_TAX
    nopat = op * (1 - min(max(tax, 0.0), 0.35))

    def nwc(y):
        ca = g['current_assets_history'].get(y)
        cl = g['current_liabilities_history'].get(y)
        if ca is None or cl is None:
            return None
        cash = g['cash_history'].get(y, 0.0) + g['st_investments_history'].get(y, 0.0)
        return ca - cash - cl

    n1, n0 = nwc(t), nwc(t - 1)
    d_nwc = (n1 - n0) if (n1 is not None and n0 is not None) else 0.0
    reinvest = (abs(capex) - abs(da) + d_nwc) / nopat
    cash = g['cash_history'].get(t, 0.0) + g['st_investments_history'].get(t, 0.0)
    ic = g['total_debt_history'].get(t, 0.0) + eq - cash
    if ic <= 0:
        return None
    return reinvest * (nopat / ic)


# ---------------------------------------------------------------------------
# Point estimate and realised growth
# ---------------------------------------------------------------------------

def estimate(eh, t, sector=None):
    """History-only growth estimate at year t: (g, signals, weights) or None.

    Reads nothing after year t (tests pin this)."""
    fcf = owner_fcf(eh)
    if t - CAGR_SPAN not in fcf:
        return None
    named = history_signals(eh, t)
    pairs = [(named['fcf_cagr'], GROWTH_WEIGHT_FCF),
             (named['rev_cagr'], GROWTH_WEIGHT_REV),
             (named['fundamental'], GROWTH_WEIGHT_FUNDAMENTAL)]
    pairs = [(s, w) for s, w in pairs if s is not None and math.isfinite(s)]
    if not pairs:
        return None
    sig, wts = [p[0] for p in pairs], [p[1] for p in pairs]
    g = sum(s * w for s, w in pairs) / sum(wts)
    cap = _get_sector_config(sector or '')['growth_cap']
    return min(cap, max(-0.15, g)), sig, wts


def history_signals(eh, t):
    """The history signals by name, as the live blend keys them."""
    fcf, rev = owner_fcf(eh), annual(eh.get('revenue_history'))
    return {'fcf_cagr': full_cagr(fcf, t), 'rev_cagr': full_cagr(rev, t),
            'fundamental': fundamental_growth(eh, t)}


def realised_growth(base, future):
    """Constant g with sum_k base·(1+g)^k = sum(future), k = 1..len(future).

    Returns (g, censored). The left side rises monotonically in g on
    (-1, inf), so bisection is exact; a total <= 0 (or beyond G_MAX) is
    censored at the bound."""
    total, n = float(sum(future)), len(future)

    def proj(g):
        return sum(base * (1 + g) ** k for k in range(1, n + 1))

    if total <= proj(G_MIN):
        return G_MIN, True
    if total >= proj(G_MAX):
        return G_MAX, True
    lo, hi = G_MIN, G_MAX
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if proj(mid) < total else (lo, mid)
    return 0.5 * (lo + hi), False


TARGETS = ('fcf', 'revenue')


def build_cases(rows, target='fcf'):
    """One case per (ticker, origin year) with a positive base FCF, a 3-year
    look-back and all five outcome years present.

    target 'fcf' (default): realised_growth of the SBC-adjusted FCF, what the
    DCF's stage 1 projects. 'revenue': the revenue CAGR over t..t+5, which
    carries no base-year noise, so the gap between the two shows how much of
    the FCF miss is a noisy base year rather than wrong growth."""
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}")
    cases = []
    for r in rows:
        eh = r.get('edgar_history')
        if not isinstance(eh, dict):
            continue
        fcf, rev = owner_fcf(eh), annual(eh.get('revenue_history'))
        for t in sorted(fcf):
            base = fcf[t]
            future = [fcf.get(t + k) for k in range(1, HORIZON + 1)]
            if base <= 0 or any(v is None for v in future):
                continue
            est = estimate(eh, t, r.get('sector'))
            if est is None:
                continue
            g, sig, wts = est
            if target == 'revenue':
                r0, r5 = rev.get(t), rev.get(t + HORIZON)
                if not (r0 and r5 and r0 > 0 and r5 > 0):
                    continue
                y, censored = (r5 / r0) ** (1 / HORIZON) - 1, False
            else:
                y, censored = realised_growth(base, future)
            vol = rev_growth_vol(rev, t)
            r0, r5 = rev.get(t), rev.get(t + HORIZON)
            cases.append({
                'named': history_signals(eh, t), 'base': base, 'future': future,
                'fcf_vol': fcf_volatility([fcf.get(y) for y in range(t - 5, t + 1)]),
                'rev_realised': ((r5 / r0) ** (1 / HORIZON) - 1
                                 if r0 and r5 and r0 > 0 and r5 > 0 else None),
                'cap': _get_sector_config(r.get('sector') or '')['growth_cap'],
                'ticker': r.get('ticker'), 'sector': r.get('sector'),
                'origin': t, 'estimate': g, 'realised': y,
                'censored': censored, 'signals': sig, 'weights': wts,
                'rev_vol': vol, 'log_rev': (math.log(rev[t])
                                            if rev.get(t, 0) > 0 else None),
            })
    return cases


# ---------------------------------------------------------------------------
# Sigma rules
# ---------------------------------------------------------------------------

def _features(c):
    vol = c['rev_vol'] if c['rev_vol'] is not None else np.nan
    return [1.0, min(vol, 1.0), c['log_rev'] if c['log_rev'] is not None else np.nan,
            abs(c['estimate'])]


def fit_scale(train):
    """Fit location bias (a + b·est) and log-scale on resolved cases.

    Returns a predictor case -> (mu_shift, sigma, sigma_nobias) or None.
    The scale constant k is the one that minimises mean training CRPS: the
    errors are heavy-tailed, so matching their variance would set sigma to
    the tails and overstate the middle band, and matching the median would
    understate the tails; CRPS weighs both, as the evaluation does."""
    X = np.array([_features(c) for c in train], dtype=float)
    ok = np.all(np.isfinite(X), axis=1)
    if ok.sum() < MIN_TRAIN:
        return None
    X = X[ok]
    est = np.array([c['estimate'] for c in train])[ok]
    y = np.array([c['realised'] for c in train])[ok]
    A = np.column_stack([np.ones_like(est), est])
    ab, *_ = np.linalg.lstsq(A, y - est, rcond=None)

    def scale(resid):
        z = np.log(np.abs(resid) + 1e-3)
        beta, *_ = np.linalg.lstsq(X, z, rcond=None)
        s = np.exp(X @ beta)
        k = min(K_GRID, key=lambda kk: float(np.mean(crps_normal(resid, 0.0, kk * s))))
        return beta, float(k)

    b_bias, k_bias = scale(y - est - A @ ab)
    b_raw, k_raw = scale(y - est)

    def predict(c):
        x = np.array(_features(c), dtype=float)
        if not np.all(np.isfinite(x)):
            return None
        shift = float(ab[0] + ab[1] * c['estimate'])
        return (shift,
                max(MC_GROWTH_SIGMA_FLOOR, k_bias * float(np.exp(x @ b_bias))),
                max(MC_GROWTH_SIGMA_FLOOR, k_raw * float(np.exp(x @ b_raw))))

    predict.coef = {'bias_a': float(ab[0]), 'bias_b': float(ab[1]),
                    'n_train': int(ok.sum())}
    return predict


def forecasts(cases):
    """Attach (mu, sigma) per rule to each case; fitted rules walk forward:
    origin t uses only cases with origin <= t - HORIZON (outcome known)."""
    by_origin = defaultdict(list)
    for c in cases:
        by_origin[c['origin']].append(c)
    coefs = {}
    for t in sorted(by_origin):
        train = [c for c in cases if c['origin'] <= t - HORIZON]
        model = fit_scale(train) if train else None
        if model:
            coefs[t] = model.coef
        for c in by_origin[t]:
            g = c['estimate']
            ev, _, _ = evidence_growth_sigma(
                g, c['signals'], c['weights'], c['rev_vol'],
                years=MC_GROWTH_SIGMA_HISTORY_YEARS, floor=MC_GROWTH_SIGMA_FLOOR,
                cap=MC_GROWTH_SIGMA_CAP, ratio=MC_GROWTH_SIGMA_RATIO)
            c['fc'] = {'ratio': (g, growth_sigma(g, MC_GROWTH_SIGMA_RATIO,
                                                 MC_GROWTH_SIGMA_FLOOR)),
                       'evidence': (g, ev)}
            p = model(c) if model else None
            if p:
                shift, s_bias, s_raw = p
                c['fc']['fitted'] = (g, s_raw)
                c['fc']['fitted_bias'] = (g + shift, s_bias)
    return coefs


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def crps_normal(y, mu, sigma):
    """Closed-form CRPS of N(mu, sigma) at outcome y (Gneiting et al. 2005)."""
    z = (np.asarray(y, dtype=float) - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / math.sqrt(math.pi))


def score(cases, rules=None):
    """Per rule: per-origin means, then their equal-weight mean.

    Only cases that carry EVERY rule are scored, so rules compare on the
    same rows (the fitted rules start once enough outcomes have resolved)."""
    rules = rules or ('ratio', 'evidence', 'fitted', 'fitted_bias')
    common = [c for c in cases if all(r in c.get('fc', {}) for r in rules)]
    per_origin = defaultdict(lambda: defaultdict(list))
    for c in common:
        for r in rules:
            mu, s = c['fc'][r]
            pit = float(norm.cdf((c['realised'] - mu) / s))
            per_origin[c['origin']][r].append(
                (float(crps_normal(c['realised'], mu, s)), pit, c['realised'] - mu, s))
    out = {'n_obs': len(common), 'origins': sorted(per_origin), 'rules': {}}
    for r in rules:
        rows = []
        for t in sorted(per_origin):
            v = per_origin[t][r]
            crps, pit, err, sig = (np.array(x) for x in zip(*v, strict=True))
            rows.append({
                'origin': t, 'n': len(v), 'crps': float(crps.mean()),
                'mean_pit': float(pit.mean()), 'mean_error': float(err.mean()),
                'median_sigma': float(np.median(sig)),
                **{f'in_{b}': float(np.mean((pit >= lo) & (pit <= hi)))
                   for b, lo, hi in BANDS}})
        out['rules'][r] = {
            'per_origin': rows,
            **{k: float(np.mean([x[k] for x in rows])) for k in
               ('crps', 'mean_pit', 'mean_error', 'median_sigma', 'in_50', 'in_90')},
        } if rows else None
    # Skill of the point estimate itself: per-origin rank correlation.
    rhos = []
    for t in sorted({c['origin'] for c in cases}):
        cs = [c for c in cases if c['origin'] == t]
        if len(cs) >= 30:
            rhos.append(spearmanr([c['estimate'] for c in cs],
                                  [c['realised'] for c in cs]).statistic)
    out['estimate_rank_ic'] = float(np.mean(rhos)) if rhos else None
    return out


# ---------------------------------------------------------------------------
# Fit the v2 growth model's parameters (models/growth_v2_params.json)
# ---------------------------------------------------------------------------

V2_HISTORY_WEIGHTS = {'fcf_cagr': 0.0, 'rev_cagr': GROWTH_WEIGHT_REV,
                      'fundamental': GROWTH_WEIGHT_FUNDAMENTAL}
# Walk-forward on the five-year cash error favours λ = 0.25, but that leans
# hardest on base rates measured on survivors only; 0.5 keeps most of the
# gain (rank IC +0.16 vs +0.09 for v1, error 0.374 vs 0.393) while halving
# the exposure to a biased base rate. Revisit once delisted filers are in.
V2_SHRINK_LAMBDA = 0.5
LAMBDA_GRID = (0.25, 0.5, 0.75, 1.0)


def _cash_sum(base, path):
    total, v = 0.0, base
    for g in path:
        v *= 1 + g
        total += v
    return total


def _dlog_cash_dg(g, n=HORIZON):
    """d ln(sum_k (1+g)^k) / dg: how a growth error moves the 5-year cash."""
    num = sum(k * (1 + g) ** (k - 1) for k in range(1, n + 1))
    return num / sum((1 + g) ** k for k in range(1, n + 1))


def _base_rates(cases, bounds=growth_v2.BASE_RATE_BOUNDS, min_n=30):
    """Sector base rate: the lower of the median realised FCF and revenue
    growth, clipped to `bounds` (survivor-only medians read high)."""
    by = defaultdict(lambda: ([], []))
    for c in cases:
        by[c['sector'] or ''][0].append(c['realised'])
        if c['rev_realised'] is not None:
            by[c['sector'] or ''][1].append(c['rev_realised'])
    every_f = [c['realised'] for c in cases]
    every_r = [c['rev_realised'] for c in cases if c['rev_realised'] is not None]

    def rate(f, r):
        return float(np.clip(min(np.median(f), np.median(r) if r else np.median(f)), *bounds))

    out = {'_all': rate(every_f, every_r)}
    for sec, (f, r) in by.items():
        out[sec] = rate(f, r) if len(f) >= min_n else out['_all']
    return out


def _v2_long_run(c, lam, rates):
    p = {'history_weights': V2_HISTORY_WEIGHTS, 'base_rates': rates, 'shrink_lambda': lam}
    return growth_v2.long_run_rate(c['named'], c['sector'], c['cap'], p)[0]


def walk_forward_lambda(cases):
    """Per λ: per-origin median |log(actual / projected 5-year cash)|, its
    bias, and the rank IC of L, with base rates from resolved outcomes only."""
    out = {lam: defaultdict(list) for lam in LAMBDA_GRID}
    for t in sorted({c['origin'] for c in cases}):
        train = [c for c in cases if c['origin'] <= t - HORIZON]
        test = [c for c in cases if c['origin'] == t and sum(c['future']) > 0]
        if len(train) < 300 or len(test) < 30:
            continue
        rates = _base_rates(train)
        for lam in LAMBDA_GRID:
            L = np.array([_v2_long_run(c, lam, rates) for c in test])
            err = np.array([math.log(sum(c['future']) / _cash_sum(c['base'], [g] * HORIZON))
                            for c, g in zip(test, L, strict=True)])
            out[lam]['abs'].append(float(np.median(np.abs(err))))
            out[lam]['bias'].append(float(np.median(err)))
            out[lam]['ic'].append(float(spearmanr(L, [c['realised'] for c in test]).statistic))
    return {str(lam): {k: float(np.mean(v)) for k, v in d.items()} | {'origins': len(d['abs'])}
            for lam, d in out.items() if d['abs']}


def fit_v2(rows, lam=V2_SHRINK_LAMBDA):
    """Fit models/growth_v2_params.json from the snapshot rows' histories."""
    cases = build_cases(rows, 'fcf')
    rates = _base_rates(cases)
    L = np.array([_v2_long_run(c, lam, rates) for c in cases])

    # Growth sigma: scale model on realised REVENUE growth (no base noise).
    gi = [i for i, c in enumerate(cases) if c['rev_realised'] is not None
          and c['rev_vol'] is not None and c['log_rev'] is not None]
    X = np.array([[1.0, min(cases[i]['rev_vol'], 1.0), cases[i]['log_rev'], abs(L[i])] for i in gi])
    e = np.array([cases[i]['rev_realised'] - L[i] for i in gi])
    beta, *_ = np.linalg.lstsq(X, np.log(np.abs(e) + 1e-3), rcond=None)
    s0 = np.exp(X @ beta)
    k = min(K_GRID, key=lambda kk: float(np.mean(crps_normal(e, 0.0, kk * s0))))
    gs = {'beta': [float(b) for b in beta], 'k': float(k), 'floor': MC_GROWTH_SIGMA_FLOOR,
          'cap': 0.25, 'vol_default': float(np.median(X[:, 1])),
          'logrev_default': float(np.median(X[:, 2]))}
    params = {'history_weights': V2_HISTORY_WEIGHTS, 'base_rates': rates,
              'shrink_lambda': lam, 'growth_sigma': gs}
    sig_g = {i: growth_v2.growth_sigma(cases[i]['rev_vol'], math.exp(cases[i]['log_rev']), L[i], params)
             for i in gi}

    # Base-FCF sigma: the 5-year cash error left once the growth sigma's
    # share is taken out, var(e) = sigma_b^2 + D^2 sigma_g^2, by FCF-volatility
    # decile; then a line through the deciles.
    bi = [i for i in gi if cases[i]['fcf_vol'] is not None and sum(cases[i]['future']) > 0]
    vol = np.array([cases[i]['fcf_vol'] for i in bi])
    elog = np.array([math.log(sum(cases[i]['future']) / _cash_sum(cases[i]['base'], [L[i]] * HORIZON))
                     for i in bi])
    gpart = np.array([(_dlog_cash_dg(L[i]) * sig_g[i]) ** 2 for i in bi])
    edges = np.quantile(vol, np.linspace(0, 1, 11))
    pts = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (vol >= lo) & (vol <= hi)
        if m.sum() < 50:
            continue
        rob = 1.4826 * np.median(np.abs(elog[m] - np.median(elog[m])))
        pts.append((float(np.median(vol[m])), math.sqrt(max(rob ** 2 - np.median(gpart[m]), 0.0)),
                    int(m.sum())))
    pv, ps, pw = (np.array(x) for x in zip(*pts, strict=True))
    A = np.column_stack([np.ones_like(pv), np.minimum(pv, 2.0)])
    ab, *_ = np.linalg.lstsq(A * np.sqrt(pw)[:, None], ps * np.sqrt(pw), rcond=None)
    params['base_sigma'] = {'a': float(ab[0]), 'b': float(ab[1]), 'floor': 0.10, 'cap': 0.60,
                            'vol_default': float(np.median(vol)),
                            'deciles': [{'fcf_vol': v, 'sigma': s, 'n': n} for v, s, n in pts]}
    params['fit'] = {
        'n_cases': len(cases), 'n_tickers': len({c['ticker'] for c in cases}),
        'origins': f"{min(c['origin'] for c in cases)}-{max(c['origin'] for c in cases)}",
        'walk_forward_lambda': walk_forward_lambda(cases),
        'note': ('Survivors only (companies listed on the fit date); base rates clipped to '
                 f'{list(growth_v2.BASE_RATE_BOUNDS)}. The analyst part of the v2 path is '
                 'not validated (no point-in-time analyst data).'),
    }
    return params


def print_report(res):
    print(f"\n  GROWTH HINDCAST, target {res['target']}  ({res['n_cases']} cases, {res['n_tickers']} "
          f"tickers, origins {res['origin_range']}; censored "
          f"{res['censored_share']:.1%})")
    print(f"  Estimate vs realised rank IC (per origin, mean): "
          f"{res['all']['estimate_rank_ic']:+.3f}")
    for label, block in (('All origins (ratio vs evidence)', res['all']),
                         ('Walk-forward origins (all rules)', res['walk_forward'])):
        print(f"\n  {label}: {block['n_obs']} obs over "
              f"{len(block['origins'])} origins — NOT independent")
        print(f"    {'Rule':<12s} {'CRPS':>7s} {'in 50%':>7s} {'in 90%':>7s}"
              f" {'meanPIT':>8s} {'bias':>7s} {'sigma':>7s}")
        for r, v in block['rules'].items():
            if v:
                print(f"    {r:<12s} {v['crps']:>7.4f} {v['in_50']:>7.0%} "
                      f"{v['in_90']:>7.0%} {v['mean_pit']:>8.2f} "
                      f"{v['mean_error']:>+7.3f} {v['median_sigma']:>7.3f}")
    print("\n  Nominal: 50% / 90% inside, mean PIT 0.50. Lower CRPS is better.")


def run(rows, target='fcf'):
    cases = build_cases(rows, target)
    if not cases:
        raise SystemExit("no cases: rows carry no usable edgar_history")
    coefs = forecasts(cases)
    origins = sorted({c['origin'] for c in cases})
    return {
        'target': target,
        'n_cases': len(cases),
        'n_tickers': len({c['ticker'] for c in cases}),
        'origin_range': f"{origins[0]}-{origins[-1]}",
        'censored_share': sum(c['censored'] for c in cases) / len(cases),
        'all': score(cases, ('ratio', 'evidence')),
        'walk_forward': score(cases),
        'fitted_coefficients': {str(k): v for k, v in coefs.items()},
    }


def _load_rows(path):
    from data.snapshot_store import read_snapshot
    data = read_snapshot(path)
    return data['results'] if isinstance(data, dict) else data


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--snapshot', help='results JSON (default: newest)')
    ap.add_argument('--results-dir', default='output')
    ap.add_argument('--out', default='output/growth_hindcast.json')
    ap.add_argument('--target', choices=TARGETS, default='fcf')
    ap.add_argument('--fit-v2', metavar='PATH',
                    help='fit the v2 growth model and write its params JSON '
                         '(models/growth_v2_params.json)')
    ap.add_argument('--all-rows', action='store_true',
                    help='every row with history, not only DCF rows')
    a = ap.parse_args(argv)
    path = a.snapshot
    if not path:
        from data.snapshot_store import list_snapshot_files
        files = list_snapshot_files(a.results_dir)
        if not files:
            raise SystemExit(f"no snapshots in {a.results_dir}")
        path = files[-1][1]
    rows = _load_rows(path)
    if not a.all_rows:
        rows = [r for r in rows if isinstance(r.get('mc_quantiles'), list)]
    if a.fit_v2:
        params = fit_v2(rows)
        params['fit']['snapshot'] = os.path.basename(str(path))
        with open(a.fit_v2, 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=1, sort_keys=True)
            f.write('\n')
        print(json.dumps({k: params[k] for k in ('base_rates', 'shrink_lambda',
                                                 'growth_sigma', 'base_sigma')}, indent=1))
        print(json.dumps(params['fit']['walk_forward_lambda'], indent=1))
        print(f"\n  wrote {a.fit_v2}")
        return
    res = run(rows, a.target)
    res['snapshot'] = os.path.basename(str(path))
    print_report(res)
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == '__main__':
    main()
