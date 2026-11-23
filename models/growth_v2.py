"""DCF growth model v2: a growth path and honest uncertainty, fitted to
realised outcomes by scripts/growth_hindcast.py --fit-v2.

v1 (the live default, config.DCF_GROWTH_MODEL = 'v1') blends six growth
signals into one constant stage-1 rate with sigma max(0.30·|g|, 2pp). The
growth hindcast (2026-11-23, ~8k company-years) found that rule holds only
7%/18% of realised five-year FCF growth in its 50%/90% bands, that the
trailing FCF CAGR ranks future FCF growth the wrong way (rank IC −0.22),
and that the history-based estimate projects ~20% too little cash. v2:

  long-run rate L   history signals at v2 weights (no FCF CAGR), shrunk
                    toward the sector's base rate: L = λ·H + (1 − λ)·b,
                    clamped to [−15%, cap]
  near-term rate g0 the analyst signals (after the haircut) plus the margin
                    and surprise adjustments, clamped; L when no analyst
                    covers the stock. Every analyst figure is treated as
                    near-term: a genuine 5-year LTG has little long-run skill
                    in the literature, and the 'long-term' slot usually holds
                    a one-year figure (dcf_inputs.analyst_lt_horizon)
  stage-1 path      g0 for years 1-2, then linear to L by year 5 (stage 2
                    still fades L to terminal growth)
  growth sigma      k·exp(β·[1, revenue-growth vol, log revenue, |L|]),
                    the scale fitted by CRPS on realised revenue growth
  base-FCF sigma    log-sd of the starting FCF's level, a + b·FCF volatility,
                    from the five-year cash projection error left after the
                    growth sigma

Parameters live in models/growth_v2_params.json (written by the fit, read
here). The analyst part of the path is NOT validated by the hindcast (no
point-in-time analyst data); everything else is, on surviving companies
only, which is why the base rates are capped (BASE_RATE_BOUNDS).
"""
import json
import math
import os

import numpy as np

PARAMS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'growth_v2_params.json')
HOLD_YEARS = 2                 # near-term rate held for years 1..HOLD_YEARS
STAGE1_YEARS = 5
GROWTH_FLOOR = -0.15           # as v1
BASE_RATE_BOUNDS = (0.0, 0.06)  # survivor-only base rates read 6-14%
_PARAMS = None


def load_params(path=None):
    """The fitted parameters (cached for the default path)."""
    global _PARAMS
    if path is None and _PARAMS is not None:
        return _PARAMS
    with open(path or PARAMS_PATH, encoding='utf-8') as f:
        p = json.load(f)
    if path is None:
        _PARAMS = p
    return p


def clamp(g, cap):
    return min(cap, max(GROWTH_FLOOR, g))


def history_blend(signals, weights):
    """Weighted mean of the history signals present (None when none are)."""
    pairs = [(signals.get(k), w) for k, w in weights.items()
             if w > 0 and signals.get(k) is not None and math.isfinite(signals.get(k))]
    if not pairs:
        return None
    return sum(s * w for s, w in pairs) / sum(w for _, w in pairs)


def base_rate(sector, params):
    br = params['base_rates']
    return br.get(sector or '', br['_all'])


def long_run_rate(history_signals, sector, cap, params):
    """(L, H, b): the shrunk long-run stage-1 rate, the raw history blend
    and the sector base rate."""
    h = history_blend(history_signals, params['history_weights'])
    b = base_rate(sector, params)
    lam = params['shrink_lambda']
    raw = b if h is None else lam * h + (1 - lam) * b
    return clamp(raw, cap), h, b


def stage1_path(g0, long_run, years=STAGE1_YEARS, hold=HOLD_YEARS):
    """Per-year stage-1 growth: g0 held `hold` years, then linear to L."""
    path = []
    for k in range(1, years + 1):
        if k <= hold:
            path.append(g0)
        else:
            path.append(g0 + (long_run - g0) * (k - hold) / (years - hold))
    return path


def path_offsets(path, long_run):
    return tuple(g - long_run for g in path)


def equivalent_rate(path):
    """The constant rate with the same cumulative stage-1 growth."""
    prod = 1.0
    for g in path:
        prod *= 1 + g
    return prod ** (1 / len(path)) - 1 if prod > 0 else min(path)


def growth_sigma(rev_growth_vol, revenue, long_run, params):
    """Growth sigma from the fitted scale model (floored and capped)."""
    m = params['growth_sigma']
    vol = rev_growth_vol if rev_growth_vol is not None and math.isfinite(rev_growth_vol) else m['vol_default']
    logrev = math.log(revenue) if revenue and revenue > 0 else m['logrev_default']
    x = np.array([1.0, min(vol, 1.0), logrev, abs(long_run)])
    s = m['k'] * float(np.exp(x @ np.array(m['beta'])))
    return min(max(s, m['floor']), m['cap'])


def fcf_volatility(values):
    """Std of year-on-year FCF changes over the mean |FCF|, last 6 years.

    Works through negative years (a log change would not). None with fewer
    than 4 points or a zero mean."""
    vals = [float(v) for v in values if v is not None and math.isfinite(v)][-6:]
    if len(vals) < 4:
        return None
    scale = float(np.mean(np.abs(vals)))
    if scale <= 0:
        return None
    return float(np.std(np.diff(vals)) / scale)


def base_sigma(fcf_values, params):
    m = params['base_sigma']
    vol = fcf_volatility(fcf_values)
    if vol is None:
        vol = m['vol_default']
    return min(max(m['a'] + m['b'] * min(vol, 2.0), m['floor']), m['cap'])


def near_term_rate(analyst_signals, analyst_weights, adjustments, long_run, cap):
    """(g0, analyst blend or None): analysts plus adjustments, else L."""
    pairs = [(analyst_signals.get(k), w) for k, w in analyst_weights.items()
             if analyst_signals.get(k) is not None and math.isfinite(analyst_signals.get(k))]
    if not pairs:
        return long_run, None
    a = sum(s * w for s, w in pairs) / sum(w for _, w in pairs)
    return clamp(a + adjustments, cap), a
