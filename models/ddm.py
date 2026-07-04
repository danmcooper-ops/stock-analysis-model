# models/ddm.py
import warnings as _py_warnings

import numpy as np

from models.valuation_types import _validate_numeric


def ddm_eligibility(div_history, payout, eps, dps, min_years=3, strict_payout=False):
    """Check whether a stock qualifies for DDM valuation.

    Parameters
    ----------
    div_history : array-like or None
        Annual dividend-per-share values, oldest first.
    payout : float or None
        Current payout ratio (0–1+).
    eps : float or None
        Trailing EPS.
    dps : float or None
        Current annual dividend per share.
    min_years : int
        Minimum consecutive years of positive dividends required.
    strict_payout : bool
        When True, payout > 100% disqualifies the stock outright instead
        of just flagging it. Default False preserves the old behavior;
        set True when downstream confidence handling can't compensate
        for a dividend that isn't earnings-funded.

    Returns
    -------
    dict with keys: eligible (bool), reason (str),
    consecutive_years (int), payout_flag (bool).
    """
    result = {
        'eligible': False,
        'reason': '',
        'consecutive_years': 0,
        'payout_flag': False,
    }

    if dps is None or dps <= 0:
        result['reason'] = 'No current dividend'
        return result

    if eps is None or eps <= 0:
        result['reason'] = 'Non-positive EPS'
        return result

    if div_history is None or len(div_history) == 0:
        result['reason'] = 'No dividend history'
        return result

    # Count consecutive positive years from most recent backward
    consecutive = 0
    for d in reversed(list(div_history)):
        if d is not None and d > 0:
            consecutive += 1
        else:
            break
    result['consecutive_years'] = consecutive

    if consecutive < min_years:
        result['reason'] = f'Only {consecutive} consecutive years (need {min_years})'
        return result

    # Payout flag: warn if > 100% (and disqualify when strict_payout is set)
    if payout is not None and payout > 1.0:
        result['payout_flag'] = True
        if strict_payout:
            result['reason'] = (
                f'Payout {payout:.0%} > 100% — dividend not earnings-funded'
            )
            return result

    result['eligible'] = True
    result['reason'] = 'Eligible'
    return result


def estimate_ddm_growth(div_history, payout, roe, analyst_ltg):
    """Weighted-average dividend growth estimate from three signals.

    Weights: 30% dividend CAGR, 40% sustainable growth (ROE × retention),
    30% analyst long-term growth.
    """
    result = {
        'growth': None,
        'div_cagr': None,
        'sustainable_growth': None,
        'signals_used': 0,
    }

    weighted_sum = 0.0
    total_weight = 0.0

    # Signal 1: Dividend CAGR (30%)
    # Measure the span between the first and last POSITIVE years by their
    # original positions, so suspension years (reindexed to 0 upstream) count
    # toward the time base instead of being filtered out and compressing it.
    if div_history is not None and len(div_history) >= 2:
        positives = [(i, d) for i, d in enumerate(div_history)
                     if d is not None and d > 0]
        if len(positives) >= 2:
            first_i, first_d = positives[0]
            last_i, last_d = positives[-1]
            years = last_i - first_i
            if years >= 1:
                cagr = (last_d / first_d) ** (1 / years) - 1
                cagr = max(min(cagr, 0.25), -0.10)
                result['div_cagr'] = cagr
                weighted_sum += 0.30 * cagr
                total_weight += 0.30
                result['signals_used'] += 1

    # Signal 2: Sustainable growth = ROE × (1 - payout) (40%)
    if roe is not None and payout is not None and roe > 0 and 0 < payout < 1.0:
        sustainable = roe * (1 - payout)
        sustainable = max(min(sustainable, 0.25), 0.0)
        result['sustainable_growth'] = sustainable
        weighted_sum += 0.40 * sustainable
        total_weight += 0.40
        result['signals_used'] += 1

    # Signal 3: Analyst LTG (30%). Allow a NEGATIVE forecast through (clamped
    # at −10%) rather than discarding it — silently dropping a bearish analyst
    # estimate biased the blend upward for deteriorating payers.
    if analyst_ltg is not None:
        ltg = max(min(analyst_ltg, 0.25), -0.10)
        weighted_sum += 0.30 * ltg
        total_weight += 0.30
        result['signals_used'] += 1

    if total_weight > 0:
        result['growth'] = weighted_sum / total_weight
    return result


def two_stage_ddm(dps, high_g, term_g, re, years=5):
    """Two-stage Dividend Discount Model.

    Stage 1: project DPS at constant `high_g` for `years` years.
    Stage 2: terminal value via Gordon Growth Model at `term_g`.
    Returns None on invalid inputs.
    """
    try:
        dps = _validate_numeric('dps', dps, positive=True)
        re = _validate_numeric('re', re, positive=True, low=0.01, high=0.40)
        term_g = _validate_numeric('term_g', term_g, low=-0.10, high=0.10)
        high_g = _validate_numeric('high_g', high_g, low=-0.50, high=1.0)
    except ValueError as e:
        _py_warnings.warn(f"two_stage_ddm input invalid: {e}", RuntimeWarning)
        return None
    if re <= term_g:
        return None

    # Minimum spread guard
    min_spread = 0.02
    effective_tg = term_g
    if re - term_g < min_spread:
        effective_tg = re - min_spread

    # Stage 1: PV of projected dividends
    pv_divs = 0.0
    projected_div = dps
    for yr in range(1, years + 1):
        projected_div = projected_div * (1 + high_g)
        pv_divs += projected_div / (1 + re) ** yr

    # Stage 2: Terminal value (Gordon Growth on last projected dividend)
    terminal_div = projected_div * (1 + effective_tg)
    terminal_value = terminal_div / (re - effective_tg)
    pv_terminal = terminal_value / (1 + re) ** years

    value = pv_divs + pv_terminal
    return value if value > 0 else None


def ddm_h_model(dps, short_g, long_g, re, half_life=5):
    """H-Model (linear growth decline) closed-form DDM.

    V = D0 × (1 + long_g) / (re - long_g) + D0 × H × (short_g - long_g) / (re - long_g)
    where H = half_life (half the period over which growth linearly declines).
    """
    try:
        dps = _validate_numeric('dps', dps, positive=True)
        re = _validate_numeric('re', re, positive=True, low=0.01, high=0.40)
        long_g = _validate_numeric('long_g', long_g, low=-0.10, high=0.10)
        short_g = _validate_numeric('short_g', short_g, low=-0.50, high=1.0)
    except ValueError as e:
        _py_warnings.warn(f"ddm_h_model input invalid: {e}", RuntimeWarning)
        return None
    if re <= long_g:
        return None

    # Clamp the spread and substitute the effective long-growth EVERYWHERE
    # (numerator included), mirroring two_stage_ddm — clamping only the
    # denominator would value the stable leg on a growth rate the spread no
    # longer reflects.
    effective_long_g = long_g
    if re - long_g < 0.02:
        effective_long_g = re - 0.02
    spread = re - effective_long_g

    # Stable component
    stable_value = dps * (1 + effective_long_g) / spread
    # Growth premium
    growth_premium = dps * half_life * (short_g - effective_long_g) / spread

    value = stable_value + growth_premium
    return value if value > 0 else None


def monte_carlo_ddm(dps, g, re, tg, n=1000,
                    g_sigma=None, re_sigma=0.01, tg_sigma=0.005,
                    years=5):
    """Vectorized Monte Carlo simulation for DDM fair value.

    Returns dict with median_fv, mean_fv, p10_fv, p90_fv, std_fv, cv,
    n_valid, n_iterations, invalid_rate, clip_rate. The last two surface
    how aggressively the constraint walls clipped samples and how many
    iterations produced non-positive value (and were filtered).
    """
    try:
        dps = _validate_numeric('dps', dps, positive=True)
    except ValueError as e:
        _py_warnings.warn(f"monte_carlo_ddm input invalid: {e}", RuntimeWarning)
        return None

    rng = np.random.default_rng(42)  # fixed seed for reproducibility

    if g_sigma is None:
        g_sigma = abs(g) * 0.30 if g != 0 else 0.02

    g_samples = rng.normal(g, max(g_sigma, 0.001), n)
    re_samples = rng.normal(re, max(re_sigma, 0.001), n)
    tg_samples = rng.normal(tg, max(tg_sigma, 0.001), n)

    # Track clip counts before applying constraint walls
    n_re_clipped = int(np.sum(re_samples < 0.03))
    re_samples = np.maximum(re_samples, 0.03)
    n_tg_clipped = int(np.sum(tg_samples > re_samples - 0.01))
    tg_samples = np.minimum(tg_samples, re_samples - 0.01)

    # --- Vectorized dividend projection: shape (n, years) ---
    projected = np.empty((n, years))
    prev = np.full(n, dps)
    for yr in range(years):
        prev = prev * (1 + g_samples)
        projected[:, yr] = prev

    # Discount factors
    years_arr = np.arange(1, years + 1)
    disc_factors = (1 + re_samples[:, np.newaxis]) ** years_arr[np.newaxis, :]
    pv_divs = np.sum(projected / disc_factors, axis=1)

    # Terminal value (Gordon Growth on last projected dividend)
    terminal_div = projected[:, -1] * (1 + tg_samples)
    spreads = re_samples - tg_samples
    tv = np.where(spreads > 0.005, terminal_div / spreads, 0)
    pv_tv = tv / (1 + re_samples) ** years

    fv = pv_divs + pv_tv

    valid = fv > 0
    n_valid = int(np.sum(valid))
    invalid_rate = 1.0 - n_valid / n
    clip_rate = (n_re_clipped + n_tg_clipped) / (2 * n)
    if n_valid < n * 0.10:
        return None

    fv_valid = fv[valid]
    mean_fv = float(np.mean(fv_valid))
    return {
        'median_fv': float(np.median(fv_valid)),
        'mean_fv': mean_fv,
        'p10_fv': float(np.percentile(fv_valid, 10)),
        'p90_fv': float(np.percentile(fv_valid, 90)),
        'std_fv': float(np.std(fv_valid)),
        'cv': float(np.std(fv_valid) / mean_fv) if mean_fv > 0 else None,
        'n_valid': n_valid,
        'n_iterations': n,
        'invalid_rate': invalid_rate,
        'clip_rate': clip_rate,
    }
