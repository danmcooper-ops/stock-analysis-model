# models/ddm.py
import warnings as _py_warnings

import numpy as np

from models.montecarlo import (
    DEFAULT_SEED,
    correlate, normal_from_uniform, sobol_uniforms, truncated_normal_from_uniform,
)
from models.valuation_types import Valuation, _validate_count, _validate_numeric

# Monte Carlo constraint walls. MC_MIN_SPREAD mirrors two_stage_ddm's
# min_spread; MC_RE_FLOOR is a plausibility bound on the cost of equity.
MC_RE_FLOOR = 0.03
MC_MIN_SPREAD = 0.02


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


def _d0_from_dps(dps, growth, dps_is_forward):
    """Return the base-year dividend D0 for a given DPS input.

    When `dps_is_forward` is True the caller is passing a forward annual
    rate (e.g. yfinance `dividendRate` = last declared payment × frequency),
    which is already the year-1 dividend D1. Backing it out by one year of
    growth means the first projected dividend equals the declared rate
    instead of being compounded a second time, which overstated every DDM
    leg by roughly (1 + g).
    """
    if not dps_is_forward:
        return dps
    # Floor the back-out growth at the deterministic models' −50% bound so
    # an unclipped Monte Carlo growth draw can't flip or explode D0.
    return dps / (1 + np.maximum(growth, -0.50))


def two_stage_ddm_valuation(dps, high_g, term_g, re, years=5,
                            dps_is_forward=False):
    """Two-stage Dividend Discount Model, as a Valuation envelope.

    Stage 1: project DPS at constant `high_g` for `years` years.
    Stage 2: terminal value via Gordon Growth Model at `term_g`.
    value is None on invalid inputs, with the reason in `warnings`.

    `dps` is the trailing (base-year, D0) dividend by default. Pass
    `dps_is_forward=True` when it is a forward annual rate (D1): the
    year-1 dividend is then `dps` itself rather than `dps × (1 + high_g)`.
    """
    method = 'two_stage_ddm'
    try:
        dps = _validate_numeric('dps', dps, positive=True)
        re = _validate_numeric('re', re, positive=True, low=0.01, high=0.40)
        term_g = _validate_numeric('term_g', term_g, low=-0.10, high=0.10)
        high_g = _validate_numeric('high_g', high_g, low=-0.50, high=1.0)
    except ValueError as e:
        _py_warnings.warn(f"two_stage_ddm input invalid: {e}", RuntimeWarning, stacklevel=3)
        return Valuation.invalid(method, f'input invalid: {e}')
    d0 = _d0_from_dps(dps, high_g, dps_is_forward)
    inputs = {'dps': dps, 'high_g': high_g, 'term_g': term_g, 're': re,
              'years': years, 'dps_is_forward': bool(dps_is_forward), 'd0': d0}
    if re <= term_g:
        return Valuation.invalid(
            method, 're <= term_g — Gordon terminal value undefined', inputs)

    # Minimum spread guard
    min_spread = 0.02
    effective_tg = term_g
    if re - term_g < min_spread:
        effective_tg = re - min_spread

    # Stage 1: PV of projected dividends
    pv_divs = 0.0
    projected_div = d0
    for yr in range(1, years + 1):
        projected_div = projected_div * (1 + high_g)
        pv_divs += projected_div / (1 + re) ** yr

    # Stage 2: Terminal value (Gordon Growth on last projected dividend)
    terminal_div = projected_div * (1 + effective_tg)
    terminal_value = terminal_div / (re - effective_tg)
    pv_terminal = terminal_value / (1 + re) ** years

    value = pv_divs + pv_terminal
    if value <= 0:
        return Valuation.invalid(method, 'non-positive present value', inputs)
    if term_g != effective_tg:
        inputs['effective_terminal_growth'] = effective_tg
    return Valuation(value=value, method=method, confidence=1.0,
                     warnings=(), inputs_used=inputs)


def two_stage_ddm(dps, high_g, term_g, re, years=5, dps_is_forward=False):
    """Legacy float|None wrapper around two_stage_ddm_valuation()."""
    return two_stage_ddm_valuation(dps, high_g, term_g, re, years=years,
                                   dps_is_forward=dps_is_forward).value


def ddm_h_model_valuation(dps, short_g, long_g, re, half_life=5,
                          dps_is_forward=False):
    """H-Model (linear growth decline) closed-form DDM, as a Valuation envelope.

    V = D0 × (1 + long_g) / (re - long_g) + D0 × H × (short_g - long_g) / (re - long_g)
    where H = half_life (half the period over which growth linearly declines).

    `dps` is D0 by default. With `dps_is_forward=True` it is treated as the
    forward (year-1) rate and backed out by one year of `short_g`, the
    growth the H-model applies in its first year.
    """
    method = 'ddm_h_model'
    try:
        dps = _validate_numeric('dps', dps, positive=True)
        re = _validate_numeric('re', re, positive=True, low=0.01, high=0.40)
        long_g = _validate_numeric('long_g', long_g, low=-0.10, high=0.10)
        short_g = _validate_numeric('short_g', short_g, low=-0.50, high=1.0)
    except ValueError as e:
        _py_warnings.warn(f"ddm_h_model input invalid: {e}", RuntimeWarning, stacklevel=3)
        return Valuation.invalid(method, f'input invalid: {e}')
    d0 = _d0_from_dps(dps, short_g, dps_is_forward)
    inputs = {'dps': dps, 'short_g': short_g, 'long_g': long_g, 're': re,
              'half_life': half_life, 'dps_is_forward': bool(dps_is_forward),
              'd0': d0}
    if re <= long_g:
        return Valuation.invalid(
            method, 're <= long_g — stable leg undefined', inputs)

    # Clamp the spread and substitute the effective long-growth EVERYWHERE
    # (numerator included), mirroring two_stage_ddm — clamping only the
    # denominator would value the stable leg on a growth rate the spread no
    # longer reflects.
    effective_long_g = long_g
    if re - long_g < 0.02:
        effective_long_g = re - 0.02
    spread = re - effective_long_g

    # Stable component
    stable_value = d0 * (1 + effective_long_g) / spread
    # Growth premium
    growth_premium = d0 * half_life * (short_g - effective_long_g) / spread

    value = stable_value + growth_premium
    if value <= 0:
        return Valuation.invalid(method, 'non-positive present value', inputs)
    if long_g != effective_long_g:
        inputs['effective_long_g'] = effective_long_g
    return Valuation(value=value, method=method, confidence=1.0,
                     warnings=(), inputs_used=inputs)


def ddm_h_model(dps, short_g, long_g, re, half_life=5, dps_is_forward=False):
    """Legacy float|None wrapper around ddm_h_model_valuation()."""
    return ddm_h_model_valuation(dps, short_g, long_g, re,
                                 half_life=half_life,
                                 dps_is_forward=dps_is_forward).value


def monte_carlo_ddm(dps, g, re, tg, n=1000,
                    g_sigma=None, re_sigma=0.01, tg_sigma=0.005,
                    years=5, re_tg_corr=0.5, seed=None, dps_is_forward=False):
    """Vectorized quasi-Monte Carlo simulation for DDM fair value.

    Same sampling scheme as monte_carlo_dcf (scrambled Sobol; g ~ normal;
    re ~ normal truncated at MC_RE_FLOOR; tg correlated with re at
    `re_tg_corr` and clipped to re - MC_MIN_SPREAD, the substitution
    two_stage_ddm applies). Pass seed=seed_from_ticker(ticker) for
    independent draws per ticker; None keeps the historical fixed seed.

    With `dps_is_forward=True` the year-1 dividend is pinned at `dps` in
    every sample (a declared forward rate carries no growth uncertainty);
    the sampled growth applies from year 2 onward.

    Returns dict with median_fv, mean_fv, p10_fv, p90_fv, std_fv, cv,
    n_valid, n_iterations, invalid_rate and the constraint diagnostics
    re_floor_rate, tg_wall_rate and clip_rate (the larger of the two).

    Inputs are validated with the same bounds as two_stage_ddm_valuation
    and re <= tg returns None, so the simulation can never report a
    distribution around a point estimate that is itself undefined.
    Returns None on invalid inputs or too few valid iterations.
    """
    try:
        dps = _validate_numeric('dps', dps, positive=True)
        re = _validate_numeric('re', re, positive=True, low=0.01, high=0.40)
        tg = _validate_numeric('tg', tg, low=-0.10, high=0.10)
        g = _validate_numeric('g', g, low=-0.50, high=1.0)
        n = _validate_count('n', n)
        years = _validate_count('years', years)
        re_tg_corr = _validate_numeric('re_tg_corr', re_tg_corr, low=-1.0, high=1.0)
        if re <= tg:
            raise ValueError(f"re <= tg — Gordon terminal value undefined (re={re}, tg={tg})")
    except ValueError as e:
        _py_warnings.warn(f"monte_carlo_ddm input invalid: {e}", RuntimeWarning, stacklevel=2)
        return None

    if g_sigma is None:
        g_sigma = abs(g) * 0.30 if g != 0 else 0.02

    seed = DEFAULT_SEED if seed is None else int(seed)
    u = sobol_uniforms(n, 3, seed)
    g_samples = normal_from_uniform(u[:, 0], g, max(g_sigma, 0.001))

    # Cost of equity: exact truncated normal at the plausibility floor.
    re_sig = max(re_sigma, 0.001)
    re_samples, re_floor_rate = truncated_normal_from_uniform(
        u[:, 1], re, re_sig, lower=MC_RE_FLOOR)

    # Terminal growth: correlated with the cost of equity, then clipped to the
    # 2% minimum spread to MATCH two_stage_ddm's min_spread. The old 1% wall
    # let clipped draws capitalise the terminal dividend on a spread up to 2x
    # tighter than the point estimate ever uses, which pushed the MC
    # median/p90 far above the point FV for exactly the low-Re payers the
    # clip_rate flags (same defect fixed earlier in monte_carlo_dcf).
    z_re = (re_samples - re) / re_sig
    z_tg = correlate(z_re, normal_from_uniform(u[:, 2], 0.0, 1.0), re_tg_corr)
    tg_samples = tg + max(tg_sigma, 0.001) * z_tg
    tg_wall = re_samples - MC_MIN_SPREAD
    n_tg_clipped = int(np.sum(tg_samples > tg_wall))
    tg_samples = np.minimum(tg_samples, tg_wall)

    # --- Vectorized dividend projection: shape (n, years) ---
    projected = np.empty((n, years))
    # Per-sample D0 so that D1 == dps exactly when dps is a forward rate.
    prev = _d0_from_dps(np.full(n, dps), g_samples, dps_is_forward)
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
    tg_wall_rate = n_tg_clipped / n
    clip_rate = max(re_floor_rate, tg_wall_rate)
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
        'seed': seed,
        'invalid_rate': invalid_rate,
        'clip_rate': clip_rate,
        're_floor_rate': re_floor_rate,
        'tg_wall_rate': tg_wall_rate,
    }
