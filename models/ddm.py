# models/ddm.py
import numpy as np


def ddm_eligibility(div_history, payout, eps, dps, min_years=3):
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

    # Payout flag: warn if > 100% but still allow DDM
    if payout is not None and payout > 1.0:
        result['payout_flag'] = True

    result['eligible'] = True
    result['reason'] = 'Eligible'
    return result


def estimate_ddm_growth(div_history, payout, roe, analyst_ltg):
    """Weighted-average dividend growth estimate from three signals.

    Weights: 30% dividend CAGR, 40% sustainable growth (ROE × retention),
    30% analyst long-term growth.

    Parameters
    ----------
    div_history : array-like
        Annual DPS values, oldest first (at least 2 values).
    payout : float or None
        Current payout ratio.
    roe : float or None
        Return on equity.
    analyst_ltg : float or None
        Analyst consensus long-term growth rate.

    Returns
    -------
    dict with keys: growth (float or None), div_cagr (float or None),
    sustainable_growth (float or None), signals_used (int).
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
    if div_history is not None and len(div_history) >= 2:
        hist = [d for d in div_history if d is not None and d > 0]
        if len(hist) >= 2:
            years = len(hist) - 1
            cagr = (hist[-1] / hist[0]) ** (1 / years) - 1
            # Cap at reasonable range
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

    # Signal 3: Analyst LTG (30%)
    if analyst_ltg is not None and analyst_ltg > 0:
        ltg = max(min(analyst_ltg, 0.25), 0.0)
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

    Parameters
    ----------
    dps : float
        Current annual dividend per share.
    high_g : float
        High-growth rate for stage 1.
    term_g : float
        Terminal (perpetual) growth rate.
    re : float
        Required return (cost of equity).
    years : int
        Number of high-growth years.

    Returns
    -------
    float or None
        Intrinsic value per share.
    """
    if dps is None or dps <= 0:
        return None
    if re is None or re <= term_g:
        return None
    if re <= 0:
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

    Parameters
    ----------
    dps : float
        Current annual dividend per share.
    short_g : float
        Initial (short-term) high growth rate.
    long_g : float
        Long-term stable growth rate.
    re : float
        Required return (cost of equity).
    half_life : int or float
        Years for growth to decline halfway (H in the formula).

    Returns
    -------
    float or None
        Intrinsic value per share.
    """
    if dps is None or dps <= 0:
        return None
    if re is None or re <= long_g:
        return None
    if re <= 0:
        return None

    spread = re - long_g
    if spread < 0.02:
        spread = 0.02

    # Stable component
    stable_value = dps * (1 + long_g) / spread

    # Growth premium
    growth_premium = dps * half_life * (short_g - long_g) / spread

    value = stable_value + growth_premium
    return value if value > 0 else None


def monte_carlo_ddm(dps, g, re, tg, n=1000,
                    g_sigma=None, re_sigma=0.01, tg_sigma=0.005,
                    years=5):
    """Vectorized Monte Carlo simulation for DDM fair value.

    Samples growth, cost-of-equity, and terminal growth from normal
    distributions. For each sample, computes two-stage DDM value.

    Parameters
    ----------
    dps : float
        Current annual dividend per share.
    g : float
        Base high-growth rate.
    re : float
        Base required return (cost of equity).
    tg : float
        Base terminal growth rate.
    n : int
        Number of iterations.
    g_sigma : float or None
        Std dev for growth sampling. Defaults to 30% of |g| or 0.02.
    re_sigma : float
        Std dev for cost-of-equity sampling.
    tg_sigma : float
        Std dev for terminal growth sampling.
    years : int
        High-growth years.

    Returns
    -------
    dict or None
        Keys: median_fv, mean_fv, p10_fv, p90_fv, std_fv, cv, n_valid, n_iterations.
    """
    if dps is None or dps <= 0:
        return None

    rng = np.random.default_rng(42)  # fixed seed for reproducibility

    if g_sigma is None:
        g_sigma = abs(g) * 0.30 if g != 0 else 0.02

    g_samples = rng.normal(g, max(g_sigma, 0.001), n)
    re_samples = rng.normal(re, max(re_sigma, 0.001), n)
    tg_samples = rng.normal(tg, max(tg_sigma, 0.001), n)

    # Enforce constraints: re > 3%, re - tg > 1%
    re_samples = np.maximum(re_samples, 0.03)
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
    }
