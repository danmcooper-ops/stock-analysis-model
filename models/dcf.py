# models/dcf.py
import numpy as np


def two_stage_ev(base_fcf, growth_rate, discount_rate, terminal_growth,
                 total_years=10, stage1_years=5, min_spread=0.025):
    """
    Two-stage DCF enterprise value.
    Stage 1 (years 1–stage1_years): constant `growth_rate`.
    Stage 2 (years stage1_years+1–total_years): linear fade to `terminal_growth`.
    Gordon Growth terminal value applied after final year.
    """
    if base_fcf is None or base_fcf <= 0 or discount_rate <= terminal_growth:
        return None
    effective_tg = terminal_growth
    if discount_rate - terminal_growth < min_spread:
        effective_tg = discount_rate - min_spread

    projected = []
    prev = base_fcf
    for yr in range(1, total_years + 1):
        if yr <= stage1_years:
            g = growth_rate
        else:
            fade = (yr - stage1_years) / (total_years - stage1_years)
            g = growth_rate + (effective_tg - growth_rate) * fade
        prev = prev * (1 + g)
        projected.append(prev)

    pv_fcfs = sum(fcf / (1 + discount_rate) ** t
                  for t, fcf in enumerate(projected, 1))
    terminal_fcf = projected[-1] * (1 + effective_tg)
    terminal_value = terminal_fcf / (discount_rate - effective_tg)
    pv_terminal = terminal_value / (1 + discount_rate) ** total_years
    return pv_fcfs + pv_terminal


def fair_value_per_share(enterprise_value, net_debt, shares_outstanding):
    """Equity value per share = (EV - Net Debt) / Shares."""
    if enterprise_value is None or shares_outstanding is None or shares_outstanding <= 0:
        return None
    equity_value = enterprise_value - (net_debt or 0)
    if equity_value <= 0:
        return None
    return equity_value / shares_outstanding


def two_stage_ev_exit_multiple(base_fcf, growth_rate, discount_rate,
                               terminal_growth, base_ebitda, exit_multiple,
                               total_years=10, stage1_years=5, min_spread=0.025):
    """Two-stage DCF with EV/EBITDA exit multiple for terminal value.

    FCF projection identical to two_stage_ev().
    Terminal Value = Year 10 EBITDA × exit_multiple (instead of Gordon Growth).
    """
    if (base_fcf is None or base_fcf <= 0 or base_ebitda is None or
            base_ebitda <= 0 or exit_multiple is None or discount_rate <= terminal_growth):
        return None
    effective_tg = terminal_growth
    if discount_rate - terminal_growth < min_spread:
        effective_tg = discount_rate - min_spread

    # Project FCFs (same as GGM version)
    projected_fcf = []
    prev_fcf = base_fcf
    for yr in range(1, total_years + 1):
        if yr <= stage1_years:
            g = growth_rate
        else:
            fade = (yr - stage1_years) / (total_years - stage1_years)
            g = growth_rate + (effective_tg - growth_rate) * fade
        prev_fcf = prev_fcf * (1 + g)
        projected_fcf.append(prev_fcf)

    pv_fcfs = sum(fcf / (1 + discount_rate) ** t
                  for t, fcf in enumerate(projected_fcf, 1))

    # Project EBITDA forward with same growth pattern
    prev_ebitda = base_ebitda
    for yr in range(1, total_years + 1):
        if yr <= stage1_years:
            g = growth_rate
        else:
            fade = (yr - stage1_years) / (total_years - stage1_years)
            g = growth_rate + (effective_tg - growth_rate) * fade
        prev_ebitda = prev_ebitda * (1 + g)

    terminal_value = prev_ebitda * exit_multiple
    pv_terminal = terminal_value / (1 + discount_rate) ** total_years
    return pv_fcfs + pv_terminal


def monte_carlo_dcf(base_fcf, growth_rate, discount_rate, terminal_growth,
                    net_debt, shares_outstanding,
                    base_ebitda=None, exit_multiple=None,
                    n_iterations=1000, growth_sigma=None,
                    wacc_sigma=0.01, tg_sigma=0.005,
                    exit_mult_sigma=None,
                    total_years=10, stage1_years=5):
    """Vectorized Monte Carlo simulation over DCF parameters.

    Samples growth_rate, discount_rate, terminal_growth, and exit_multiple
    from normal distributions. For each sample, computes fair value using
    GGM + exit multiple terminal values (averaged when both available).

    Returns dict with median_fv, p10_fv, p90_fv, std_fv, cv, n_valid,
    or None if too few valid iterations.
    """
    if base_fcf is None or base_fcf <= 0 or shares_outstanding is None or shares_outstanding <= 0:
        return None

    rng = np.random.default_rng(42)  # fixed seed for reproducibility

    if growth_sigma is None:
        growth_sigma = abs(growth_rate) * 0.30 if growth_rate != 0 else 0.02
    if exit_mult_sigma is None and exit_multiple:
        exit_mult_sigma = exit_multiple * 0.15

    n = n_iterations
    g_samples = rng.normal(growth_rate, max(growth_sigma, 0.001), n)
    w_samples = rng.normal(discount_rate, max(wacc_sigma, 0.001), n)
    tg_samples = rng.normal(terminal_growth, max(tg_sigma, 0.001), n)

    # Enforce constraints: WACC > 3%, WACC - TG > 1%
    w_samples = np.maximum(w_samples, 0.03)
    tg_samples = np.minimum(tg_samples, w_samples - 0.01)

    # --- Vectorized FCF projection: shape (n, total_years) ---
    projected = np.empty((n, total_years))
    prev = np.full(n, base_fcf)
    for yr in range(total_years):
        yr1 = yr + 1  # 1-indexed
        if yr1 <= stage1_years:
            g = g_samples
        else:
            fade = (yr1 - stage1_years) / (total_years - stage1_years)
            g = g_samples + (tg_samples - g_samples) * fade
        prev = prev * (1 + g)
        projected[:, yr] = prev

    # Discount factors: shape (n, total_years)
    years_arr = np.arange(1, total_years + 1)
    disc_factors = (1 + w_samples[:, np.newaxis]) ** years_arr[np.newaxis, :]
    pv_fcfs = np.sum(projected / disc_factors, axis=1)

    # --- GGM terminal value ---
    terminal_fcf = projected[:, -1] * (1 + tg_samples)
    spreads = w_samples - tg_samples
    tv_ggm = np.where(spreads > 0.005, terminal_fcf / spreads, 0)
    pv_tv_ggm = tv_ggm / (1 + w_samples) ** total_years
    ev_ggm = pv_fcfs + pv_tv_ggm

    equity_ggm = ev_ggm - (net_debt or 0)
    fv_ggm = np.where(equity_ggm > 0, equity_ggm / shares_outstanding, 0)

    # --- Exit multiple terminal value (if available) ---
    fv_exit = np.zeros(n)
    has_exit = (base_ebitda is not None and base_ebitda > 0 and
                exit_multiple is not None)
    if has_exit:
        em_samples = rng.normal(exit_multiple, max(exit_mult_sigma or 1.0, 0.5), n)
        em_samples = np.maximum(em_samples, 3.0)  # floor at 3x

        # Project EBITDA with same growth pattern
        prev_ebitda = np.full(n, base_ebitda)
        for yr in range(total_years):
            yr1 = yr + 1
            if yr1 <= stage1_years:
                g = g_samples
            else:
                fade = (yr1 - stage1_years) / (total_years - stage1_years)
                g = g_samples + (tg_samples - g_samples) * fade
            prev_ebitda = prev_ebitda * (1 + g)

        tv_exit = prev_ebitda * em_samples
        pv_tv_exit = tv_exit / (1 + w_samples) ** total_years
        ev_exit = pv_fcfs + pv_tv_exit
        equity_exit = ev_exit - (net_debt or 0)
        fv_exit = np.where(equity_exit > 0, equity_exit / shares_outstanding, 0)

    # --- Average methods ---
    if has_exit:
        both_valid = (fv_ggm > 0) & (fv_exit > 0)
        fv_combined = np.where(both_valid, (fv_ggm + fv_exit) / 2,
                               np.where(fv_ggm > 0, fv_ggm, fv_exit))
    else:
        fv_combined = fv_ggm

    valid = fv_combined > 0
    n_valid = int(np.sum(valid))
    if n_valid < n * 0.10:
        return None

    fv_valid = fv_combined[valid]
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


def dcf_sensitivity(base_fcf, growth_rate, base_wacc, base_terminal_growth,
                    net_debt, shares_outstanding, years=10, stage1=5):
    """
    Worksheet Step 5A sensitivity table: WACC ±1% vs terminal growth ±0.5%.
    Uses two-stage DCF (consistent with main valuation).
    Returns dict keyed by (wacc_delta, growth_delta) -> fair value per share.
    """
    wacc_deltas = [-0.01, -0.005, 0.0, 0.005, 0.01]
    growth_deltas = [-0.005, -0.0025, 0.0, 0.0025, 0.005]
    table = {}
    for dw in wacc_deltas:
        for dg in growth_deltas:
            w = base_wacc + dw
            g = base_terminal_growth + dg
            ev = two_stage_ev(base_fcf, growth_rate, w, g, years, stage1) if w > g else None
            fv = fair_value_per_share(ev, net_debt, shares_outstanding)
            table[(round(dw, 4), round(dg, 4))] = fv
    return table


def reverse_dcf(price, fcf, wacc, shares_outstanding, net_debt=0,
                terminal_g=0.03, total_years=10, stage1_years=5,
                growth_range=(0.0, 0.30), tol=1e-6, max_iter=80):
    """Solve for implied growth rate that makes DCF fair value equal market price.

    Uses bisection (no scipy dependency) to find the growth rate g such that
    fair_value_per_share(two_stage_ev(fcf, g, wacc, terminal_g), ...) == price.

    Parameters
    ----------
    price : float
        Current market price per share.
    fcf : float
        Base free cash flow (total, not per-share).
    wacc : float
        Weighted average cost of capital.
    shares_outstanding : float
        Shares outstanding.
    net_debt : float
        Net debt (Total Debt - Cash).
    terminal_g : float
        Terminal growth rate for Gordon Growth.
    growth_range : tuple
        (low, high) bounds for growth search.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum bisection iterations.

    Returns
    -------
    dict or None
        {'implied_growth': float, 'converged': bool} or None if inputs invalid.
    """
    if (price is None or price <= 0 or fcf is None or fcf <= 0 or
            wacc is None or wacc <= 0 or
            shares_outstanding is None or shares_outstanding <= 0):
        return None

    def _fv_at_growth(g):
        ev = two_stage_ev(fcf, g, wacc, terminal_g, total_years, stage1_years)
        fv = fair_value_per_share(ev, net_debt, shares_outstanding)
        return fv if fv is not None else 0.0

    lo, hi = growth_range
    fv_lo = _fv_at_growth(lo) - price
    fv_hi = _fv_at_growth(hi) - price

    # If both same sign, implied growth is outside range
    if fv_lo * fv_hi > 0:
        # If FV at low growth already exceeds price, implied growth < lo
        if fv_lo > 0:
            return {'implied_growth': lo, 'converged': False}
        # If FV at high growth still below price, implied growth > hi
        return {'implied_growth': hi, 'converged': False}

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        fv_mid = _fv_at_growth(mid) - price
        if abs(fv_mid) < tol or (hi - lo) / 2.0 < tol:
            return {'implied_growth': mid, 'converged': True}
        if fv_mid * fv_lo > 0:
            lo = mid
            fv_lo = fv_mid
        else:
            hi = mid

    return {'implied_growth': (lo + hi) / 2.0, 'converged': True}
