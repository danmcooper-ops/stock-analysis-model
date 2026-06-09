# models/dcf.py
import warnings as _py_warnings

import numpy as np

from models.valuation_types import _validate_numeric


def _dcf_validate_core(base_fcf, growth_rate, discount_rate, terminal_growth):
    """Shared input validation for DCF entry points.

    Returns clean floats or raises ValueError. Bounds chosen wide enough
    to accept any sensible input but reject obvious garbage (NaN, inf,
    growth = 200%, negative discount rate).
    """
    base_fcf = _validate_numeric('base_fcf', base_fcf, positive=True)
    growth_rate = _validate_numeric('growth_rate', growth_rate, low=-0.5, high=1.0)
    discount_rate = _validate_numeric('discount_rate', discount_rate,
                                      positive=True, low=0.01, high=0.50)
    terminal_growth = _validate_numeric('terminal_growth', terminal_growth,
                                        low=-0.10, high=0.10)
    return base_fcf, growth_rate, discount_rate, terminal_growth


def two_stage_ev(base_fcf, growth_rate, discount_rate, terminal_growth,
                 total_years=10, stage1_years=5, min_spread=0.025):
    """
    Two-stage DCF enterprise value.
    Stage 1 (years 1–stage1_years): constant `growth_rate`.
    Stage 2 (years stage1_years+1–total_years): linear fade to `terminal_growth`.
    Gordon Growth terminal value applied after final year.

    Returns None on invalid inputs or when discount_rate <= terminal_growth.
    Emits Python warnings for soft issues (initial growth > 25%, negative
    growth, terminal-value share > 70% of EV) so they surface in logs.
    """
    try:
        base_fcf, growth_rate, discount_rate, terminal_growth = _dcf_validate_core(
            base_fcf, growth_rate, discount_rate, terminal_growth)
    except ValueError as e:
        _py_warnings.warn(f"two_stage_ev input invalid: {e}", RuntimeWarning)
        return None

    if discount_rate <= terminal_growth:
        return None

    effective_tg = terminal_growth
    if discount_rate - terminal_growth < min_spread:
        effective_tg = discount_rate - min_spread

    if growth_rate > 0.25:
        _py_warnings.warn(
            f'Initial growth {growth_rate:.1%} is aggressive — sustained >25% is rare',
            RuntimeWarning,
        )
    if growth_rate < 0:
        _py_warnings.warn(
            f'Negative growth ({growth_rate:.1%}) — verify this reflects secular decline, '
            'not a transient dip',
            RuntimeWarning,
        )

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
    ev = pv_fcfs + pv_terminal

    tv_share = pv_terminal / ev if ev > 0 else 1.0
    if tv_share > 0.80:
        _py_warnings.warn(
            f'Terminal value is {tv_share:.0%} of EV — result rests almost entirely '
            'on terminal assumptions',
            RuntimeWarning,
        )

    return ev


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
    Returns None on invalid inputs.
    """
    try:
        base_fcf, growth_rate, discount_rate, terminal_growth = _dcf_validate_core(
            base_fcf, growth_rate, discount_rate, terminal_growth)
        base_ebitda = _validate_numeric('base_ebitda', base_ebitda, positive=True)
        exit_multiple = _validate_numeric('exit_multiple', exit_multiple,
                                          positive=True, low=3.0, high=40.0)
    except ValueError as e:
        _py_warnings.warn(f"two_stage_ev_exit_multiple input invalid: {e}", RuntimeWarning)
        return None

    if discount_rate <= terminal_growth:
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
    ev = pv_fcfs + pv_terminal

    tv_share = pv_terminal / ev if ev > 0 else 1.0
    if tv_share > 0.80:
        _py_warnings.warn(
            f'Terminal value is {tv_share:.0%} of EV — exit-multiple assumption drives result',
            RuntimeWarning,
        )
    return ev


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
    invalid_rate, clip_rate — the last two surface the upward bias from
    constraint-flooring so callers can judge whether the median is
    trustworthy. Returns None on invalid inputs or too few valid iterations.
    """
    try:
        base_fcf, growth_rate, discount_rate, terminal_growth = _dcf_validate_core(
            base_fcf, growth_rate, discount_rate, terminal_growth)
        shares_outstanding = _validate_numeric('shares_outstanding',
                                               shares_outstanding, positive=True)
    except ValueError as e:
        _py_warnings.warn(f"monte_carlo_dcf input invalid: {e}", RuntimeWarning)
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

    # Track how many samples we have to wall-clip — this is real signal
    # of how violently the model is being forced to behave.
    n_w_clipped = int(np.sum(w_samples < 0.03))
    w_samples = np.maximum(w_samples, 0.03)
    n_tg_clipped = int(np.sum(tg_samples > w_samples - 0.01))
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
    invalid_rate = 1.0 - n_valid / n
    clip_rate = (n_w_clipped + n_tg_clipped) / (2 * n)
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
        # New diagnostics — surface the upward bias from clipping/filtering.
        'invalid_rate': invalid_rate,
        'clip_rate': clip_rate,
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
