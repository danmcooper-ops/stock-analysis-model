# models/dcf.py
import warnings as _py_warnings

import numpy as np

from models.montecarlo import (
    DEFAULT_SEED,
    correlate, lognormal_from_uniform, normal_from_uniform, sobol_uniforms,
    truncated_normal_from_uniform,
)
from models.valuation_types import Valuation, _validate_count, _validate_numeric

# Monte Carlo constraint walls. MC_MIN_SPREAD mirrors two_stage_ev's default
# min_spread so a clipped draw is valued exactly the way the point estimate
# values a too-tight spread; MC_WACC_FLOOR is a plausibility bound on the
# discount rate (the point estimate has no such floor).
MC_WACC_FLOOR = 0.03
MC_MIN_SPREAD = 0.025


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


def _validate_years(total_years, stage1_years):
    """Reject year parameters that would corrupt the projection loops.

    Non-integer or non-positive years break the vectorized Monte Carlo
    array shapes; stage1 > total silently disables the fade stage.
    """
    for name, v in (('total_years', total_years), ('stage1_years', stage1_years)):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            raise ValueError(f"{name} must be an integer, got {v!r}")
    if total_years < 1:
        raise ValueError(f"total_years must be >= 1, got {total_years}")
    if not (0 <= stage1_years <= total_years):
        raise ValueError(
            f"stage1_years must be in [0, total_years], got "
            f"stage1_years={stage1_years}, total_years={total_years}")


def _confidence_from_warnings(warns):
    """Heuristic confidence for a Valuation: 1.0 minus 0.15 per soft
    warning, floored at 0.4. Invalid inputs use Valuation.invalid (0.0)."""
    return max(0.4, 1.0 - 0.15 * len(warns))


def two_stage_ev_valuation(base_fcf, growth_rate, discount_rate, terminal_growth,
                           total_years=10, stage1_years=5, min_spread=0.025):
    """
    Two-stage DCF enterprise value, returned as a Valuation envelope.
    Stage 1 (years 1–stage1_years): constant `growth_rate`.
    Stage 2 (years stage1_years+1–total_years): linear fade to `terminal_growth`.
    Gordon Growth terminal value applied after final year.

    value is None on invalid inputs or when discount_rate <= terminal_growth,
    with the reason in `warnings`. Soft issues (initial growth > 25%, negative
    growth, terminal-value share > 80% of EV) are still emitted as Python
    warnings AND recorded on the envelope so downstream consumers see them
    without scraping logs.
    """
    method = 'two_stage_ev_ggm'
    warns = []

    def _warn(msg):
        warns.append(msg)
        _py_warnings.warn(msg, RuntimeWarning, stacklevel=3)

    try:
        base_fcf, growth_rate, discount_rate, terminal_growth = _dcf_validate_core(
            base_fcf, growth_rate, discount_rate, terminal_growth)
        _validate_years(total_years, stage1_years)
    except ValueError as e:
        _py_warnings.warn(f"two_stage_ev input invalid: {e}", RuntimeWarning, stacklevel=3)
        return Valuation.invalid(method, f'input invalid: {e}')

    inputs = {'base_fcf': base_fcf, 'growth_rate': growth_rate,
              'discount_rate': discount_rate, 'terminal_growth': terminal_growth,
              'total_years': total_years, 'stage1_years': stage1_years}

    if discount_rate <= terminal_growth:
        return Valuation.invalid(
            method, 'discount_rate <= terminal_growth — Gordon terminal value undefined',
            inputs)

    effective_tg = terminal_growth
    if discount_rate - terminal_growth < min_spread:
        effective_tg = discount_rate - min_spread
        inputs['effective_terminal_growth'] = effective_tg

    if growth_rate > 0.25:
        _warn(f'Initial growth {growth_rate:.1%} is aggressive — sustained >25% is rare')
    if growth_rate < 0:
        _warn(f'Negative growth ({growth_rate:.1%}) — verify this reflects secular decline, '
              'not a transient dip')

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
        _warn(f'Terminal value is {tv_share:.0%} of EV — result rests almost entirely '
              'on terminal assumptions')

    return Valuation(value=ev, method=method,
                     confidence=_confidence_from_warnings(warns),
                     warnings=tuple(warns), inputs_used=inputs)


def two_stage_ev(base_fcf, growth_rate, discount_rate, terminal_growth,
                 total_years=10, stage1_years=5, min_spread=0.025):
    """Legacy float|None wrapper around two_stage_ev_valuation()."""
    return two_stage_ev_valuation(
        base_fcf, growth_rate, discount_rate, terminal_growth,
        total_years=total_years, stage1_years=stage1_years,
        min_spread=min_spread).value


def fair_value_per_share(enterprise_value, net_debt, shares_outstanding):
    """Equity value per share = (EV - Net Debt) / Shares."""
    if enterprise_value is None or shares_outstanding is None or shares_outstanding <= 0:
        return None
    equity_value = enterprise_value - (net_debt or 0)
    if equity_value <= 0:
        return None
    return equity_value / shares_outstanding


def two_stage_ev_exit_multiple_valuation(base_fcf, growth_rate, discount_rate,
                                         terminal_growth, base_ebitda, exit_multiple,
                                         total_years=10, stage1_years=5,
                                         min_spread=0.025):
    """Two-stage DCF with EV/EBITDA exit multiple, as a Valuation envelope.

    FCF projection identical to two_stage_ev().
    Terminal Value = Year 10 EBITDA × exit_multiple (instead of Gordon Growth).
    value is None on invalid inputs, with the reason in `warnings`.
    """
    method = 'two_stage_ev_exit_multiple'
    warns = []

    def _warn(msg):
        warns.append(msg)
        _py_warnings.warn(msg, RuntimeWarning, stacklevel=3)

    try:
        base_fcf, growth_rate, discount_rate, terminal_growth = _dcf_validate_core(
            base_fcf, growth_rate, discount_rate, terminal_growth)
        base_ebitda = _validate_numeric('base_ebitda', base_ebitda, positive=True)
        exit_multiple = _validate_numeric('exit_multiple', exit_multiple,
                                          positive=True, low=3.0, high=40.0)
        _validate_years(total_years, stage1_years)
    except ValueError as e:
        _py_warnings.warn(f"two_stage_ev_exit_multiple input invalid: {e}", RuntimeWarning, stacklevel=3)
        return Valuation.invalid(method, f'input invalid: {e}')

    inputs = {'base_fcf': base_fcf, 'growth_rate': growth_rate,
              'discount_rate': discount_rate, 'terminal_growth': terminal_growth,
              'base_ebitda': base_ebitda, 'exit_multiple': exit_multiple,
              'total_years': total_years, 'stage1_years': stage1_years}

    if discount_rate <= terminal_growth:
        return Valuation.invalid(
            method, 'discount_rate <= terminal_growth — fade target undefined', inputs)

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
        _warn(f'Terminal value is {tv_share:.0%} of EV — exit-multiple assumption drives result')

    return Valuation(value=ev, method=method,
                     confidence=_confidence_from_warnings(warns),
                     warnings=tuple(warns), inputs_used=inputs)


def blend_fair_value_legs(fv_ggm, fv_exit):
    """Combine the GGM and exit-multiple per-share legs into the headline
    fair value: the average when both are available, otherwise whichever
    one is. Shared by run_forward_dcf and reverse_dcf so the reverse solve
    targets the SAME number the forward model reports.
    """
    if fv_ggm and fv_exit:
        return (fv_ggm + fv_exit) / 2.0
    return fv_ggm or fv_exit


def two_stage_ev_exit_multiple(base_fcf, growth_rate, discount_rate,
                               terminal_growth, base_ebitda, exit_multiple,
                               total_years=10, stage1_years=5, min_spread=0.025):
    """Legacy float|None wrapper around two_stage_ev_exit_multiple_valuation()."""
    return two_stage_ev_exit_multiple_valuation(
        base_fcf, growth_rate, discount_rate, terminal_growth,
        base_ebitda, exit_multiple, total_years=total_years,
        stage1_years=stage1_years, min_spread=min_spread).value


def monte_carlo_dcf(base_fcf, growth_rate, discount_rate, terminal_growth,
                    net_debt, shares_outstanding,
                    base_ebitda=None, exit_multiple=None,
                    n_iterations=1000, growth_sigma=None,
                    wacc_sigma=0.01, tg_sigma=0.005,
                    exit_mult_sigma=None, exit_mult_floor=3.0,
                    total_years=10, stage1_years=5,
                    wacc_tg_corr=0.5, seed=None):
    """Vectorized quasi-Monte Carlo simulation over DCF parameters.

    Scenarios come from a scrambled Sobol sequence (see models.montecarlo):
    growth_rate ~ normal; discount_rate ~ normal truncated at MC_WACC_FLOOR;
    terminal_growth ~ normal correlated with the discount rate at
    `wacc_tg_corr`, then clipped to discount_rate - MC_MIN_SPREAD (the same
    substitution two_stage_ev applies); exit_multiple ~ log-normal around the
    point multiple, floored at `exit_mult_floor`. For each scenario the fair
    value uses GGM + exit-multiple terminal values (averaged when both are
    available). `seed` selects the sequence — pass seed_from_ticker(ticker)
    so tickers get independent draws; None keeps the historical fixed seed.

    Returns dict with median_fv, mean_fv, p10_fv, p90_fv, std_fv, cv,
    n_valid, n_iterations, invalid_rate and the constraint diagnostics
    wacc_floor_rate (probability mass the discount-rate floor excluded),
    tg_wall_rate (share of draws clipped to the minimum spread),
    exit_floor_rate (share of exit multiples floored; None without the exit
    leg) and clip_rate = the largest of those, i.e. how hard the most
    binding wall was forcing the model. Returns None on invalid inputs or
    too few valid iterations.

    The exit-multiple leg runs only when BOTH base_ebitda (> 0) and
    exit_multiple (> 0) are supplied; a non-finite or non-positive
    exit_multiple, or a non-finite base_ebitda, drops the leg with a
    RuntimeWarning rather than silently valuing it at the floor multiple.
    """
    try:
        base_fcf, growth_rate, discount_rate, terminal_growth = _dcf_validate_core(
            base_fcf, growth_rate, discount_rate, terminal_growth)
        shares_outstanding = _validate_numeric('shares_outstanding',
                                               shares_outstanding, positive=True)
        # None = leverage unknown, treated as 0 (matches fair_value_per_share).
        net_debt = 0.0 if net_debt is None else _validate_numeric('net_debt', net_debt)
        n = _validate_count('n_iterations', n_iterations)
        wacc_tg_corr = _validate_numeric('wacc_tg_corr', wacc_tg_corr, low=-1.0, high=1.0)
        _validate_years(total_years, stage1_years)
    except ValueError as e:
        _py_warnings.warn(f"monte_carlo_dcf input invalid: {e}", RuntimeWarning, stacklevel=2)
        return None

    has_exit = False
    if base_ebitda is not None and exit_multiple is not None:
        try:
            base_ebitda = _validate_numeric('base_ebitda', base_ebitda)
            exit_multiple = _validate_numeric('exit_multiple', exit_multiple, positive=True)
        except ValueError as e:
            _py_warnings.warn(f"monte_carlo_dcf exit-multiple leg skipped: {e}",
                              RuntimeWarning, stacklevel=2)
        else:
            has_exit = base_ebitda > 0

    if growth_sigma is None:
        growth_sigma = abs(growth_rate) * 0.30 if growth_rate != 0 else 0.02
    if exit_mult_sigma is None and has_exit:
        exit_mult_sigma = exit_multiple * 0.15

    # Four Sobol dimensions regardless of the exit leg, so the GGM draws are
    # identical with and without it (a dropped exit leg = the GGM-only run).
    seed = DEFAULT_SEED if seed is None else int(seed)
    u = sobol_uniforms(n, 4, seed)
    g_samples = normal_from_uniform(u[:, 0], growth_rate, max(growth_sigma, 0.001))

    # Discount rate: exact truncated normal at the plausibility floor. The
    # excluded mass is real signal of how hard the model is being forced.
    w_sigma = max(wacc_sigma, 0.001)
    w_samples, wacc_floor_rate = truncated_normal_from_uniform(
        u[:, 1], discount_rate, w_sigma, lower=MC_WACC_FLOOR)

    # Terminal growth: correlated with the discount rate, then clipped to the
    # 2.5% minimum spread to MATCH the point estimate's min_spread
    # (two_stage_ev). A looser wall let clipped draws compute terminal values
    # on a tighter spread than the point estimate ever uses, biasing the MC
    # median/p90 above the point FV in exactly the clip-heavy cases.
    z_w = (w_samples - discount_rate) / w_sigma
    z_tg = correlate(z_w, normal_from_uniform(u[:, 2], 0.0, 1.0), wacc_tg_corr)
    tg_samples = terminal_growth + max(tg_sigma, 0.001) * z_tg
    tg_wall = w_samples - MC_MIN_SPREAD
    n_tg_clipped = int(np.sum(tg_samples > tg_wall))
    tg_samples = np.minimum(tg_samples, tg_wall)

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

    equity_ggm = ev_ggm - net_debt
    fv_ggm = np.where(equity_ggm > 0, equity_ggm / shares_outstanding, 0)

    # --- Exit multiple terminal value (if available) ---
    fv_exit = np.zeros(n)
    exit_floor_rate = None
    if has_exit:
        # Log-normal around the point multiple: never negative, right-skewed
        # like observed multiples. The floor stays as the pipeline's rule
        # (EXIT_MULT_MIN) and now only binds for genuinely low multiples.
        em_samples = lognormal_from_uniform(u[:, 3], exit_multiple,
                                            max(exit_mult_sigma or 1.0, 0.5))
        n_em_floored = int(np.sum(em_samples < exit_mult_floor))
        exit_floor_rate = n_em_floored / n
        em_samples = np.maximum(em_samples, exit_mult_floor)

        # EBITDA follows the same growth path as FCF, so terminal EBITDA is
        # the base scaled by the FCF path's cumulative growth.
        terminal_ebitda = base_ebitda * projected[:, -1] / base_fcf
        tv_exit = terminal_ebitda * em_samples
        pv_tv_exit = tv_exit / (1 + w_samples) ** total_years
        ev_exit = pv_fcfs + pv_tv_exit
        equity_exit = ev_exit - net_debt
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
    tg_wall_rate = n_tg_clipped / n
    clip_rate = max(wacc_floor_rate, tg_wall_rate, exit_floor_rate or 0.0)
    if n_valid < n * 0.10:
        return None

    # Percentiles over ALL draws — insolvent outcomes are clamped to 0 above,
    # so p10 reflects genuine wipeout risk instead of conditioning on survival
    # (the old survivors-only p10 dropped exactly the bear scenarios a p10 is
    # meant to capture). Central tendency is over all draws for consistency.
    mean_fv = float(np.mean(fv_combined))
    std_fv = float(np.std(fv_combined))
    return {
        'median_fv': float(np.median(fv_combined)),
        'mean_fv': mean_fv,
        'p10_fv': float(np.percentile(fv_combined, 10)),
        'p90_fv': float(np.percentile(fv_combined, 90)),
        'std_fv': std_fv,
        'cv': float(std_fv / mean_fv) if mean_fv > 0 else None,
        'n_valid': n_valid,
        'n_iterations': n,
        'seed': seed,
        # Constraint diagnostics — how hard the walls were forcing the model.
        'invalid_rate': invalid_rate,
        'clip_rate': clip_rate,
        'wacc_floor_rate': wacc_floor_rate,
        'tg_wall_rate': tg_wall_rate,
        'exit_floor_rate': exit_floor_rate,
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
                growth_range=(-0.30, 0.30), tol=1e-6, max_iter=80,
                base_ebitda=None, exit_multiple=None):
    """Solve for the implied growth rate that makes DCF fair value equal
    the market price.

    Uses bisection (no scipy dependency) on the same per-share fair value
    run_forward_dcf reports: the GGM leg, plus the exit-multiple leg when
    `base_ebitda` and `exit_multiple` are supplied, blended by
    blend_fair_value_legs. Pass the forward model's own inputs (adjusted
    base FCF, EV→equity bridge as `net_debt`, EBITDA, multiple) so the
    implied growth is comparable with its estimated growth.

    The default range spans decline as well as growth: the forward
    estimate floors at -15%, so a solve floored at 0% blanked the implied
    growth for every stock priced for shrinkage.
    """
    if (price is None or price <= 0 or fcf is None or fcf <= 0 or
            wacc is None or wacc <= 0 or
            shares_outstanding is None or shares_outstanding <= 0):
        return None

    use_exit = (base_ebitda is not None and base_ebitda > 0
                and exit_multiple is not None and exit_multiple > 0)

    def _fv_at_growth(g):
        # Solver probes are not user inputs: silence the aggressive-growth /
        # negative-growth advisories the model would otherwise emit on every
        # bisection step.
        with _py_warnings.catch_warnings():
            _py_warnings.simplefilter('ignore', RuntimeWarning)
            ev = two_stage_ev(fcf, g, wacc, terminal_g, total_years, stage1_years)
            fv_ggm = fair_value_per_share(ev, net_debt, shares_outstanding)
            fv_exit = None
            if use_exit:
                ev_exit = two_stage_ev_exit_multiple(
                    fcf, g, wacc, terminal_g, base_ebitda, exit_multiple,
                    total_years, stage1_years)
                fv_exit = fair_value_per_share(ev_exit, net_debt, shares_outstanding)
        fv = blend_fair_value_legs(fv_ggm, fv_exit)
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

    # Exhausted max_iter without meeting tolerance — report non-convergence.
    return {'implied_growth': (lo + hi) / 2.0, 'converged': False}
