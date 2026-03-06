# models/dcf.py
import numpy as np


def project_fcf(historical_fcfs, projection_years=5, growth_rate=None):
    positive_fcfs = [f for f in historical_fcfs if f > 0]

    if growth_rate is None:
        if len(positive_fcfs) >= 2:
            cagr = (positive_fcfs[-1] / positive_fcfs[0]) ** (1 / (len(positive_fcfs) - 1)) - 1
            growth_rate = max(-0.20, min(0.30, cagr))
        else:
            growth_rate = 0.0

    # Use most recent positive FCF as the projection base
    if historical_fcfs[-1] > 0:
        base_fcf = historical_fcfs[-1]
    elif positive_fcfs:
        base_fcf = positive_fcfs[-1]
    else:
        return None

    projected = [base_fcf * (1 + growth_rate) ** t for t in range(1, projection_years + 1)]

    return {
        'projected_fcfs': projected,
        'base_fcf': base_fcf,
        'growth_rate': growth_rate,
        'projection_years': projection_years,
    }


def calculate_terminal_value(final_fcf, discount_rate, terminal_growth_rate, min_spread=0.01):
    if final_fcf <= 0:
        return {'terminal_value': 0.0, 'effective_growth_rate': terminal_growth_rate, 'was_clamped': False}

    effective_tgr = terminal_growth_rate
    if discount_rate - terminal_growth_rate < min_spread:
        effective_tgr = discount_rate - min_spread

    tv = final_fcf * (1 + effective_tgr) / (discount_rate - effective_tgr)

    return {
        'terminal_value': tv,
        'effective_growth_rate': effective_tgr,
        'was_clamped': effective_tgr != terminal_growth_rate,
    }


def dcf_sensitivity(historical_fcfs, base_discount_rate, terminal_growth_rate,
                    projection_years=5, dr_spread=0.02, gr_spread=0.02):
    base_proj = project_fcf(historical_fcfs, projection_years=projection_years)
    if base_proj is None:
        return None

    base_gr = base_proj['growth_rate']
    growth_rates = [base_gr - gr_spread, base_gr, base_gr + gr_spread]
    discount_rates = [base_discount_rate - dr_spread, base_discount_rate, base_discount_rate + dr_spread]

    # Clamp growth rates to valid bounds
    growth_rates = [max(-0.20, min(0.30, g)) for g in growth_rates]
    discount_rates = [d for d in discount_rates if d > 0]
    if not discount_rates:
        return None

    matrix = {}
    for dr in discount_rates:
        for gr in growth_rates:
            proj = project_fcf(historical_fcfs, projection_years=projection_years, growth_rate=gr)
            if proj is None:
                continue
            tv = calculate_terminal_value(proj['projected_fcfs'][-1], dr, terminal_growth_rate)
            ev = calculate_dcf(proj['projected_fcfs'], dr, tv['terminal_value'])
            if ev is not None:
                matrix[(dr, gr)] = ev

    if not matrix:
        return None

    values = list(matrix.values())
    return {
        'matrix': matrix,
        'low': min(values),
        'base': matrix.get((base_discount_rate, base_gr)),
        'high': max(values),
        'discount_rates': sorted(set(dr for dr, _ in matrix)),
        'growth_rates': sorted(set(gr for _, gr in matrix)),
    }


def calculate_dcf(free_cash_flows, discount_rate, terminal_value, periods=None):
    if discount_rate <= 0:
        return None
    if periods is None:
        periods = len(free_cash_flows)
    discounted_fcf = [fcf / (1 + discount_rate) ** t for t, fcf in enumerate(free_cash_flows, 1)]
    discounted_terminal = terminal_value / (1 + discount_rate) ** periods
    return sum(discounted_fcf) + discounted_terminal
