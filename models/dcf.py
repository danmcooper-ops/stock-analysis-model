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


def calculate_dcf(free_cash_flows, discount_rate, terminal_value, periods=None):
    if discount_rate <= 0:
        return None
    if periods is None:
        periods = len(free_cash_flows)
    discounted_fcf = [fcf / (1 + discount_rate) ** t for t, fcf in enumerate(free_cash_flows, 1)]
    discounted_terminal = terminal_value / (1 + discount_rate) ** periods
    return sum(discounted_fcf) + discounted_terminal
