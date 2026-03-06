# models/dcf.py
import numpy as np


def calculate_dcf(free_cash_flows, discount_rate, terminal_value, periods=None):
    """Original DCF: discount historical FCFs + a pre-computed terminal value."""
    if discount_rate <= 0:
        return None
    if periods is None:
        periods = len(free_cash_flows)
    discounted_fcf = [fcf / (1 + discount_rate) ** t for t, fcf in enumerate(free_cash_flows, 1)]
    discounted_terminal = terminal_value / (1 + discount_rate) ** periods
    return sum(discounted_fcf) + discounted_terminal


def project_forward_dcf(base_fcf, growth_rate, discount_rate, terminal_growth,
                         years=5, min_spread=0.01):
    """
    Forward-looking DCF (Worksheet Step 5A).
    Projects FCF for `years` at `growth_rate`, then applies Gordon Growth terminal value.
    Returns enterprise value. Clamps terminal growth to ensure at least `min_spread`
    below discount rate to prevent near-infinite terminal values.
    """
    if base_fcf is None or discount_rate <= terminal_growth:
        return None
    # Guard against near-zero denominator in terminal value
    effective_tg = terminal_growth
    if discount_rate - terminal_growth < min_spread:
        effective_tg = discount_rate - min_spread
    projected = [base_fcf * (1 + growth_rate) ** t for t in range(1, years + 1)]
    pv_fcfs = sum(fcf / (1 + discount_rate) ** t for t, fcf in enumerate(projected, 1))
    terminal_fcf = projected[-1] * (1 + effective_tg)
    terminal_value = terminal_fcf / (discount_rate - effective_tg)
    pv_terminal = terminal_value / (1 + discount_rate) ** years
    return pv_fcfs + pv_terminal


def fair_value_per_share(enterprise_value, net_debt, shares_outstanding):
    """Equity value per share = (EV - Net Debt) / Shares."""
    if enterprise_value is None or shares_outstanding is None or shares_outstanding <= 0:
        return None
    equity_value = enterprise_value - (net_debt or 0)
    if equity_value <= 0:
        return None
    return equity_value / shares_outstanding


def dcf_sensitivity(base_fcf, growth_rate, base_wacc, base_terminal_growth,
                    net_debt, shares_outstanding, years=5):
    """
    Worksheet Step 5A sensitivity table: WACC ±1% vs terminal growth ±0.5%.
    Returns dict keyed by (wacc_delta, growth_delta) -> fair value per share.
    """
    wacc_deltas = [-0.01, -0.005, 0.0, 0.005, 0.01]
    growth_deltas = [-0.005, -0.0025, 0.0, 0.0025, 0.005]
    table = {}
    for dw in wacc_deltas:
        for dg in growth_deltas:
            w = base_wacc + dw
            g = base_terminal_growth + dg
            ev = project_forward_dcf(base_fcf, growth_rate, w, g, years) if w > g else None
            fv = fair_value_per_share(ev, net_debt, shares_outstanding)
            table[(round(dw, 4), round(dg, 4))] = fv
    return table


def ggm_value(dividend_per_share, cost_of_equity, growth_rate):
    """Gordon Growth Model cross-check: V = D1 / (Re - g)."""
    if dividend_per_share is None or cost_of_equity is None:
        return None
    if cost_of_equity <= growth_rate or dividend_per_share <= 0:
        return None
    d1 = dividend_per_share * (1 + growth_rate)
    return d1 / (cost_of_equity - growth_rate)
