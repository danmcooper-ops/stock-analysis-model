# models/epv.py
"""Earnings Power Value (EPV) — zero-growth valuation baseline.

EPV assumes the company can sustain current earnings indefinitely with no growth.
It provides a conservative floor valuation: if a stock trades below EPV, the
market is pricing in *earnings decline*, which is a strong buy signal.
"""
import warnings as _py_warnings

from models.valuation_types import _validate_numeric


def earnings_power_value(ebit, tax_rate, cost_of_capital, shares_outstanding,
                         excess_cash=0):
    """Zero-growth valuation: NOPAT / cost_of_capital + excess cash, per share.

    Returns None on invalid inputs; computes value otherwise.
    """
    try:
        ebit = _validate_numeric('ebit', ebit, positive=True)
        cost_of_capital = _validate_numeric('cost_of_capital', cost_of_capital,
                                            positive=True, low=0.01, high=0.40)
        shares_outstanding = _validate_numeric('shares_outstanding',
                                               shares_outstanding, positive=True)
    except ValueError as e:
        _py_warnings.warn(f"earnings_power_value input invalid: {e}", RuntimeWarning)
        return None

    tax_rate = max(0, min(tax_rate if tax_rate is not None else 0.21, 0.50))
    nopat = ebit * (1 - tax_rate)
    epv = nopat / cost_of_capital + (excess_cash or 0)
    return epv / shares_outstanding if epv > 0 else None


def epv_with_growth_premium(epv_base, roe, cost_of_equity):
    """Growth-adjusted EPV: scales EPV when ROE > cost of equity.

    When ROE exceeds the cost of equity, growth creates value.
    Growth-adjusted EPV = EPV_base * (ROE / cost_of_equity).
    When ROE < cost_of_equity, growth actually destroys value, so we
    return the base EPV as a floor.

    The multiplier cap is 2.0× (tightened from 3.0× in this revision).
    A multiplier near 3× would mean a company's ROE is 3× its cost of
    equity sustained in perpetuity — that's a rare, durable moat
    territory that EPV's zero-growth premise can't model honestly.
    Capping at 2× keeps EPV conservative as the "floor" valuation it
    was designed to be; if a stock genuinely warrants a higher
    premium, DCF or RIM should make the case.

    Emits a warning when ROE > 30% — that range usually signals
    leverage- or buyback-inflated ROE rather than genuine compounding
    quality, and growth-adjusting EPV by it is misleading.
    """
    if epv_base is None or epv_base <= 0:
        return None
    if roe is None or cost_of_equity is None or cost_of_equity <= 0:
        return None
    if roe <= 0:
        return epv_base  # no growth premium for negative ROE

    if roe > 0.30:
        _py_warnings.warn(
            f'ROE {roe:.0%} is very high — often signals leverage or buyback '
            'inflation rather than genuine compounding quality. Cross-check ROIC '
            'before treating the growth-adjusted EPV as a fair-value upgrade.',
            RuntimeWarning,
        )

    multiplier = min(roe / cost_of_equity, 2.0)  # tightened from 3.0
    return epv_base * max(multiplier, 1.0)
