# models/epv.py
"""Earnings Power Value (EPV) — zero-growth valuation baseline.

EPV assumes the company can sustain current earnings indefinitely with no growth.
It provides a conservative floor valuation: if a stock trades below EPV, the
market is pricing in *earnings decline*, which is a strong buy signal.
"""


def earnings_power_value(ebit, tax_rate, cost_of_capital, shares_outstanding,
                         excess_cash=0):
    """Zero-growth valuation: NOPAT / cost_of_capital + excess cash, per share.

    Parameters
    ----------
    ebit : float or None
        Operating income (EBIT).
    tax_rate : float or None
        Effective tax rate (0-1). Defaults to 21% if None.
    cost_of_capital : float
        WACC or cost of equity.
    shares_outstanding : float
        Shares outstanding.
    excess_cash : float
        Cash above operating needs (added to equity value).

    Returns
    -------
    float or None
        EPV per share, or None if inputs invalid.
    """
    if ebit is None or ebit <= 0:
        return None
    if cost_of_capital is None or cost_of_capital <= 0:
        return None
    if shares_outstanding is None or shares_outstanding <= 0:
        return None

    tax_rate = max(0, min(tax_rate if tax_rate is not None else 0.21, 0.50))
    nopat = ebit * (1 - tax_rate)
    epv = nopat / cost_of_capital + (excess_cash or 0)
    return epv / shares_outstanding if epv > 0 else None


def epv_with_growth_premium(epv_base, roe, cost_of_equity):
    """Growth-adjusted EPV: scales EPV when ROE > cost of equity.

    When ROE exceeds the cost of equity, growth creates value.
    Growth-adjusted EPV = EPV_base * (ROE / cost_of_equity).
    When ROE < cost_of_equity, growth actually destroys value,
    so we return the base EPV as a floor.

    Parameters
    ----------
    epv_base : float or None
        Base EPV per share from earnings_power_value().
    roe : float or None
        Return on equity.
    cost_of_equity : float
        Cost of equity (required return).

    Returns
    -------
    float or None
    """
    if epv_base is None or epv_base <= 0:
        return None
    if roe is None or cost_of_equity is None or cost_of_equity <= 0:
        return None
    if roe <= 0:
        return epv_base  # no growth premium for negative ROE
    # Cap the multiplier to avoid extreme values
    multiplier = min(roe / cost_of_equity, 3.0)
    return epv_base * max(multiplier, 1.0)
