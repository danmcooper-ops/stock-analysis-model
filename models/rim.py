# models/rim.py
"""Residual Income Model (RIM) — book value plus present value of excess earnings.

Value = Book Value + PV of Residual Income stream + Terminal Value.
Residual Income = Book Value * (ROE - cost_of_equity).
Useful complement to DCF: anchored to book value rather than free cash flow.
"""
import numpy as np


def residual_income_model(book_value_per_share, roe, cost_of_equity,
                          g=0.03, years=10):
    """Residual Income Model: BV + PV of excess earnings.

    Value = BV + sum(RI_t / (1+Re)^t) + TV
    where RI_t = BV_t-1 * (ROE - Re), BV_t = BV_t-1 * (1 + ROE * retention)
    Terminal value uses Gordon Growth on final RI.

    Parameters
    ----------
    book_value_per_share : float or None
        Current book value per share.
    roe : float or None
        Return on equity (assumed constant).
    cost_of_equity : float or None
        Required return.
    g : float
        Long-term growth rate for residual income in terminal period.
    years : int
        Explicit forecast horizon.

    Returns
    -------
    float or None
        Intrinsic value per share, or None if inputs invalid.
    """
    if book_value_per_share is None or book_value_per_share <= 0:
        return None
    if roe is None or cost_of_equity is None or cost_of_equity <= 0:
        return None
    if cost_of_equity <= g:
        return None

    bv = book_value_per_share
    pv_ri = 0.0

    for t in range(1, years + 1):
        ri = bv * (roe - cost_of_equity)
        pv_ri += ri / (1 + cost_of_equity) ** t
        # Book value grows by retained earnings (ROE * retention)
        # Simplified: assume all excess earnings retained
        bv = bv * (1 + g)

    # Terminal value: RI continues growing at g
    ri_terminal = bv * (roe - cost_of_equity)
    if ri_terminal > 0 and (cost_of_equity - g) > 0.005:
        tv = ri_terminal * (1 + g) / (cost_of_equity - g)
        pv_tv = tv / (1 + cost_of_equity) ** years
    else:
        pv_tv = 0.0

    intrinsic = book_value_per_share + pv_ri + pv_tv
    return intrinsic if intrinsic > 0 else None
