# models/rim.py
"""Residual Income Model (RIM) — book value plus present value of excess earnings.

Value = Book Value + PV of Residual Income stream + Terminal Value.
Residual Income = Book Value * (ROE - cost_of_equity).
Useful complement to DCF: anchored to book value rather than free cash flow.
"""
import warnings as _py_warnings

import numpy as np

from models.valuation_types import _validate_numeric


def residual_income_model(book_value_per_share, roe, cost_of_equity,
                          g=0.03, years=10, retention_ratio=None):
    """Residual Income Model: BV + PV of excess earnings.

    Value = BV + sum(RI_t / (1+Re)^t) + TV
    where RI_t = BV_t-1 * (ROE - Re), BV_t = BV_t-1 * (1 + ROE × retention)
    Terminal value uses Gordon Growth on final RI.

    Parameters
    ----------
    book_value_per_share, roe, cost_of_equity, g, years
        Standard RIM inputs.
    retention_ratio : float or None
        Fraction of earnings retained (i.e. not paid out as dividend or
        buyback). Drives how fast book value compounds. When None
        (default), it is inferred from g/ROE — the Gordon-consistent
        retention rate that produces the stated long-term growth — and
        a warning is emitted. Pass an explicit value to override.

        The earlier implementation silently assumed all earnings were
        retained (book grew at fixed `g` regardless of payout), which
        inflated terminal-period book value for any firm that paid a
        dividend or bought back stock. The new default is closer to
        the truth for most firms; pass `retention_ratio=1.0` only if
        the company genuinely reinvests every dollar.
    """
    if book_value_per_share is None or book_value_per_share <= 0:
        return None
    try:
        roe = _validate_numeric('roe', roe, low=-1.0, high=2.0)
        cost_of_equity = _validate_numeric('cost_of_equity', cost_of_equity,
                                           positive=True, low=0.01, high=0.40)
        g = _validate_numeric('g', g, low=-0.05, high=0.10)
    except ValueError as e:
        _py_warnings.warn(f"residual_income_model input invalid: {e}", RuntimeWarning)
        return None
    if cost_of_equity <= g:
        return None

    # Resolve retention ratio
    if retention_ratio is None:
        if roe > 0:
            retention_ratio = max(min(g / roe, 1.0), 0.0)
            _py_warnings.warn(
                f'retention_ratio not provided — inferred {retention_ratio:.2f} '
                'from g/ROE (Gordon-consistent). Pass an explicit value if you '
                'have payout data.',
                RuntimeWarning,
            )
        else:
            retention_ratio = 1.0
            _py_warnings.warn(
                'retention_ratio not provided and ROE <= 0 — falling back to '
                '100% retention. Book-value evolution is suspect.',
                RuntimeWarning,
            )
    retention_ratio = max(min(retention_ratio, 1.0), 0.0)

    bv = book_value_per_share
    pv_ri = 0.0

    for t in range(1, years + 1):
        ri = bv * (roe - cost_of_equity)
        pv_ri += ri / (1 + cost_of_equity) ** t
        # Book value grows by retained earnings.
        bv = bv * (1 + roe * retention_ratio)

    # Terminal value: RI continues growing at g
    ri_terminal = bv * (roe - cost_of_equity)
    if ri_terminal > 0 and (cost_of_equity - g) > 0.005:
        tv = ri_terminal * (1 + g) / (cost_of_equity - g)
        pv_tv = tv / (1 + cost_of_equity) ** years
    else:
        pv_tv = 0.0

    if abs(roe - cost_of_equity) < 0.005:
        _py_warnings.warn(
            'ROE ≈ cost_of_equity — RIM degenerates toward book value '
            '(residual income near zero)',
            RuntimeWarning,
        )

    intrinsic = book_value_per_share + pv_ri + pv_tv
    return intrinsic if intrinsic > 0 else None
