# models/epv.py
"""Earnings Power Value (EPV) — zero-growth valuation baseline.

EPV assumes the company can sustain current earnings indefinitely with no growth.
It provides a conservative floor valuation: if a stock trades below EPV, the
market is pricing in *earnings decline*, which is a strong buy signal.
"""
import warnings as _py_warnings

from models.valuation_types import Valuation, _validate_numeric

STATUTORY_TAX_RATE = 0.21
# Confidence haircut per recorded caveat (defaulted/clamped tax rate,
# point-in-time EBIT). Floored so a fully-caveated EPV still reads as a
# computed number rather than a rejected one.
_CONFIDENCE_STEP = 0.15
_CONFIDENCE_FLOOR = 0.5


def earnings_power_value_valuation(ebit, tax_rate, cost_of_capital,
                                   shares_outstanding, excess_cash=0,
                                   total_debt=0, ebit_source=None):
    """Zero-growth valuation as a Valuation envelope:
    (NOPAT / cost_of_capital + cash - debt) per share.

    NOPAT capitalized at the cost of capital is an *enterprise* value; the
    equity bridge (add cash, subtract debt) converts it to equity value
    before dividing by shares — without it, levered firms get an EPV
    "floor" overstated by debt-per-share. Callers may pass net debt
    through `total_debt` (negative for net-cash firms) with
    `excess_cash=0`.

    `ebit_source` ('normalized' | 'point' | None) records whether the EBIT
    is a through-cycle figure; a point-in-time EBIT is a caveat on the
    envelope, not a rejection.

    Every input is validated: NaN/inf anywhere yields value None. A missing
    tax rate falls back to the statutory rate and an out-of-band one is
    clamped to [0, 0.50] — both recorded in `warnings`, each lowering
    `confidence` by a fixed step.

    value is None on invalid inputs or a debt-swamped equity bridge, with
    the reason in `warnings`.
    """
    method = 'epv_zero_growth'
    caveats = []
    try:
        ebit = _validate_numeric('ebit', ebit, positive=True)
        cost_of_capital = _validate_numeric('cost_of_capital', cost_of_capital,
                                            positive=True, low=0.01, high=0.40)
        shares_outstanding = _validate_numeric('shares_outstanding',
                                               shares_outstanding, positive=True)
        if tax_rate is None:
            tax_rate = STATUTORY_TAX_RATE
            caveats.append(
                f'tax_rate missing — statutory {STATUTORY_TAX_RATE:.0%} assumed')
        else:
            tax_rate = _validate_numeric('tax_rate', tax_rate)
        # Net debt may legitimately be negative (net-cash firm), so only
        # finiteness is enforced on the bridge terms.
        excess_cash = _validate_numeric('excess_cash', excess_cash or 0)
        total_debt = _validate_numeric('total_debt', total_debt or 0)
    except ValueError as e:
        _py_warnings.warn(f"earnings_power_value input invalid: {e}", RuntimeWarning, stacklevel=3)
        return Valuation.invalid(method, f'input invalid: {e}')

    clamped_tax = max(0.0, min(tax_rate, 0.50))
    if clamped_tax != tax_rate:
        caveats.append(f'tax_rate {tax_rate:.1%} clamped to {clamped_tax:.0%}')
        tax_rate = clamped_tax
    if ebit_source == 'point':
        caveats.append('point-in-time EBIT — no through-cycle margin history '
                       'to normalize')

    inputs = {'ebit': ebit, 'tax_rate': tax_rate,
              'cost_of_capital': cost_of_capital,
              'shares_outstanding': shares_outstanding,
              'excess_cash': excess_cash, 'total_debt': total_debt,
              'ebit_source': ebit_source}
    nopat = ebit * (1 - tax_rate)
    epv = nopat / cost_of_capital + excess_cash - total_debt
    if epv <= 0:
        return Valuation.invalid(
            method, 'equity bridge non-positive — debt exceeds capitalized NOPAT + cash',
            inputs)
    confidence = max(_CONFIDENCE_FLOOR, 1.0 - _CONFIDENCE_STEP * len(caveats))
    return Valuation(value=epv / shares_outstanding, method=method,
                     confidence=confidence, warnings=tuple(caveats),
                     inputs_used=inputs)


def earnings_power_value(ebit, tax_rate, cost_of_capital, shares_outstanding,
                         excess_cash=0, total_debt=0):
    """Legacy float|None wrapper around earnings_power_value_valuation()."""
    return earnings_power_value_valuation(
        ebit, tax_rate, cost_of_capital, shares_outstanding,
        excess_cash=excess_cash, total_debt=total_debt).value


def epv_with_growth_premium(epv_base, return_rate, discount_rate):
    """Growth-adjusted EPV: scales EPV when the return on capital exceeds
    its cost.

    Growth-adjusted EPV = EPV_base * (return_rate / discount_rate), a
    heuristic franchise multiplier. The two rates must share a basis: the
    zero-growth EPV is an enterprise figure capitalized at WACC, so the
    pipeline pairs it with through-cycle ROIC vs WACC (book-equity ROE is
    inflated by leverage and buybacks); ROE vs cost of equity is the
    fallback when ROIC is unavailable. When the return is below its cost,
    growth destroys value, so the base EPV is returned as a floor.

    The multiplier cap is 2.0×. A multiplier near 3× would mean a return
    3× its cost sustained in perpetuity — rare, durable-moat territory
    that EPV's zero-growth premise can't model honestly. Capping at 2×
    keeps EPV conservative as the "floor" valuation it was designed to
    be; if a stock genuinely warrants a higher premium, DCF or RIM should
    make the case.

    Emits a warning when the return exceeds 30% — that range usually
    signals leverage- or buyback-inflated returns (or a tiny invested
    capital base) rather than genuine compounding quality, and
    growth-adjusting EPV by it is misleading.
    """
    if epv_base is None or epv_base <= 0:
        return None
    if return_rate is None or discount_rate is None or discount_rate <= 0:
        return None
    if return_rate != return_rate or discount_rate != discount_rate:  # NaN
        return None
    if return_rate <= 0:
        return epv_base  # no growth premium for a negative return

    if return_rate > 0.30:
        _py_warnings.warn(
            f'return on capital {return_rate:.0%} is very high — often signals '
            'leverage, buyback inflation or a tiny capital base rather than '
            'genuine compounding quality. Cross-check ROIC before treating the '
            'growth-adjusted EPV as a fair-value upgrade.',
            RuntimeWarning,
            stacklevel=2,
        )

    multiplier = min(return_rate / discount_rate, 2.0)
    return epv_base * max(multiplier, 1.0)
