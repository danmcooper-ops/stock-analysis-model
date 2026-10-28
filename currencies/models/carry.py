"""Carry pillar — what does holding this currency pay, net of inflation?

Carry is the return earned for simply holding a currency: the interest rate
differential against the funding currency (USD here). It is the most reliable
positive-expectancy signal in FX, and also the one most prone to blowing up,
which is why this module reports both raw carry and carry per unit of risk.

Three deliberate choices:

**Real, not nominal, is the headline.** A 37% policy rate against 35%
inflation is not a 37% carry — it is roughly break-even, and the currency is
usually depreciating to match. Nominal carry alone systematically ranks
high-inflation currencies at the top, which is precisely the trap that makes
naive carry strategies lose money. Turkey is the live example in this
universe.

**Carry is measured against USD**, the base currency, so USD's own carry
gates are structurally inapplicable (a currency cannot yield a spread against
itself). Handled by the scoring layer's `_appl_not_base`.

**Carry-to-vol, not raw carry, for sizing.** 4% carry on a currency with 6%
annualized volatility is a very different proposition from 4% on one with 25%.
"""

from models import series


def real_rate(policy_rate_pct, inflation_pct):
    """Ex-post real policy rate in percent, or None if either leg is missing.

    Uses the simple difference rather than the Fisher form
    ((1+i)/(1+pi) - 1). At double-digit rates the two diverge materially, but
    the simple form is what practitioners quote and what the gate thresholds
    below are calibrated against. The divergence is monotonic, so the ranking
    is unaffected.
    """
    if policy_rate_pct is None or inflation_pct is None:
        return None
    return policy_rate_pct - inflation_pct


def compute(policy_rate_pct, inflation_pct, base_policy_rate_pct,
            base_inflation_pct, vol_1y):
    """Carry metrics, all in percent except carry_to_vol (a ratio).

    Args:
        policy_rate_pct: this currency's central bank policy rate.
        inflation_pct: this currency's latest annual CPI inflation.
        base_policy_rate_pct / base_inflation_pct: the same for USD.
        vol_1y: annualized realized volatility as a *fraction* (0.09 = 9%).
    """
    nominal_carry = None
    if policy_rate_pct is not None and base_policy_rate_pct is not None:
        nominal_carry = policy_rate_pct - base_policy_rate_pct

    own_real = real_rate(policy_rate_pct, inflation_pct)
    base_real = real_rate(base_policy_rate_pct, base_inflation_pct)

    real_carry = None
    if own_real is not None and base_real is not None:
        real_carry = own_real - base_real

    # Risk-adjusted carry. Guard the denominator: a pegged currency can post
    # near-zero realized volatility, which would otherwise divide a modest
    # carry into an enormous ratio and rank the peg top of the book.
    carry_to_vol = None
    if nominal_carry is not None and vol_1y and vol_1y > 0.01:
        carry_to_vol = (nominal_carry / 100.0) / vol_1y

    return {
        'policy_rate': policy_rate_pct,
        'real_policy_rate': own_real,
        'nominal_carry': nominal_carry,
        'real_carry': real_carry,
        'carry_to_vol': carry_to_vol,
    }
