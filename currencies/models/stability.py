"""Stability pillar — what does holding this currency cost in risk?

Valuation, carry and momentum all describe expected return. This pillar
describes the variance around it, and is the reason a 4% carry on MXN and a
4% carry on CHF do not score alike.

Realized volatility and drawdown are computed from the ECB daily series, so
unlike the External pillar they are current to the last business day.

A note on pegs. A hard peg posts near-zero realized volatility and near-zero
drawdown, and would sweep this pillar if left unmasked. That is not a real
finding — it measures the anchor's stability, not the currency's, and it
inverts violently on the day a peg breaks. The scoring layer marks these
gates structurally inapplicable for pegged currencies rather than letting
them collect free points here.
"""

from models import series

DAYS_3M = 64
DAYS_1Y = 258
DAYS_3Y = 774


def compute(spot, macro_inflation_vol=None):
    """Risk metrics from the daily USD-per-unit spot series.

    Args:
        spot: sorted [(iso_date, usd_per_unit)].
        macro_inflation_vol: stdev of annual inflation, from external.py.
    """
    vol_1y = series.realized_vol(spot, DAYS_1Y)
    vol_3m = series.realized_vol(spot, DAYS_3M)

    # Ratio of recent to trailing volatility. Above 1 means the currency is
    # currently more turbulent than its own baseline — an early warning that
    # a regime is changing, and one that fires well before the annual macro
    # indicators in the External pillar update.
    vol_ratio = None
    if vol_1y and vol_3m and vol_1y > 0:
        vol_ratio = vol_3m / vol_1y

    return {
        'vol_1y': vol_1y,
        'vol_3m': vol_3m,
        'vol_ratio': vol_ratio,
        'max_dd_3y': series.max_drawdown(spot, DAYS_3Y),
        'inflation_vol': macro_inflation_vol,
    }
