"""Momentum pillar — is the currency already moving in the direction of value?

Trend is the standard companion to valuation in cross-asset work: valuation
says what should happen, momentum says whether it has started. All measures
are on USD-per-unit spot, so positive always means the currency appreciated.

The 12-minus-1 construction is borrowed from equities and is equally
motivated in FX: short-horizon (one month) exchange-rate moves show
reversal, while three-to-twelve-month moves show continuation. Including the
most recent month in a twelve-month signal blends the two and blunts both.
"""

from models import series

# Trading-day lookbacks on the ECB daily series (~258 observations/year).
DAYS_1M = 21
DAYS_3M = 64
DAYS_6M = 129
DAYS_12M = 258
MA_WINDOW = 200

# REER is monthly.
REER_TREND_MONTHS = 6


def compute(spot, reer):
    """Momentum metrics from daily spot and monthly REER.

    Args:
        spot: sorted [(iso_date, usd_per_unit)].
        reer: sorted [(YYYY-MM, index)].
    """
    ma200 = series.moving_average(spot, MA_WINDOW)
    last = series.last_value(spot)

    spot_vs_ma200 = None
    if ma200 and last:
        spot_vs_ma200 = last / ma200 - 1.0

    return {
        'ret_3m': series.pct_change(spot, DAYS_3M),
        'ret_6m': series.pct_change(spot, DAYS_6M),
        'ret_12m': series.pct_change(spot, DAYS_12M),

        # 12-month return excluding the most recent month, to keep
        # short-horizon reversal out of the medium-horizon trend signal.
        'ret_12m_1m': series.pct_change_between(spot, DAYS_12M, DAYS_1M),

        'spot_vs_ma200': spot_vs_ma200,

        # Real-terms trend. Nominal spot can drift purely on an inflation
        # differential; a rising REER means the currency is gaining real
        # purchasing power, which is a stronger statement.
        'reer_trend_6m': series.pct_change(reer, REER_TREND_MONTHS),

        'spot': last,
        'spot_date': series.last_period(spot),
        'spot_days': len(spot) if spot else 0,
    }
