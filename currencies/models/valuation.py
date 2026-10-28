"""Valuation pillar — is the currency cheap or expensive in real terms?

Built entirely on the BIS real broad effective exchange rate. REER is
trade-weighted against 64 partners and deflated by relative consumer prices,
so it answers "what can this currency actually buy abroad, relative to its own
history" — which is the currency analogue of a multiple versus its own history.

**Sign convention.** A high REER means an expensive currency. Every function
here returns the raw REER-relative measure, so *lower is better* throughout
and the scoring layer inverts. This is the single easiest thing to get
backwards in the whole project; models/test coverage pins Japan as the
canonical cheap currency (REER 164 in 1994, ~65 today).

No World Bank dependency, deliberately. Valuation carries the largest pillar
weight, so it is built on the most reliable source available rather than on
the endpoint that 502s. PPP-based misalignment would be a natural fourth
measure but would make the heaviest pillar hostage to the flakiest API.
"""

from models import series

# REER is monthly, so lookbacks are in months.
LOOKBACK_20Y = 240
LOOKBACK_10Y = 120
LOOKBACK_5Y = 60

# Below this many observations a currency has no usable valuation anchor at
# all; the orchestrator flags it and the rating caps act on it.
MIN_REER_MONTHS = 60


def compute(reer):
    """Valuation metrics from a sorted [(YYYY-MM, index)] REER series.

    Returns a dict of raw metrics. Keys are absent-as-None rather than
    omitted, so downstream code can rely on the shape.
    """
    return {
        # Primary anchor: where today's real rate sits in its own 20-year
        # distribution. Negative = cheap. Preferred over a raw percentage
        # deviation because it normalizes by each currency's own volatility —
        # a 10% deviation means something very different for CHF than for TRY.
        'reer_z20': series.z_score(reer, LOOKBACK_20Y),

        # Percentage deviations at two shorter horizons. These are not
        # redundant with the z-score: they answer "how far" where the z-score
        # answers "how unusual", and they disagree exactly when a currency's
        # volatility regime has shifted.
        'reer_dev10': series.pct_deviation(reer, LOOKBACK_10Y),
        'reer_dev5': series.pct_deviation(reer, LOOKBACK_5Y),

        'reer_level': series.last_value(reer),
        'reer_period': series.last_period(reer),
        'reer_months': len(reer) if reer else 0,
    }


def has_anchor(row):
    """Whether this currency has enough REER history to be valued at all."""
    return (row.get('reer_months') or 0) >= MIN_REER_MONTHS
