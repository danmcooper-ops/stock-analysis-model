"""External pillar — can the country fund itself without selling its currency?

This is the solvency side of the model. A currency can be cheap on REER and
pay a fat carry and still be a bad hold if the country is running a large
external deficit on thin reserves, because the adjustment mechanism for that
imbalance *is* currency depreciation.

All inputs are annual World Bank indicators and are therefore one to two
years stale by construction. That is acceptable for the question being asked —
external vulnerability is a slow-moving structural feature — but it does mean
this pillar cannot catch a fast-developing balance-of-payments crisis. The
Stability pillar's realized volatility is what moves quickly.

Inflation appears here rather than in Stability because its *level* is an
external-competitiveness fact (persistent inflation erodes the real exchange
rate), while its *variability* is a risk fact. Stability takes the variance.
"""

from models import series

# Inflation is scored against a band, not monotonically: deflation is a
# problem in its own right, so a currency printing -2% should not outrank one
# printing 2%. These bracket the range most inflation-targeting central banks
# aim at.
INFLATION_TARGET_LOW = 1.0
INFLATION_TARGET_HIGH = 3.0


def inflation_gap(inflation_pct):
    """Absolute distance outside the target band, in percentage points.

    0.0 anywhere inside [1%, 3%]; 5.0 at either 8% inflation or -4% deflation.
    Scoring this rather than raw inflation is what keeps a deflating economy
    from screening as maximally price-stable.
    """
    if inflation_pct is None:
        return None
    if inflation_pct < INFLATION_TARGET_LOW:
        return INFLATION_TARGET_LOW - inflation_pct
    if inflation_pct > INFLATION_TARGET_HIGH:
        return inflation_pct - INFLATION_TARGET_HIGH
    return 0.0


def compute(macro):
    """External metrics from a World Bank macro dict for one currency.

    Missing indicators propagate as None; the scoring layer treats that as
    missing data (scores 0, stays in the denominator) rather than excusing it.
    """
    macro = macro or {}
    inflation = macro.get('inflation_pct')

    return {
        # Positive = surplus. The single best external-vulnerability summary.
        'current_account_pct_gdp': macro.get('current_account_pct_gdp'),

        # Import cover. The conventional adequacy floor is three months.
        'reserves_months_imports': macro.get('reserves_months_imports'),

        # Lower is better. Sparse for advanced economies (the World Bank
        # reports it mainly for developing countries), so expect N/A on the
        # G10 rows — that sparsity is real, not a fetch failure.
        'external_debt_pct_gni': macro.get('external_debt_pct_gni'),

        'inflation_pct': inflation,
        'inflation_gap': inflation_gap(inflation),
        'gdp_growth_pct': macro.get('gdp_growth_pct'),
        'gov_debt_pct_gdp': macro.get('gov_debt_pct_gdp'),

        # Recorded so the report can show how stale each row's macro is.
        'macro_year': macro.get('current_account_pct_gdp_year')
                      or macro.get('inflation_pct_year'),
    }


def inflation_volatility(macro):
    """Standard deviation of the annual inflation series, in percentage points.

    Consumed by the Stability pillar. Lives here because this module owns the
    World Bank macro shape.
    """
    return series.annual_series_stdev((macro or {}).get('inflation_pct_series'))
