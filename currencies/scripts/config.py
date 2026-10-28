"""Model parameters. No I/O, no keys, no paths — constants only.

Every threshold in scoring.py that is not a literal in a gate's score curve
lives here, so calibration can sweep them without touching gate definitions.
"""

# ---------------------------------------------------------------------------
# Pillar weights — must sum to 1.0 (asserted below).
# ---------------------------------------------------------------------------
# Valuation carries the most weight because mean reversion in the real
# exchange rate is the best-evidenced medium-horizon effect in FX, and because
# it rests on the most reliable data source in the project (BIS REER).
#
# Carry is second: reliable positive expectancy, but fat left tail, which is
# why the pillar internally weights *real* carry over nominal.
#
# External is third — slow-moving and one to two years stale, so it acts as a
# solvency filter rather than a timing signal.
#
# Momentum and Stability are deliberately light. Momentum is a confirmation
# overlay on valuation, not an independent thesis; Stability is a risk
# adjustment that is also partly double-counted inside carry-to-vol.
SCORE_WEIGHT_VALUATION = 0.30
SCORE_WEIGHT_CARRY = 0.25
SCORE_WEIGHT_EXTERNAL = 0.20
SCORE_WEIGHT_MOMENTUM = 0.15
SCORE_WEIGHT_STABILITY = 0.10

assert abs(SCORE_WEIGHT_VALUATION + SCORE_WEIGHT_CARRY + SCORE_WEIGHT_EXTERNAL
           + SCORE_WEIGHT_MOMENTUM + SCORE_WEIGHT_STABILITY - 1.0) < 1e-9

# ---------------------------------------------------------------------------
# Rating bands
# ---------------------------------------------------------------------------
# Provisional. The sibling stock model quantile-matched its bands against a
# 2,211-name universe; with 30 rows that is not meaningful, so these are set
# on the composite's natural 0-100 scale and should be revisited once enough
# daily snapshots exist to calibrate against forward returns.
RATING_THRESHOLD_LONG = 57.0
RATING_THRESHOLD_LEAN = 39.0
RATING_THRESHOLD_NEUTRAL = 25.0

# ---------------------------------------------------------------------------
# Rating cap triggers
# ---------------------------------------------------------------------------
# Annual CPI above this is a monetary regime, not a carry trade. Nominal carry
# will look spectacular and the currency will depreciate to match.
HYPERINFLATION_PCT = 50.0

# Below this share of applicable gates covered, the composite is an artifact
# of what happened to be missing rather than a judgement.
MIN_DATA_COVERAGE = 25.0

# Minimum months of REER history before the valuation anchor means anything.
MIN_REER_MONTHS = 60

# Minimum daily spot observations for momentum and volatility to be computed
# over their full lookbacks (3 years).
MIN_SPOT_DAYS = 774

# ---------------------------------------------------------------------------
# Universe / display
# ---------------------------------------------------------------------------
PILLAR_ORDER = ['Valuation', 'Carry', 'External', 'Momentum', 'Stability']

PILLAR_WEIGHTS = {
    'Valuation': SCORE_WEIGHT_VALUATION,
    'Carry': SCORE_WEIGHT_CARRY,
    'External': SCORE_WEIGHT_EXTERNAL,
    'Momentum': SCORE_WEIGHT_MOMENTUM,
    'Stability': SCORE_WEIGHT_STABILITY,
}

# Pillar colours, shared between the Python gate metadata and the report CSS
# so the Gate Matrix header bands cannot drift from the legend.
PILLAR_COLORS = {
    'Valuation': {'dark': '#2F5496', 'light': '#D6E0F0'},
    'Carry':     {'dark': '#548235', 'light': '#DDEBD3'},
    'External':  {'dark': '#C55A11', 'light': '#FBE2D5'},
    'Momentum':  {'dark': '#7030A0', 'light': '#E4D7F0'},
    'Stability': {'dark': '#BF8F00', 'light': '#FBEEC8'},
}
