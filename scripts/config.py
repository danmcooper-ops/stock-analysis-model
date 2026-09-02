# scripts/config.py
"""Constants and sector-specific DCF parameters for the stock analysis pipeline."""

# --- Constants ---
DEFAULT_RISK_FREE_RATE = 0.04  # Fallback if live Treasury fetch fails
ERP = 0.045                    # Equity Risk Premium. Damodaran's implied ERP has run
                               # ~4.2-4.6% since 2024; 4.5% is the mid-point (set
                               # 2026-09, refresh annually; models/capm.buildup_re
                               # mirrors it). The macro overlay flexes it by regime
                               # (models/macro.py).
TERMINAL_GROWTH_RATE = 0.03
MIN_MARKET_CAP = 0             # No market-cap floor
WACC_FLOOR = 0.07              # Morningstar-aligned WACC bounds (global default)
WACC_CAP = 0.13

# Growth estimation weights (6-signal system, auto-normalised when signals missing)
GROWTH_WEIGHT_FCF = 0.10        # FCF CAGR — noisy but direct
GROWTH_WEIGHT_REV = 0.10        # Revenue CAGR — stable, backward-looking
GROWTH_WEIGHT_ANALYST_ST = 0.15 # Analyst 1-year revenue growth
GROWTH_WEIGHT_ANALYST_LT = 0.30 # Analyst long-term growth (~5yr) — highest value signal
GROWTH_WEIGHT_EARNINGS_G = 0.15 # Analyst earnings growth (1yr forward)
GROWTH_WEIGHT_FUNDAMENTAL = 0.20 # Reinvestment Rate × ROIC — theoretically grounded

# Earnings surprise adjustment
SURPRISE_THRESHOLD = 0.05       # Avg surprise > 5% triggers adjustment
SURPRISE_UPLIFT = 0.015         # +1.5% growth for consistent beaters

# Margin trend adjustment
MARGIN_TREND_SENSITIVITY = 0.5  # 50% of margin annual change flows to growth premium

# Residual Income Model (see models/rim.py)
RIM_SPREAD_PERSISTENCE = 0.5   # Share of the ROE−Re spread surviving into the terminal period
RIM_MAX_BOOK_GROWTH = 0.25     # Cap on clean-surplus book growth (ROE × retention) per year

# Cost-of-equity bounds
BETA_MIN, BETA_MAX = 0.1, 4.0         # Valid beta range
RE_MIN, RE_MAX = 0.04, 0.30           # Valid cost-of-equity range
# Precision-weighted beta shrinkage (models/capm.shrink_beta): the regression
# beta is pulled toward BETA_PRIOR_MEAN with weight sd² / (sd² + SE²).
BETA_PRIOR_MEAN = 1.0
BETA_PRIOR_SD = 0.20
# Equity-model discount-rate cap = sector wacc_cap + this spread. Re must
# exceed WACC by construction, so capping Re at the WACC cap itself collapsed
# every beta above ~1.6 onto one discount rate. The floor stays shared.
RE_CAP_SPREAD = 0.03

# DCF parameters
CAPEX_DA_THRESHOLD = 2.0       # Owner earnings: capex > 2× D&A triggers growth-capex adj
EXCESS_CAPEX_ADDBACK = 0.50    # Add back 50% of capex above the threshold band (continuous at the threshold)
YIELD_CEILING_MULT = 1.25      # Mean-reversion: cap base FCF at 1.25× own trailing avg positive FCF (pre-adjustment basis)
HYPER_GROWTH_YIELD = 0.025     # FCF yield below 2.5% signals hyper-growth pricing
HYPER_GROWTH_CAP = 0.25        # Absolute ceiling on hyper-growth override
ANALYST_HAIRCUT = 0.80         # Apply 20% haircut to analyst growth estimate
FALLBACK_GROWTH = 0.05         # Default growth if no signals available
DCF_YEARS = 10                 # Total projection years
DCF_STAGE1 = 5                 # High-growth stage years

# Exit multiple cross-check
EXIT_MULT_DIVERGENCE_THRESHOLD = 0.30  # Flag low confidence if TV methods diverge >30%
EXIT_MULT_DEFAULT_EV_EBITDA = 12.0     # Default exit multiple if no sector median
EXIT_MULT_MIN = 5.0                     # Floor on exit multiple
EXIT_MULT_MAX = 30.0                    # Cap on exit multiple

# Monte Carlo simulation
# Scrambled Sobol points per simulation. A power of two keeps the sequence
# balanced; 1024 puts the median's sampling error under 1% at ~1 ms per
# ticker (the simulation is vectorized — the old "250 saves 2 min" predates that).
MC_ITERATIONS = 1024
MC_GROWTH_SIGMA_RATIO = 0.30    # Growth sigma = 30% of point estimate
MC_WACC_SIGMA = 0.01            # WACC sigma = 1 percentage point
MC_TERMINAL_GROWTH_SIGMA = 0.005 # Terminal growth sigma = 0.5pp
MC_EXIT_MULT_SIGMA_RATIO = 0.15 # Exit multiple sigma = 15% of point estimate
MC_HIGH_DIVERGENCE_SIGMA_MULT = 1.5  # Widen sigma 50% if TV methods diverge >30%
# Discount rate and terminal growth share the inflation / real-rate component;
# drawing them independently overstates how often the terminal spread collapses.
MC_WACC_TG_CORRELATION = 0.5
# MC confidence label is downgraded one notch when the simulation had to force
# more than this share of draws against a constraint wall (clip) or more than
# this share wiped out equity (invalid): the median then reflects the walls as
# much as the inputs.
MC_CLIP_RATE_DOWNGRADE = 0.20
MC_INVALID_RATE_DOWNGRADE = 0.10

# DDM (Dividend Discount Model) parameters
DDM_HIGH_GROWTH_YEARS = 5              # High-growth stage years
DDM_BLEND_WEIGHT = 0.30               # DDM weight in blended fair value
DCF_BLEND_WEIGHT_WITH_DDM = 0.70      # DCF weight when DDM is available
DDM_DIVERGENCE_THRESHOLD = 0.50       # Flag low confidence if DDM/DCF diverge >50%

# Continuous scoring weights by category. Rebalanced from a moat-first 40/20
# tilt toward a value orientation: Moat 40->30 (it was ~24% of the composite
# resting on ROIC counted three ways) and Valuation 20->30 so the "cheapness"
# side carries real weight rather than being outvoted 2:1 by quality/moat.
SCORE_WEIGHT_VALUATION = 0.30
SCORE_WEIGHT_QUALITY = 0.20
SCORE_WEIGHT_MOAT = 0.30
SCORE_WEIGHT_GROWTH = 0.10
SCORE_WEIGHT_OWNERSHIP = 0.10

# Macro narrative (Claude API). Generated at report-build time for the
# Macro Outlook tab; skipped cleanly when ANTHROPIC_API_KEY is unset or the
# API is unreachable. See data/claude_narrative.py.
CLAUDE_NARRATIVE_ENABLED = True
CLAUDE_NARRATIVE_MODEL = 'claude-opus-5'
CLAUDE_NARRATIVE_MAX_TOKENS = 6000   # ~2-3 paragraphs + 11 sector outlooks

# Post-processing
BLEND_TRIGGER = 1.5            # DCF > 1.5× multiples-FV triggers blending
BLEND_DCF_WEIGHT = 0.60        # Blend: 60% DCF
BLEND_MULT_WEIGHT = 0.40       # Blend: 40% multiples
EV_EBITDA_OUTLIER_MAX = 200    # Filter EV/EBITDA outliers above 200×
MIN_SECTOR_STOCKS = 3          # Min stocks per sector for median calculation
DATA_QUALITY_MIN = 40          # Skip tickers with quality score below this
MIN_MORNINGSTAR_SAMPLE = 5     # Min stocks for Morningstar comparison stats

# Minimum median daily dollar volume (3-month) for a name to stay BUY-rated.
# Below this a position can't be built or exited at a sane price, so the rating
# is capped at HOLD — the business may still be excellent, it just isn't
# actionable. Applied in scoring._rating_cap_for_row, and only when the metric
# is actually present, so missing volume data never demotes a stock.
MIN_ADV_FOR_BUY = 1_000_000

# ---------------------------------------------------------------------------
# Sector-specific DCF parameters (Fixes C/D/E/F)
# ---------------------------------------------------------------------------
SECTOR_CONFIG = {
    'Technology': {
        'growth_cap': 0.15, 'wacc_floor': 0.08, 'wacc_cap': 0.13,
        'avg_fcf_years': 1, 'check_owner_earnings': True,
        'norm_fcf_yield': 0.03, 'terminal_growth': 0.035,
    },
    'Communication Services': {
        'growth_cap': 0.12, 'wacc_floor': 0.07, 'wacc_cap': 0.12,
        'avg_fcf_years': 1, 'check_owner_earnings': True,
        'norm_fcf_yield': 0.03, 'terminal_growth': 0.03,
    },
    'Healthcare': {
        'growth_cap': 0.15, 'wacc_floor': 0.08, 'wacc_cap': 0.13,
        'avg_fcf_years': 1, 'check_owner_earnings': False,
        'norm_fcf_yield': 0.03, 'terminal_growth': 0.035,
    },
    'Consumer Cyclical': {
        'growth_cap': 0.12, 'wacc_floor': 0.07, 'wacc_cap': 0.12,
        'avg_fcf_years': 1, 'check_owner_earnings': True,
        'norm_fcf_yield': 0.035, 'terminal_growth': 0.025,
    },
    'Consumer Defensive': {
        'growth_cap': 0.08, 'wacc_floor': 0.06, 'wacc_cap': 0.10,
        'avg_fcf_years': 1, 'check_owner_earnings': False,
        'norm_fcf_yield': 0.04, 'terminal_growth': 0.02,
    },
    'Industrials': {
        'growth_cap': 0.10, 'wacc_floor': 0.07, 'wacc_cap': 0.11,
        'avg_fcf_years': 1, 'check_owner_earnings': False,
        'norm_fcf_yield': 0.035, 'terminal_growth': 0.025,
    },
    'Energy': {
        'growth_cap': 0.05, 'wacc_floor': 0.08, 'wacc_cap': 0.12,
        'avg_fcf_years': 3, 'check_owner_earnings': False,
        'norm_fcf_yield': 0.06, 'terminal_growth': 0.015,
    },
    'Basic Materials': {
        'growth_cap': 0.05, 'wacc_floor': 0.08, 'wacc_cap': 0.12,
        'avg_fcf_years': 3, 'check_owner_earnings': False,
        'norm_fcf_yield': 0.05, 'terminal_growth': 0.015,
    },
    'Utilities': {
        'growth_cap': 0.05, 'wacc_floor': 0.05, 'wacc_cap': 0.09,
        'avg_fcf_years': 1, 'check_owner_earnings': False,
        'norm_fcf_yield': 0.04, 'terminal_growth': 0.02,
    },
    'Real Estate': {
        'growth_cap': 0.06, 'wacc_floor': 0.06, 'wacc_cap': 0.10,
        'avg_fcf_years': 1, 'check_owner_earnings': False,
        'norm_fcf_yield': 0.04, 'terminal_growth': 0.02,
    },
}
SECTOR_DEFAULT = {
    'growth_cap': 0.12, 'wacc_floor': 0.07, 'wacc_cap': 0.13,
    'avg_fcf_years': 1, 'check_owner_earnings': False,
    'norm_fcf_yield': 0.035, 'terminal_growth': TERMINAL_GROWTH_RATE,
}


def _get_sector_config(sector):
    """Look up sector-specific DCF parameters with default fallback."""
    return SECTOR_CONFIG.get(sector, SECTOR_DEFAULT)
