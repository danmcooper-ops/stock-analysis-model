# scripts/param_set.py
"""ParamSet: overridable parameter sets for the stock analysis pipeline.

A ParamSet is a plain dict mapping parameter names to values.
``default_params()`` returns the current config.py constants.
``merge_params(overrides)`` merges user overrides on top of defaults.
``validate_params(params)`` checks constraints (weight sums, ranges).

When no overrides are given, the pipeline behaves identically to the
hardcoded-constant path.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.config import (
    ERP, TERMINAL_GROWTH_RATE, MIN_MARKET_CAP, WACC_FLOOR, WACC_CAP,
    GROWTH_WEIGHT_FCF, GROWTH_WEIGHT_REV,
    GROWTH_WEIGHT_ANALYST_ST, GROWTH_WEIGHT_ANALYST_LT,
    GROWTH_WEIGHT_EARNINGS_G, GROWTH_WEIGHT_FUNDAMENTAL,
    SURPRISE_THRESHOLD, SURPRISE_UPLIFT, MARGIN_TREND_SENSITIVITY,
    CAPEX_DA_THRESHOLD, EXCESS_CAPEX_ADDBACK, YIELD_CEILING_MULT,
    HYPER_GROWTH_YIELD, HYPER_GROWTH_CAP,
    ANALYST_HAIRCUT, FALLBACK_GROWTH, DCF_YEARS, DCF_STAGE1,
    EXIT_MULT_DIVERGENCE_THRESHOLD, EXIT_MULT_DEFAULT_EV_EBITDA,
    EXIT_MULT_MIN, EXIT_MULT_MAX,
    MC_ITERATIONS, MC_GROWTH_SIGMA_RATIO, MC_WACC_SIGMA,
    MC_TERMINAL_GROWTH_SIGMA, MC_EXIT_MULT_SIGMA_RATIO,
    MC_HIGH_DIVERGENCE_SIGMA_MULT,
    DDM_HIGH_GROWTH_YEARS, DDM_BLEND_WEIGHT,
    DCF_BLEND_WEIGHT_WITH_DDM, DDM_DIVERGENCE_THRESHOLD,
    BLEND_TRIGGER, BLEND_DCF_WEIGHT, BLEND_MULT_WEIGHT,
    SCORE_WEIGHT_VALUATION, SCORE_WEIGHT_QUALITY,
    SCORE_WEIGHT_MOAT, SCORE_WEIGHT_GROWTH,
)


def default_params():
    """Return the default parameter set mirroring config.py constants.

    Returns:
        dict: {param_name: default_value} for every tunable parameter.
    """
    return {
        # Core valuation assumptions
        'erp': ERP,
        'terminal_growth_rate': TERMINAL_GROWTH_RATE,
        'min_market_cap': MIN_MARKET_CAP,
        'wacc_floor': WACC_FLOOR,
        'wacc_cap': WACC_CAP,

        # Growth estimation weights (6-signal system)
        'growth_weight_fcf': GROWTH_WEIGHT_FCF,
        'growth_weight_rev': GROWTH_WEIGHT_REV,
        'growth_weight_analyst_st': GROWTH_WEIGHT_ANALYST_ST,
        'growth_weight_analyst_lt': GROWTH_WEIGHT_ANALYST_LT,
        'growth_weight_earnings_g': GROWTH_WEIGHT_EARNINGS_G,
        'growth_weight_fundamental': GROWTH_WEIGHT_FUNDAMENTAL,

        # Earnings surprise & margin adjustments
        'surprise_threshold': SURPRISE_THRESHOLD,
        'surprise_uplift': SURPRISE_UPLIFT,
        'margin_trend_sensitivity': MARGIN_TREND_SENSITIVITY,

        # DCF parameters
        'capex_da_threshold': CAPEX_DA_THRESHOLD,
        'excess_capex_addback': EXCESS_CAPEX_ADDBACK,
        'yield_ceiling_mult': YIELD_CEILING_MULT,
        'hyper_growth_yield': HYPER_GROWTH_YIELD,
        'hyper_growth_cap': HYPER_GROWTH_CAP,
        'analyst_haircut': ANALYST_HAIRCUT,
        'fallback_growth': FALLBACK_GROWTH,
        'dcf_years': DCF_YEARS,
        'dcf_stage1': DCF_STAGE1,

        # Exit multiple cross-check
        'exit_mult_divergence_threshold': EXIT_MULT_DIVERGENCE_THRESHOLD,
        'exit_mult_default_ev_ebitda': EXIT_MULT_DEFAULT_EV_EBITDA,
        'exit_mult_min': EXIT_MULT_MIN,
        'exit_mult_max': EXIT_MULT_MAX,

        # Monte Carlo
        'mc_iterations': MC_ITERATIONS,
        'mc_growth_sigma_ratio': MC_GROWTH_SIGMA_RATIO,
        'mc_wacc_sigma': MC_WACC_SIGMA,
        'mc_terminal_growth_sigma': MC_TERMINAL_GROWTH_SIGMA,
        'mc_exit_mult_sigma_ratio': MC_EXIT_MULT_SIGMA_RATIO,
        'mc_high_divergence_sigma_mult': MC_HIGH_DIVERGENCE_SIGMA_MULT,

        # DDM blending
        'ddm_high_growth_years': DDM_HIGH_GROWTH_YEARS,
        'ddm_blend_weight': DDM_BLEND_WEIGHT,
        'dcf_blend_weight_with_ddm': DCF_BLEND_WEIGHT_WITH_DDM,
        'ddm_divergence_threshold': DDM_DIVERGENCE_THRESHOLD,

        # Post-processing blending
        'blend_trigger': BLEND_TRIGGER,
        'blend_dcf_weight': BLEND_DCF_WEIGHT,
        'blend_mult_weight': BLEND_MULT_WEIGHT,

        # Composite scoring category weights
        'score_weight_valuation': SCORE_WEIGHT_VALUATION,
        'score_weight_quality': SCORE_WEIGHT_QUALITY,
        'score_weight_moat': SCORE_WEIGHT_MOAT,
        'score_weight_growth': SCORE_WEIGHT_GROWTH,
    }


def merge_params(overrides=None):
    """Return default params with optional overrides applied.

    Args:
        overrides: Optional dict of {param_name: value}.
            Unknown keys raise ``ValueError``.

    Returns:
        dict: Complete parameter set.
    """
    params = default_params()
    if overrides:
        for k, v in overrides.items():
            if k not in params:
                raise ValueError(f"Unknown parameter: '{k}'")
            params[k] = v
    return params


def validate_params(params):
    """Return a list of validation error strings (empty if valid).

    Checks:
    - Composite scoring weights sum to ~1.0
    - Blend weights sum to ~1.0
    - DDM blend weights sum to ~1.0
    - ERP in sensible range [0.02, 0.10]
    - Terminal growth in [0.0, 0.06]
    - Each scoring weight >= 0.05
    - DCF stage1 <= DCF years
    - MC iterations >= 100
    """
    errors = []

    # Scoring weights must sum to 1.0
    sw = (params.get('score_weight_valuation', 0)
          + params.get('score_weight_quality', 0)
          + params.get('score_weight_moat', 0)
          + params.get('score_weight_growth', 0))
    if abs(sw - 1.0) > 0.01:
        errors.append(f"Scoring weights sum to {sw:.3f}, expected 1.0")

    # Each scoring weight should be >= 0.05
    for key in ('score_weight_valuation', 'score_weight_quality',
                'score_weight_moat', 'score_weight_growth'):
        v = params.get(key, 0)
        if v < 0.05:
            errors.append(f"{key} = {v:.3f} is below minimum 0.05")

    # Blend weights
    bw = params.get('blend_dcf_weight', 0) + params.get('blend_mult_weight', 0)
    if abs(bw - 1.0) > 0.01:
        errors.append(f"Blend weights sum to {bw:.3f}, expected 1.0")

    # DDM blend weights
    dw = (params.get('dcf_blend_weight_with_ddm', 0)
          + params.get('ddm_blend_weight', 0))
    if abs(dw - 1.0) > 0.01:
        errors.append(f"DDM blend weights sum to {dw:.3f}, expected 1.0")

    # ERP range
    erp = params.get('erp', 0)
    if not (0.02 <= erp <= 0.10):
        errors.append(f"ERP {erp:.4f} outside valid range [0.02, 0.10]")

    # Terminal growth range
    tg = params.get('terminal_growth_rate', 0)
    if not (0.0 <= tg <= 0.06):
        errors.append(f"Terminal growth {tg:.4f} outside valid range [0.0, 0.06]")

    # DCF stage1 <= DCF years
    if params.get('dcf_stage1', 0) > params.get('dcf_years', 0):
        errors.append(
            f"dcf_stage1 ({params['dcf_stage1']}) > dcf_years ({params['dcf_years']})")

    # MC iterations
    mc = params.get('mc_iterations', 0)
    if mc < 100:
        errors.append(f"mc_iterations = {mc} is below minimum 100")

    return errors
