# models/capm.py
import numpy as np

from models.valuation_types import _validate_returns


def calculate_beta(stock_returns, market_returns, adjust=True, min_obs=24):
    """Raw + Blume-adjusted beta with R² and standard error.

    Hardened: requires equal-length series, min_obs observations, finite
    returns, and nonzero market variance. Raises ValueError on any of
    these so callers can't silently propagate inf / NaN into downstream
    discount-rate math. Wrap the call in try/except if upstream data
    quality is uncertain.

    Uses sample variance (ddof=1) for the unbiased Blume estimator and
    clips R² into [0, 1] to absorb floating-point noise that can put it
    a hair outside the valid range.
    """
    s = _validate_returns('stock_returns', stock_returns, min_obs=min_obs)
    m = _validate_returns('market_returns', market_returns, min_obs=min_obs)
    if s.size != m.size:
        raise ValueError(
            f"stock_returns ({s.size}) and market_returns ({m.size}) "
            "must be equal length"
        )

    market_var = float(np.var(m, ddof=1))
    if market_var <= 0:
        raise ValueError("market_returns variance is zero — beta is undefined")

    cov_matrix = np.cov(s, m, ddof=1)
    raw_beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    adjusted_beta = (2 / 3) * raw_beta + (1 / 3) * 1.0 if adjust else raw_beta

    r_squared = float(np.clip(np.corrcoef(s, m)[0, 1] ** 2, 0.0, 1.0))

    n = s.size
    se_beta = (np.sqrt((1 - r_squared) / (n - 2))
               * (np.std(s, ddof=1) / np.std(m, ddof=1))) if n > 2 else None

    warnings = []
    if n < 60:
        warnings.append(f'Only {n} observations — beta estimate is noisy')
    if r_squared < 0.20:
        warnings.append(
            f'Low R² ({r_squared:.2f}) — market exposure does not explain '
            'returns; beta unreliable as a discount-rate input'
        )

    return {
        'raw_beta': float(raw_beta),
        'adjusted_beta': float(adjusted_beta),
        'r_squared': r_squared,
        'se_beta': float(se_beta) if se_beta is not None else None,
        'n_observations': n,
        'warnings': warnings,
    }


def r2_diagnostic(r2):
    """
    Worksheet Step 4A gate: determines cost-of-equity method based on R².
    Returns (classification, method) where:
      >= 60%: 'reliable'     -> 'capm'
      40-59%: 'directional' -> 'capm_plus_alternative'
      < 40%:  'unreliable'  -> 'fundamental_only'
    """
    if r2 >= 0.60:
        return 'reliable', 'capm'
    elif r2 >= 0.40:
        return 'directional', 'capm_plus_alternative'
    else:
        return 'unreliable', 'fundamental_only'


def expected_return(risk_free_rate, beta, market_return):
    """CAPM: Re = Rf + β(Rm - Rf)."""
    return risk_free_rate + beta * (market_return - risk_free_rate)


def ggm_implied_re(dividend_yield, growth_rate):
    """GGM-implied Re = D1/P + g = dividend_yield*(1+g) + g."""
    if dividend_yield is None or dividend_yield <= 0:
        return None
    return dividend_yield * (1 + growth_rate) + growth_rate


def buildup_re(risk_free_rate, erp=0.055, size_premium=0.02, industry_premium=0.01):
    """Build-Up Re = Rf + ERP + size premium + industry premium."""
    return risk_free_rate + erp + size_premium + industry_premium
