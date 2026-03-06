# models/capm.py
import numpy as np


def calculate_beta(stock_returns, market_returns, adjust=True):
    n = len(stock_returns)
    cov_matrix = np.cov(stock_returns, market_returns)
    raw_beta = cov_matrix[0, 1] / cov_matrix[1, 1]

    adjusted_beta = (2 / 3) * raw_beta + (1 / 3) * 1.0 if adjust else raw_beta

    r_squared = np.corrcoef(stock_returns, market_returns)[0, 1] ** 2

    se_beta = (np.sqrt((1 - r_squared) / (n - 2))
               * (np.std(stock_returns) / np.std(market_returns))) if n > 2 else None

    return {
        'raw_beta': raw_beta,
        'adjusted_beta': adjusted_beta,
        'r_squared': r_squared,
        'se_beta': se_beta,
        'n_observations': n,
    }


def calculate_r2(stock_returns, market_returns):
    """R² from regression of stock returns on market returns."""
    corr = np.corrcoef(stock_returns, market_returns)[0, 1]
    return corr ** 2


def calculate_alpha(stock_returns, market_returns):
    """Annualized regression alpha (Jensen's alpha proxy): intercept * 252."""
    beta = calculate_beta(stock_returns, market_returns, adjust=False)['raw_beta']
    daily_alpha = np.mean(stock_returns) - beta * np.mean(market_returns)
    return daily_alpha * 252


def calculate_residual_sigma(stock_returns, market_returns):
    """Annualized residual standard deviation from OLS regression."""
    beta = calculate_beta(stock_returns, market_returns, adjust=False)['raw_beta']
    alpha_daily = np.mean(stock_returns) - beta * np.mean(market_returns)
    residuals = stock_returns - (alpha_daily + beta * market_returns)
    return np.std(residuals) * np.sqrt(252)


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


def geometric_annualized_return(daily_returns, trading_days=252):
    if len(daily_returns) == 0:
        return None
    cum = np.prod(1 + daily_returns) - 1
    years = len(daily_returns) / trading_days
    if years <= 0 or cum <= -1:
        return None
    return (1 + cum) ** (1 / years) - 1


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
