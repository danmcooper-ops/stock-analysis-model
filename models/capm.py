# models/capm.py
import numpy as np


def calculate_beta(stock_returns, market_returns, adjust=True):
    n = len(stock_returns)
    cov_matrix = np.cov(stock_returns, market_returns)
    raw_beta = cov_matrix[0, 1] / cov_matrix[1, 1]

    adjusted_beta = (2 / 3) * raw_beta + (1 / 3) * 1.0 if adjust else raw_beta

    r_squared = np.corrcoef(stock_returns, market_returns)[0, 1] ** 2

    se_beta = np.sqrt((1 - r_squared) / (n - 2)) * (np.std(stock_returns) / np.std(market_returns)) if n > 2 else None

    return {
        'raw_beta': raw_beta,
        'adjusted_beta': adjusted_beta,
        'r_squared': r_squared,
        'se_beta': se_beta,
        'n_observations': n,
    }


def geometric_annualized_return(daily_returns, trading_days=252):
    if len(daily_returns) == 0:
        return None
    cum = np.prod(1 + daily_returns) - 1
    years = len(daily_returns) / trading_days
    if years <= 0 or cum <= -1:
        return None
    return (1 + cum) ** (1 / years) - 1


def expected_return(risk_free_rate, beta, market_return):
    return risk_free_rate + beta * (market_return - risk_free_rate)
