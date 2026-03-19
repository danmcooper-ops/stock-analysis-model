# tests/conftest.py
import sys
import os
import pytest
import pandas as pd
import numpy as np

# Add project root to sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def sample_info():
    """Minimal yfinance info dict with typical fields."""
    return {
        'marketCap': 100e9,
        'currentPrice': 150.0,
        'sharesOutstanding': 666_666_667,
        'beta': 1.2,
        'dividendRate': 2.0,
        'regularMarketPrice': 150.0,
        'enterpriseValue': 110e9,
        'enterpriseToEbitda': 15.0,
        'trailingPE': 25.0,
        'priceToBook': 5.0,
        'revenueGrowth': 0.12,
        'recommendationKey': 'buy',
        'numberOfAnalystOpinions': 30,
        'targetMeanPrice': 170.0,
        'targetHighPrice': 200.0,
        'targetLowPrice': 130.0,
        'sector': 'Technology',
        'industry': 'Software',
        'payoutRatio': 0.30,
        'trailingEps': 6.0,
        'fiftyTwoWeekHigh': 180.0,
        'fiftyTwoWeekLow': 110.0,
        'bookValue': 45.0,
        'heldPercentInsiders': 0.05,
    }


@pytest.fixture
def sample_balance_sheet():
    """Two years of balance sheet data (columns are dates, newest first)."""
    dates = pd.to_datetime(['2024-12-31', '2023-12-31'])
    data = {
        dates[0]: {
            'Total Assets': 50e9,
            'Stockholders Equity': 30e9,
            'Total Debt': 10e9,
            'Current Assets': 20e9,
            'Current Liabilities': 10e9,
            'Cash And Cash Equivalents': 5e9,
            'Retained Earnings': 20e9,
            'Total Liabilities Net Minority Interest': 20e9,
            'Ordinary Shares Number': 666_666_667,
            'Long Term Debt': 8e9,
            'Accounts Receivable': 3e9,
            'Net PPE': 15e9,
        },
        dates[1]: {
            'Total Assets': 45e9,
            'Stockholders Equity': 27e9,
            'Total Debt': 11e9,
            'Current Assets': 18e9,
            'Current Liabilities': 10e9,
            'Cash And Cash Equivalents': 4e9,
            'Retained Earnings': 17e9,
            'Total Liabilities Net Minority Interest': 18e9,
            'Ordinary Shares Number': 670_000_000,
            'Long Term Debt': 9e9,
            'Accounts Receivable': 2.5e9,
            'Net PPE': 13e9,
        },
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_income_statement():
    """Two years of income statement."""
    dates = pd.to_datetime(['2024-12-31', '2023-12-31'])
    data = {
        dates[0]: {
            'Total Revenue': 40e9,
            'Operating Income': 12e9,
            'Net Income': 8e9,
            'Tax Provision': 2e9,
            'Pretax Income': 10e9,
            'Interest Expense': 0.5e9,
            'Gross Profit': 25e9,
            'Net Income Common Stockholders': 8e9,
            'EBIT': 12e9,
            'Selling General And Administration': 8e9,
        },
        dates[1]: {
            'Total Revenue': 35e9,
            'Operating Income': 10e9,
            'Net Income': 6e9,
            'Tax Provision': 1.5e9,
            'Pretax Income': 7.5e9,
            'Interest Expense': 0.6e9,
            'Gross Profit': 21e9,
            'Net Income Common Stockholders': 6e9,
            'EBIT': 10e9,
            'Selling General And Administration': 7e9,
        },
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_cash_flow():
    """Two years of cash flow."""
    dates = pd.to_datetime(['2024-12-31', '2023-12-31'])
    data = {
        dates[0]: {
            'Free Cash Flow': 6e9,
            'Operating Cash Flow': 10e9,
            'Capital Expenditure': -4e9,
            'Depreciation And Amortization': 3e9,
        },
        dates[1]: {
            'Free Cash Flow': 5e9,
            'Operating Cash Flow': 8e9,
            'Capital Expenditure': -3e9,
            'Depreciation And Amortization': 2.5e9,
        },
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_financials(sample_info, sample_balance_sheet,
                      sample_income_statement, sample_cash_flow):
    """Complete financials dict as returned by YFinanceClient.fetch_financials()."""
    return {
        'info': sample_info,
        'balance_sheet': sample_balance_sheet,
        'income_statement': sample_income_statement,
        'cash_flow': sample_cash_flow,
    }


@pytest.fixture
def sample_dividend_history():
    """Five years of stable dividends (oldest first)."""
    return [1.50, 1.55, 1.60, 1.68, 1.76]


@pytest.fixture
def sample_growing_dividend_history():
    """Ten years of steadily growing dividends (oldest first)."""
    return [1.00, 1.08, 1.17, 1.26, 1.36, 1.47, 1.59, 1.72, 1.86, 2.01]


@pytest.fixture
def synthetic_returns():
    """Synthetic stock and market returns for CAPM testing.

    Stock returns = 1.3 * market_returns + noise, so true beta ≈ 1.3.
    """
    np.random.seed(42)
    n = 500
    market = np.random.normal(0.0004, 0.01, n)  # ~10% annualized
    noise = np.random.normal(0, 0.005, n)
    stock = 1.3 * market + noise
    return stock, market
