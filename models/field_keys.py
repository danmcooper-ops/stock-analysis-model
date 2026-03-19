# models/field_keys.py
"""Shared infrastructure for extracting values from yfinance financial statements.

Provides `_get()` for robust field lookup and canonical key-name lists for
balance sheet, income statement, and cash flow fields.
"""
import pandas as pd


def _get(series, keys, allow_zero=True):
    """Retrieve the first non-null value from *series* matching any of *keys*."""
    for key in keys:
        if key in series.index:
            val = series[key]
            if pd.notna(val) and (allow_zero or val != 0):
                return val
    return None


# --- Canonical field-key lists (DRY across all functions) ---
EQUITY_KEYS = ['Stockholders Equity', 'Total Stockholder Equity', 'Common Stock Equity']
DEBT_KEYS = ['Total Debt', 'Long Term Debt']
CASH_KEYS = ['Cash And Cash Equivalents', 'Cash Cash Equivalents And Short Term Investments', 'Cash Financial']
CURRENT_ASSETS_KEYS = ['Current Assets', 'Total Current Assets']
CURRENT_LIABILITIES_KEYS = ['Current Liabilities', 'Total Current Liabilities']
NET_INCOME_KEYS = ['Net Income', 'Net Income Common Stockholders']
TOTAL_ASSETS_KEYS = ['Total Assets']
EBITDA_KEYS = ['EBITDA', 'Normalized EBITDA']
OPERATING_CF_KEYS = ['Operating Cash Flow', 'Cash From Operations']
REVENUE_KEYS = ['Total Revenue']
OPERATING_INCOME_KEYS = ['Operating Income', 'Total Operating Income As Reported']
INTEREST_KEYS = ['Interest Expense', 'Interest Expense Non Operating']
DA_KEYS = ['Depreciation And Amortization', 'Depreciation Amortization Depletion']
AR_KEYS = ['Accounts Receivable', 'Net Receivables', 'Receivables']
PPE_KEYS = ['Net PPE', 'Property Plant Equipment Net', 'Property Plant And Equipment Net']
SGA_KEYS = ['Selling General And Administration', 'Selling And Marketing Expense']
GROSS_PROFIT_KEYS = ['Gross Profit']
LTD_KEYS = ['Long Term Debt']
