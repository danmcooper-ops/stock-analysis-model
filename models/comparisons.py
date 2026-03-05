# models/comparisons.py
import pandas as pd

def compute_ratios(financials):
    ratios = {}
    bs = financials.get('balance_sheet')
    inc = financials.get('income_statement')
    cf = financials.get('cash_flow')

    if bs is None or inc is None or bs.empty or inc.empty:
        return ratios

    latest_bs = bs.iloc[:, 0]
    latest_inc = inc.iloc[:, 0]

    total_equity = _get(latest_bs, ['Stockholders Equity', 'Total Stockholder Equity', 'Common Stock Equity'])
    total_assets = _get(latest_bs, ['Total Assets'])
    total_liabilities = _get(latest_bs, ['Total Liabilities Net Minority Interest', 'Total Liab'])
    current_assets = _get(latest_bs, ['Current Assets', 'Total Current Assets'])
    current_liabilities = _get(latest_bs, ['Current Liabilities', 'Total Current Liabilities'])
    net_income = _get(latest_inc, ['Net Income', 'Net Income Common Stockholders'])

    if total_equity and net_income:
        ratios['ROE'] = net_income / total_equity

    if total_equity and total_liabilities:
        ratios['Debt-to-Equity'] = total_liabilities / total_equity

    if current_assets and current_liabilities:
        ratios['Current Ratio'] = current_assets / current_liabilities

    if total_assets and net_income:
        ratios['ROA'] = net_income / total_assets

    return ratios


def _get(series, keys):
    for key in keys:
        if key in series.index:
            val = series[key]
            if pd.notna(val) and val != 0:
                return val
    return None
