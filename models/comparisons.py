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


def calculate_roic(financials):
    bs = financials.get('balance_sheet')
    inc = financials.get('income_statement')

    if bs is None or inc is None or bs.empty or inc.empty:
        return None

    common_years = bs.columns.intersection(inc.columns)
    if len(common_years) == 0:
        return None

    roic_by_year = {}
    for year in common_years:
        bs_year = bs[year]
        inc_year = inc[year]

        operating_income = _get(inc_year, ['Operating Income', 'Total Operating Income As Reported'])
        tax_provision = _get(inc_year, ['Tax Provision'])
        pretax_income = _get(inc_year, ['Pretax Income'])
        total_equity = _get(bs_year, ['Stockholders Equity', 'Total Stockholder Equity', 'Common Stock Equity'])
        total_debt = _get(bs_year, ['Total Debt', 'Long Term Debt'])
        cash = _get(bs_year, ['Cash And Cash Equivalents', 'Cash Cash Equivalents And Short Term Investments', 'Cash Financial'])

        if not all([operating_income, pretax_income, total_equity]):
            continue

        tax_rate = (tax_provision / pretax_income) if tax_provision and pretax_income else 0.21
        nopat = operating_income * (1 - tax_rate)
        invested_capital = total_equity + (total_debt or 0) - (cash or 0)

        if invested_capital <= 0:
            continue

        roic_by_year[str(year.year) if hasattr(year, 'year') else str(year)] = nopat / invested_capital

    if not roic_by_year:
        return None

    avg_roic = sum(roic_by_year.values()) / len(roic_by_year)
    return {'roic_by_year': roic_by_year, 'avg_roic': avg_roic}


def _get(series, keys):
    for key in keys:
        if key in series.index:
            val = series[key]
            if pd.notna(val) and val != 0:
                return val
    return None
