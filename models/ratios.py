# models/ratios.py
"""Core financial ratios: WACC, ROIC, DuPont decomposition, fundamental growth."""
import pandas as pd
from models.field_keys import (
    _get, EQUITY_KEYS, DEBT_KEYS, CASH_KEYS, CURRENT_ASSETS_KEYS,
    CURRENT_LIABILITIES_KEYS, NET_INCOME_KEYS, TOTAL_ASSETS_KEYS,
    OPERATING_INCOME_KEYS, INTEREST_KEYS, DA_KEYS, REVENUE_KEYS,
    OPERATING_CF_KEYS,
)

def compute_ratios(financials):
    ratios = {}
    bs = financials.get('balance_sheet')
    inc = financials.get('income_statement')
    if bs is None or inc is None or bs.empty or inc.empty:
        return ratios

    latest_bs = bs.iloc[:, 0]
    latest_inc = inc.iloc[:, 0]

    total_equity = _get(latest_bs, EQUITY_KEYS, allow_zero=False)
    total_assets = _get(latest_bs, TOTAL_ASSETS_KEYS, allow_zero=False)
    total_liabilities = _get(latest_bs, ['Total Liabilities Net Minority Interest', 'Total Liab'])
    current_assets = _get(latest_bs, CURRENT_ASSETS_KEYS)
    current_liabilities = _get(latest_bs, CURRENT_LIABILITIES_KEYS, allow_zero=False)
    net_income = _get(latest_inc, NET_INCOME_KEYS)

    if total_equity and net_income:
        ratios['ROE'] = net_income / total_equity
    if total_equity and total_liabilities:
        ratios['Debt-to-Equity'] = total_liabilities / total_equity
    if current_assets and current_liabilities:
        ratios['Current Ratio'] = current_assets / current_liabilities
    if total_assets and net_income:
        ratios['ROA'] = net_income / total_assets
    return ratios


def calculate_wacc(financials, cost_of_equity):
    """
    WACC = (E/V)*Re + (D/V)*Rd*(1-T).
    Uses market cap for equity weight (not book), per worksheet Step 4C.
    """
    bs = financials.get('balance_sheet')
    inc = financials.get('income_statement')
    info = financials.get('info') or {}

    if bs is None or inc is None or bs.empty or inc.empty:
        return None

    latest_bs = bs.iloc[:, 0]
    latest_inc = inc.iloc[:, 0]

    market_cap = info.get('marketCap')
    total_equity_book = _get(latest_bs, EQUITY_KEYS, allow_zero=False)
    total_equity = market_cap if (market_cap and market_cap > 0) else total_equity_book

    total_debt = _get(latest_bs, DEBT_KEYS)
    interest_expense = _get(latest_inc, INTEREST_KEYS)
    tax_provision = _get(latest_inc, ['Tax Provision'])
    pretax_income = _get(latest_inc, ['Pretax Income'], allow_zero=False)

    if not total_equity or total_equity <= 0:
        return None

    debt = total_debt if total_debt else 0
    total_capital = total_equity + debt
    weight_equity = total_equity / total_capital
    weight_debt = debt / total_capital

    cost_of_debt = (interest_expense / debt) if (interest_expense and debt > 0) else 0
    tax_rate = (tax_provision / pretax_income) if (tax_provision and pretax_income and pretax_income != 0) else 0.21
    return weight_equity * cost_of_equity + weight_debt * cost_of_debt * (1 - tax_rate)


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

        operating_income = _get(inc_year, OPERATING_INCOME_KEYS)
        tax_provision = _get(inc_year, ['Tax Provision'])
        pretax_income = _get(inc_year, ['Pretax Income'], allow_zero=False)
        total_equity = _get(bs_year, EQUITY_KEYS, allow_zero=False)
        total_debt = _get(bs_year, DEBT_KEYS)
        cash = _get(bs_year, CASH_KEYS)

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


def dupont_decomposition(net_income, revenue, total_assets, equity):
    """3-factor DuPont decomposition: ROE = Margin x Turnover x Leverage.

    Parameters
    ----------
    net_income : float or None
    revenue : float or None
    total_assets : float or None
    equity : float or None

    Returns
    -------
    dict or None
        {'margin': float, 'turnover': float, 'leverage': float, 'roe': float}
        or None if inputs invalid.
    """
    if (net_income is None or revenue is None or
            total_assets is None or equity is None):
        return None
    if revenue <= 0 or total_assets <= 0 or equity <= 0:
        return None

    margin = net_income / revenue
    turnover = revenue / total_assets
    leverage = total_assets / equity
    return {
        'margin': margin,
        'turnover': turnover,
        'leverage': leverage,
        'roe': margin * turnover * leverage,
    }


def compute_dupont(financials):
    """Compute DuPont decomposition from financials dict."""
    bs = financials.get('balance_sheet')
    inc = financials.get('income_statement')
    if bs is None or inc is None or bs.empty or inc.empty:
        return None
    latest_bs = bs.iloc[:, 0]
    latest_inc = inc.iloc[:, 0]
    ni = _get(latest_inc, NET_INCOME_KEYS)
    rev = _get(latest_inc, REVENUE_KEYS)
    ta = _get(latest_bs, TOTAL_ASSETS_KEYS)
    eq = _get(latest_bs, EQUITY_KEYS, allow_zero=False)
    return dupont_decomposition(ni, rev, ta, eq)


# ---------------------------------------------------------------------------
# New: Piotroski F-Score (Worksheet Step 1 quality filter)
# ---------------------------------------------------------------------------


def calculate_fundamental_growth(financials, roic_override=None):
    """Fundamental growth rate = Reinvestment Rate × ROIC.

    Reinvestment Rate = (Capex - D&A + ΔWorkingCapital) / NOPAT
    NOPAT = Operating Income × (1 - tax_rate)

    Returns dict with 'fundamental_growth', 'reinvestment_rate', 'roic_used'
    or empty dict on insufficient data.
    """
    inc = financials.get('income_statement')
    cf = financials.get('cash_flow')
    bs = financials.get('balance_sheet')
    if inc is None or inc.empty or cf is None or cf.empty:
        return {}

    latest_inc = inc.iloc[:, 0]
    latest_cf = cf.iloc[:, 0]

    operating_income = _get(latest_inc, OPERATING_INCOME_KEYS)
    if not operating_income or operating_income <= 0:
        return {}

    tax_provision = _get(latest_inc, ['Tax Provision'])
    pretax_income = _get(latest_inc, ['Pretax Income'], allow_zero=False)
    tax_rate = (tax_provision / pretax_income) if (tax_provision and pretax_income) else 0.21
    tax_rate = max(0, min(tax_rate, 0.50))  # clamp to sensible range
    nopat = operating_income * (1 - tax_rate)
    if nopat <= 0:
        return {}

    capex = _get(latest_cf, ['Capital Expenditure', 'Capital Expenditures'])
    da = _get(latest_cf, DA_KEYS)
    if capex is None or da is None:
        return {}
    capex = abs(capex)
    da = abs(da)

    # Delta working capital (need 2 years of balance sheet)
    delta_wc = 0
    if bs is not None and not bs.empty and bs.shape[1] >= 2:
        curr_bs = bs.iloc[:, 0]
        prev_bs = bs.iloc[:, 1]
        ca_curr = _get(curr_bs, CURRENT_ASSETS_KEYS) or 0
        cl_curr = _get(curr_bs, CURRENT_LIABILITIES_KEYS) or 0
        ca_prev = _get(prev_bs, CURRENT_ASSETS_KEYS) or 0
        cl_prev = _get(prev_bs, CURRENT_LIABILITIES_KEYS) or 0
        delta_wc = (ca_curr - cl_curr) - (ca_prev - cl_prev)

    net_reinvestment = capex - da + delta_wc
    reinvestment_rate = net_reinvestment / nopat
    reinvestment_rate = max(0, min(reinvestment_rate, 1.0))  # clamp [0, 1]

    # ROIC
    roic_val = roic_override
    if roic_val is None:
        roic_result = calculate_roic(financials)
        roic_val = roic_result['avg_roic'] if roic_result else None
    if roic_val is None or roic_val <= 0:
        return {}

    fundamental_growth = reinvestment_rate * roic_val
    fundamental_growth = max(0, min(fundamental_growth, 0.30))  # clamp [0, 30%]

    return {
        'fundamental_growth': fundamental_growth,
        'reinvestment_rate': reinvestment_rate,
        'roic_used': roic_val,
    }


