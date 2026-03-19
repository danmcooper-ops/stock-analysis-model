# models/quality.py
"""Quality scoring and risk detection: Altman Z, Beneish M, Piotroski F,
earnings quality, interest coverage, net debt metrics, revenue CAGR."""
import pandas as pd
from models.field_keys import (
    _get, EQUITY_KEYS, DEBT_KEYS, CASH_KEYS, CURRENT_ASSETS_KEYS,
    CURRENT_LIABILITIES_KEYS, NET_INCOME_KEYS, TOTAL_ASSETS_KEYS,
    EBITDA_KEYS, OPERATING_CF_KEYS, REVENUE_KEYS, OPERATING_INCOME_KEYS,
    INTEREST_KEYS, DA_KEYS, AR_KEYS, PPE_KEYS, SGA_KEYS,
    GROSS_PROFIT_KEYS, LTD_KEYS,
)

def calculate_earnings_quality(financials):
    """
    Accruals ratio: (Net Income - CFO) / Total Assets — lower is better.
    Cash conversion: CFO / Net Income — want > 0.8.
    """
    bs = financials.get('balance_sheet')
    inc = financials.get('income_statement')
    cf = financials.get('cash_flow')
    if any(x is None or (hasattr(x, 'empty') and x.empty) for x in [bs, inc, cf]):
        return {}

    latest_bs = bs.iloc[:, 0]
    latest_inc = inc.iloc[:, 0]
    latest_cf = cf.iloc[:, 0]

    net_income = _get(latest_inc, NET_INCOME_KEYS)
    total_assets = _get(latest_bs, TOTAL_ASSETS_KEYS)
    cfo = _get(latest_cf, OPERATING_CF_KEYS)

    result = {}
    if cfo is not None and net_income is not None and net_income != 0:
        result['cash_conversion'] = cfo / net_income
    if net_income is not None and cfo is not None and total_assets:
        result['accruals_ratio'] = (net_income - cfo) / total_assets
    return result


# ---------------------------------------------------------------------------
# New: Altman Z-Score (Worksheet Step 3B)
# ---------------------------------------------------------------------------


def calculate_altman_z(financials):
    """
    Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
    > 2.99: safe zone, 1.81-2.99: grey, < 1.81: distress zone.
    Uses market cap for X4 when available.
    """
    bs = financials.get('balance_sheet')
    inc = financials.get('income_statement')
    info = financials.get('info') or {}
    if bs is None or inc is None or bs.empty or inc.empty:
        return None

    latest_bs = bs.iloc[:, 0]
    latest_inc = inc.iloc[:, 0]

    total_assets = _get(latest_bs, TOTAL_ASSETS_KEYS)
    if not total_assets or total_assets <= 0:
        return None

    current_assets = _get(latest_bs, CURRENT_ASSETS_KEYS) or 0
    current_liabilities = _get(latest_bs, CURRENT_LIABILITIES_KEYS) or 0
    retained_earnings = _get(latest_bs, ['Retained Earnings']) or 0
    ebit = _get(latest_inc, ['Operating Income']) or 0
    revenue = _get(latest_inc, REVENUE_KEYS) or 0
    total_liabilities = _get(latest_bs, ['Total Liabilities Net Minority Interest', 'Total Liab']) or 1

    market_cap = info.get('marketCap')
    equity_book = _get(latest_bs, EQUITY_KEYS) or 0
    equity_val = market_cap if (market_cap and market_cap > 0) else equity_book

    x1 = (current_assets - current_liabilities) / total_assets
    x2 = retained_earnings / total_assets
    x3 = ebit / total_assets
    x4 = equity_val / total_liabilities if total_liabilities > 0 else 0
    x5 = revenue / total_assets
    return 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5


# ---------------------------------------------------------------------------
# New: Beneish M-Score (Earnings Manipulation Detection)
# ---------------------------------------------------------------------------


def calculate_beneish_m(financials):
    """Beneish M-Score: 8-variable earnings manipulation detection.

    M = -4.84 + 0.920*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI
        + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI

    M > -1.78 -> likely manipulation flag.
    Requires 2 years of balance sheet, income statement, and cash flow data.

    Returns dict with m_score, manipulation_flag, components, or None.
    """
    bs = financials.get('balance_sheet')
    inc = financials.get('income_statement')
    cf = financials.get('cash_flow')
    if bs is None or inc is None or bs.empty or inc.empty:
        return None
    if bs.shape[1] < 2 or inc.shape[1] < 2:
        return None

    curr_bs = bs.iloc[:, 0]
    prev_bs = bs.iloc[:, 1]
    curr_inc = inc.iloc[:, 0]
    prev_inc = inc.iloc[:, 1]

    # Required fields
    rev_t = _get(curr_inc, REVENUE_KEYS)
    rev_t1 = _get(prev_inc, REVENUE_KEYS)
    if not rev_t or not rev_t1 or rev_t <= 0 or rev_t1 <= 0:
        return None

    gp_t = _get(curr_inc, GROSS_PROFIT_KEYS)
    gp_t1 = _get(prev_inc, GROSS_PROFIT_KEYS)
    ta_t = _get(curr_bs, TOTAL_ASSETS_KEYS)
    ta_t1 = _get(prev_bs, TOTAL_ASSETS_KEYS)
    if not ta_t or not ta_t1 or ta_t <= 0 or ta_t1 <= 0:
        return None

    ni_t = _get(curr_inc, NET_INCOME_KEYS) or 0
    cfo_t = None
    if cf is not None and not cf.empty:
        cfo_t = _get(cf.iloc[:, 0], OPERATING_CF_KEYS)

    components = {}
    n_valid = 0

    # DSRI: Days Sales in Receivables Index
    ar_t = _get(curr_bs, AR_KEYS)
    ar_t1 = _get(prev_bs, AR_KEYS)
    if ar_t is not None and ar_t1 is not None and ar_t1 > 0:
        dsri = (ar_t / rev_t) / (ar_t1 / rev_t1)
        components['dsri'] = dsri
        n_valid += 1
    else:
        components['dsri'] = 1.0  # neutral

    # GMI: Gross Margin Index
    if gp_t is not None and gp_t1 is not None and gp_t > 0:
        gm_t = gp_t / rev_t
        gm_t1 = gp_t1 / rev_t1
        components['gmi'] = gm_t1 / gm_t if gm_t > 0 else 1.0
        n_valid += 1
    else:
        components['gmi'] = 1.0

    # AQI: Asset Quality Index
    ca_t = _get(curr_bs, CURRENT_ASSETS_KEYS) or 0
    ca_t1 = _get(prev_bs, CURRENT_ASSETS_KEYS) or 0
    ppe_t = _get(curr_bs, PPE_KEYS) or 0
    ppe_t1 = _get(prev_bs, PPE_KEYS) or 0
    if ta_t > 0 and ta_t1 > 0:
        aq_t = 1 - (ca_t + ppe_t) / ta_t
        aq_t1 = 1 - (ca_t1 + ppe_t1) / ta_t1
        components['aqi'] = aq_t / aq_t1 if abs(aq_t1) > 1e-9 else 1.0
        n_valid += 1
    else:
        components['aqi'] = 1.0

    # SGI: Sales Growth Index
    components['sgi'] = rev_t / rev_t1
    n_valid += 1

    # DEPI: Depreciation Index
    da_t = None
    da_t1 = None
    if cf is not None and not cf.empty:
        da_t = _get(cf.iloc[:, 0], DA_KEYS)
        if cf.shape[1] >= 2:
            da_t1 = _get(cf.iloc[:, 1], DA_KEYS)
    if da_t is not None and da_t1 is not None and ppe_t > 0 and ppe_t1 > 0:
        dep_rate_t = da_t / (da_t + ppe_t) if (da_t + ppe_t) > 0 else 0
        dep_rate_t1 = da_t1 / (da_t1 + ppe_t1) if (da_t1 + ppe_t1) > 0 else 0
        components['depi'] = dep_rate_t1 / dep_rate_t if dep_rate_t > 0 else 1.0
        n_valid += 1
    else:
        components['depi'] = 1.0

    # SGAI: SGA Expense Index
    sga_t = _get(curr_inc, SGA_KEYS)
    sga_t1 = _get(prev_inc, SGA_KEYS)
    if sga_t is not None and sga_t1 is not None and sga_t1 > 0:
        components['sgai'] = (sga_t / rev_t) / (sga_t1 / rev_t1)
        n_valid += 1
    else:
        components['sgai'] = 1.0

    # TATA: Total Accruals to Total Assets
    if cfo_t is not None:
        components['tata'] = (ni_t - cfo_t) / ta_t
        n_valid += 1
    else:
        components['tata'] = 0.0

    # LVGI: Leverage Index
    ltd_t = _get(curr_bs, LTD_KEYS) or _get(curr_bs, DEBT_KEYS) or 0
    cl_t = _get(curr_bs, CURRENT_LIABILITIES_KEYS) or 0
    ltd_t1 = _get(prev_bs, LTD_KEYS) or _get(prev_bs, DEBT_KEYS) or 0
    cl_t1 = _get(prev_bs, CURRENT_LIABILITIES_KEYS) or 0
    lev_t = (ltd_t + cl_t) / ta_t if ta_t > 0 else 0
    lev_t1 = (ltd_t1 + cl_t1) / ta_t1 if ta_t1 > 0 else 0
    if lev_t1 > 0:
        components['lvgi'] = lev_t / lev_t1
        n_valid += 1
    else:
        components['lvgi'] = 1.0

    # Need at least 5 of 8 valid variables
    if n_valid < 5:
        return None

    m_score = (-4.84
               + 0.920 * components['dsri']
               + 0.528 * components['gmi']
               + 0.404 * components['aqi']
               + 0.892 * components['sgi']
               + 0.115 * components['depi']
               - 0.172 * components['sgai']
               + 4.679 * components['tata']
               - 0.327 * components['lvgi'])

    return {
        'm_score': m_score,
        'manipulation_flag': m_score > -1.78,
        'components': components,
    }


# ---------------------------------------------------------------------------
# New: DuPont Decomposition
# ---------------------------------------------------------------------------


def calculate_piotroski_f(financials):
    """
    Piotroski F-Score (0-9): profitability, leverage/liquidity, operating efficiency.
    Higher = better quality. 7-9: strong, 4-6: medium, 0-3: weak.
    """
    bs = financials.get('balance_sheet')
    inc = financials.get('income_statement')
    cf = financials.get('cash_flow')
    if bs is None or inc is None or cf is None:
        return None
    if bs.empty or inc.empty or cf.empty:
        return None
    if bs.shape[1] < 2 or inc.shape[1] < 2:
        return None

    latest_bs = bs.iloc[:, 0]
    prev_bs = bs.iloc[:, 1]
    latest_inc = inc.iloc[:, 0]
    prev_inc = inc.iloc[:, 1]
    latest_cf = cf.iloc[:, 0]

    total_assets = _get(latest_bs, TOTAL_ASSETS_KEYS)
    total_assets_prev = _get(prev_bs, TOTAL_ASSETS_KEYS)
    if not total_assets or total_assets <= 0:
        return None

    score = 0

    # --- Profitability ---
    net_income = _get(latest_inc, NET_INCOME_KEYS)
    net_income_prev = _get(prev_inc, NET_INCOME_KEYS)
    cfo = _get(latest_cf, OPERATING_CF_KEYS)

    roa = net_income / total_assets if net_income else None
    # F1: ROA > 0
    if roa is not None and roa > 0:
        score += 1
    # F2: CFO > 0
    if cfo is not None and cfo > 0:
        score += 1
    # F3: CFO > Net Income (accruals quality)
    if cfo is not None and net_income is not None and cfo > net_income:
        score += 1
    # F9: ROA improved year-over-year
    if net_income_prev and total_assets_prev and total_assets_prev > 0:
        roa_prev = net_income_prev / total_assets_prev
        if roa is not None and roa > roa_prev:
            score += 1

    # --- Leverage / Liquidity ---
    total_debt = _get(latest_bs, DEBT_KEYS) or 0
    total_debt_prev = _get(prev_bs, DEBT_KEYS) or 0
    current_assets = _get(latest_bs, CURRENT_ASSETS_KEYS)
    current_liabilities = _get(latest_bs, CURRENT_LIABILITIES_KEYS)
    current_assets_prev = _get(prev_bs, CURRENT_ASSETS_KEYS)
    current_liabilities_prev = _get(prev_bs, CURRENT_LIABILITIES_KEYS)
    assets_prev = total_assets_prev or total_assets

    # F4: Leverage (debt/assets) decreased
    leverage = total_debt / total_assets
    leverage_prev = total_debt_prev / assets_prev if assets_prev else leverage
    if leverage < leverage_prev:
        score += 1

    # F5: Current ratio improved
    cr = current_assets / current_liabilities if (current_assets and current_liabilities) else None
    cr_prev = current_assets_prev / current_liabilities_prev if (current_assets_prev and current_liabilities_prev) else None
    if cr is not None and cr_prev is not None and cr > cr_prev:
        score += 1

    # F6: No share dilution (skip if data unavailable)
    shares_curr = _get(latest_bs, ['Ordinary Shares Number', 'Share Issued'])
    shares_prev = _get(prev_bs, ['Ordinary Shares Number', 'Share Issued'])
    if shares_curr is not None and shares_prev is not None and shares_curr <= shares_prev:
        score += 1

    # --- Operating Efficiency ---
    revenue = _get(latest_inc, REVENUE_KEYS)
    revenue_prev = _get(prev_inc, REVENUE_KEYS)
    gross_profit = _get(latest_inc, ['Gross Profit'])
    gross_profit_prev = _get(prev_inc, ['Gross Profit'])

    # F7: Gross margin improved
    gm = gross_profit / revenue if (gross_profit and revenue) else None
    gm_prev = gross_profit_prev / revenue_prev if (gross_profit_prev and revenue_prev) else None
    if gm is not None and gm_prev is not None and gm > gm_prev:
        score += 1

    # F8: Asset turnover improved
    at = revenue / total_assets if revenue else None
    at_prev = revenue_prev / assets_prev if (revenue_prev and assets_prev) else None
    if at is not None and at_prev is not None and at > at_prev:
        score += 1

    return score


# ---------------------------------------------------------------------------
# New: Revenue CAGR (Worksheet Step 1 screen)
# ---------------------------------------------------------------------------


def calculate_revenue_cagr(financials, years=3):
    """Revenue CAGR over up to `years` years (yfinance returns most-recent first)."""
    inc = financials.get('income_statement')
    if inc is None or inc.empty or inc.shape[1] < 2:
        return None

    revenues = []
    for col in inc.columns:
        rev = _get(inc[col], REVENUE_KEYS)
        if rev and rev > 0:
            revenues.append(rev)

    revenues = revenues[:years + 1]
    if len(revenues) < 2:
        return None

    n = len(revenues) - 1
    return (revenues[0] / revenues[-1]) ** (1 / n) - 1


# ---------------------------------------------------------------------------
# New: Relative valuation multiples (Worksheet Step 2)
# ---------------------------------------------------------------------------


def calculate_interest_coverage(financials):
    """EBIT / Interest Expense — want > 3x."""
    inc = financials.get('income_statement')
    if inc is None or inc.empty:
        return None
    latest_inc = inc.iloc[:, 0]
    ebit = _get(latest_inc, ['Operating Income'])
    interest = _get(latest_inc, INTEREST_KEYS)
    if ebit and interest and interest > 0:
        return ebit / interest
    return None



def calculate_net_debt_ebitda(financials):
    """Net Debt / EBITDA — want < 3x."""
    inc = financials.get('income_statement')
    cf = financials.get('cash_flow')
    if inc is None or inc.empty:
        return None

    latest_inc = inc.iloc[:, 0]
    net_debt = get_net_debt(financials)

    ebit = _get(latest_inc, ['Operating Income']) or 0
    da = 0
    if cf is not None and not cf.empty:
        da = _get(cf.iloc[:, 0], DA_KEYS) or 0

    ebitda = ebit + da
    if ebitda <= 0:
        return None
    return net_debt / ebitda


def get_net_debt(financials):
    """Total Debt - Cash."""
    bs = financials.get('balance_sheet')
    if bs is None or bs.empty:
        return 0
    latest_bs = bs.iloc[:, 0]
    total_debt = _get(latest_bs, DEBT_KEYS) or 0
    cash = _get(latest_bs, CASH_KEYS) or 0
    return total_debt - cash


# ---------------------------------------------------------------------------
# New: Analyst consensus (Worksheet Step 8)
# ---------------------------------------------------------------------------

