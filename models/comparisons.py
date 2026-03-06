# models/comparisons.py
import pandas as pd


def _get(series, keys, allow_zero=True):
    for key in keys:
        if key in series.index:
            val = series[key]
            if pd.notna(val) and (allow_zero or val != 0):
                return val
    return None


# ---------------------------------------------------------------------------
# Existing functions (ROIC, WACC updated to use market cap, ratios)
# ---------------------------------------------------------------------------

def compute_ratios(financials):
    ratios = {}
    bs = financials.get('balance_sheet')
    inc = financials.get('income_statement')
    if bs is None or inc is None or bs.empty or inc.empty:
        return ratios

    latest_bs = bs.iloc[:, 0]
    latest_inc = inc.iloc[:, 0]

    total_equity = _get(latest_bs, ['Stockholders Equity', 'Total Stockholder Equity', 'Common Stock Equity'], allow_zero=False)
    total_assets = _get(latest_bs, ['Total Assets'], allow_zero=False)
    total_liabilities = _get(latest_bs, ['Total Liabilities Net Minority Interest', 'Total Liab'])
    current_assets = _get(latest_bs, ['Current Assets', 'Total Current Assets'])
    current_liabilities = _get(latest_bs, ['Current Liabilities', 'Total Current Liabilities'], allow_zero=False)
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
    total_equity_book = _get(latest_bs, ['Stockholders Equity', 'Total Stockholder Equity', 'Common Stock Equity'], allow_zero=False)
    total_equity = market_cap if (market_cap and market_cap > 0) else total_equity_book

    total_debt = _get(latest_bs, ['Total Debt', 'Long Term Debt'])
    interest_expense = _get(latest_inc, ['Interest Expense', 'Interest Expense Non Operating'])
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

        operating_income = _get(inc_year, ['Operating Income', 'Total Operating Income As Reported'])
        tax_provision = _get(inc_year, ['Tax Provision'])
        pretax_income = _get(inc_year, ['Pretax Income'], allow_zero=False)
        total_equity = _get(bs_year, ['Stockholders Equity', 'Total Stockholder Equity', 'Common Stock Equity'], allow_zero=False)
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


# ---------------------------------------------------------------------------
# New: Earnings Quality (Worksheet Step 3B)
# ---------------------------------------------------------------------------

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

    net_income = _get(latest_inc, ['Net Income', 'Net Income Common Stockholders'])
    total_assets = _get(latest_bs, ['Total Assets'])
    cfo = _get(latest_cf, ['Operating Cash Flow', 'Cash From Operations'])

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

    total_assets = _get(latest_bs, ['Total Assets'])
    if not total_assets or total_assets <= 0:
        return None

    current_assets = _get(latest_bs, ['Current Assets', 'Total Current Assets']) or 0
    current_liabilities = _get(latest_bs, ['Current Liabilities', 'Total Current Liabilities']) or 0
    retained_earnings = _get(latest_bs, ['Retained Earnings']) or 0
    ebit = _get(latest_inc, ['Operating Income']) or 0
    revenue = _get(latest_inc, ['Total Revenue']) or 0
    total_liabilities = _get(latest_bs, ['Total Liabilities Net Minority Interest', 'Total Liab']) or 1

    market_cap = info.get('marketCap')
    equity_book = _get(latest_bs, ['Stockholders Equity', 'Total Stockholder Equity', 'Common Stock Equity']) or 0
    equity_val = market_cap if (market_cap and market_cap > 0) else equity_book

    x1 = (current_assets - current_liabilities) / total_assets
    x2 = retained_earnings / total_assets
    x3 = ebit / total_assets
    x4 = equity_val / total_liabilities if total_liabilities > 0 else 0
    x5 = revenue / total_assets
    return 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5


# ---------------------------------------------------------------------------
# New: Piotroski F-Score (Worksheet Step 1 quality filter)
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

    total_assets = _get(latest_bs, ['Total Assets'])
    total_assets_prev = _get(prev_bs, ['Total Assets'])
    if not total_assets or total_assets <= 0:
        return None

    score = 0

    # --- Profitability ---
    net_income = _get(latest_inc, ['Net Income', 'Net Income Common Stockholders'])
    net_income_prev = _get(prev_inc, ['Net Income', 'Net Income Common Stockholders'])
    cfo = _get(latest_cf, ['Operating Cash Flow', 'Cash From Operations'])

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
    total_debt = _get(latest_bs, ['Total Debt', 'Long Term Debt']) or 0
    total_debt_prev = _get(prev_bs, ['Total Debt', 'Long Term Debt']) or 0
    current_assets = _get(latest_bs, ['Current Assets', 'Total Current Assets'])
    current_liabilities = _get(latest_bs, ['Current Liabilities', 'Total Current Liabilities'])
    current_assets_prev = _get(prev_bs, ['Current Assets', 'Total Current Assets'])
    current_liabilities_prev = _get(prev_bs, ['Current Liabilities', 'Total Current Liabilities'])
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
    revenue = _get(latest_inc, ['Total Revenue'])
    revenue_prev = _get(prev_inc, ['Total Revenue'])
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
        rev = _get(inc[col], ['Total Revenue'])
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

def compute_relative_multiples(financials):
    """Pull valuation multiples and market data from yfinance info."""
    info = financials.get('info') or {}
    cf = financials.get('cash_flow')

    market_cap = info.get('marketCap')
    pfcf = None
    if cf is not None and not cf.empty and market_cap:
        fcf = _get(cf.iloc[:, 0], ['Free Cash Flow'])
        if fcf and fcf > 0:
            pfcf = market_cap / fcf

    return {
        'pe': info.get('trailingPE') or info.get('forwardPE'),
        'ev_ebitda': info.get('enterpriseToEbitda'),
        'ev_revenue': info.get('enterpriseToRevenue'),
        'pb': info.get('priceToBook'),
        'peg': info.get('pegRatio'),
        'pfcf': pfcf,
        'market_cap': market_cap,
        'enterprise_value': info.get('enterpriseValue'),
        'shares': info.get('sharesOutstanding'),
        'price': info.get('currentPrice') or info.get('regularMarketPrice'),
        'div_yield': (info.get('dividendRate') / (info.get('currentPrice') or info.get('regularMarketPrice')))
                     if (info.get('dividendRate') and (info.get('currentPrice') or info.get('regularMarketPrice')))
                     else None,
        'payout_ratio': info.get('payoutRatio'),
        'eps': info.get('trailingEps'),
    }


# ---------------------------------------------------------------------------
# New: Balance sheet health (Worksheet Step 3C)
# ---------------------------------------------------------------------------

def calculate_interest_coverage(financials):
    """EBIT / Interest Expense — want > 3x."""
    inc = financials.get('income_statement')
    if inc is None or inc.empty:
        return None
    latest_inc = inc.iloc[:, 0]
    ebit = _get(latest_inc, ['Operating Income'])
    interest = _get(latest_inc, ['Interest Expense', 'Interest Expense Non Operating'])
    if ebit and interest and interest > 0:
        return ebit / interest
    return None


def calculate_net_debt_ebitda(financials):
    """Net Debt / EBITDA — want < 3x."""
    bs = financials.get('balance_sheet')
    inc = financials.get('income_statement')
    cf = financials.get('cash_flow')
    if bs is None or inc is None or bs.empty or inc.empty:
        return None

    latest_bs = bs.iloc[:, 0]
    latest_inc = inc.iloc[:, 0]

    total_debt = _get(latest_bs, ['Total Debt', 'Long Term Debt']) or 0
    cash = _get(latest_bs, ['Cash And Cash Equivalents', 'Cash Cash Equivalents And Short Term Investments', 'Cash Financial']) or 0
    net_debt = total_debt - cash

    ebit = _get(latest_inc, ['Operating Income']) or 0
    da = 0
    if cf is not None and not cf.empty:
        da = _get(cf.iloc[:, 0], ['Depreciation And Amortization', 'Depreciation Amortization Depletion']) or 0

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
    total_debt = _get(latest_bs, ['Total Debt', 'Long Term Debt']) or 0
    cash = _get(latest_bs, ['Cash And Cash Equivalents', 'Cash Cash Equivalents And Short Term Investments', 'Cash Financial']) or 0
    return total_debt - cash


# ---------------------------------------------------------------------------
# New: Analyst consensus (Worksheet Step 8)
# ---------------------------------------------------------------------------

def compute_analyst_consensus(financials):
    """
    Pull analyst sentiment from yfinance info.
    recommendationKey: 'strong_buy' | 'buy' | 'hold' | 'sell' | 'strong_sell'
    """
    info = financials.get('info') or {}
    return {
        'rec_key': info.get('recommendationKey'),
        'num_analysts': info.get('numberOfAnalystOpinions'),
        'target_mean': info.get('targetMeanPrice'),
        'target_high': info.get('targetHighPrice'),
        'target_low': info.get('targetLowPrice'),
    }


# ---------------------------------------------------------------------------
# New: Composite Buy/Sell/Hold rating (Worksheet Decision Matrix, Step 8)
# ---------------------------------------------------------------------------

def compute_rating(row):
    """
    Composite rating from worksheet decision matrix signals.
    Scoring: BUY >= 6 | LEAN BUY 3-5 | HOLD 0-2 | PASS < 0
    """
    score = 0.0

    # DCF Margin of Safety (Step 5)
    mos = row.get('mos')
    if mos is not None:
        if mos > 0.20:   score += 2
        elif mos > 0.10: score += 1
        elif mos < -0.10: score -= 1

    # ROIC-WACC spread (Step 1)
    spread = row.get('spread')
    if spread is not None:
        if spread > 0.20:   score += 2
        elif spread > 0.10: score += 1
        elif spread > 0.05: score += 0.5

    # Piotroski F-Score (Step 1 quality filter)
    pf = row.get('piotroski')
    if pf is not None:
        if pf >= 7:   score += 2
        elif pf <= 3: score -= 2

    # Altman Z (Step 3C)
    az = row.get('altman_z')
    if az is not None:
        if az > 2.99:   score += 1
        elif az < 1.81: score -= 2

    # Earnings quality: cash conversion (Step 3B)
    cc = row.get('cash_conv')
    if cc is not None:
        if cc > 0.8:   score += 1
        elif cc < 0.5: score -= 1

    # Analyst consensus (Step 8)
    rec = (row.get('analyst_rec') or '').lower().replace('_', '').replace(' ', '')
    if rec in ('strongbuy',):   score += 2
    elif rec == 'buy':           score += 1
    elif rec in ('sell', 'strongsell'): score -= 1

    # Revenue growth (Step 1)
    cagr = row.get('rev_cagr')
    if cagr is not None:
        if cagr > 0.10:  score += 1
        elif cagr < 0:   score -= 1

    # Leverage (Step 3C)
    de = row.get('de')
    if de is not None:
        if de > 2.0:   score -= 1
        elif de < 1.0: score += 0.5

    # Interest coverage (Step 3C)
    ic = row.get('int_cov')
    if ic is not None:
        if ic > 5:    score += 0.5
        elif ic < 1:  score -= 2

    # ROIC trend direction (moat durability signal)
    roic_by_year = row.get('roic_by_year')
    wacc = row.get('wacc')
    if roic_by_year and len(roic_by_year) >= 2:
        sorted_years = sorted(roic_by_year.keys())
        first_roic = roic_by_year[sorted_years[0]]
        last_roic = roic_by_year[sorted_years[-1]]
        roic_delta = last_roic - first_roic
        if roic_delta > 0.02:
            # Improving ROIC — strengthening moat
            score += 1.5
        elif roic_delta > 0.005:
            score += 0.5
        elif roic_delta < -0.02:
            # Declining ROIC — deteriorating moat
            score -= 1.5
            # Extra penalty if declining toward WACC
            if wacc is not None and last_roic < wacc * 1.1:
                score -= 1
        elif roic_delta < -0.005:
            score -= 0.5

    if score >= 6:   return 'BUY'
    elif score >= 3: return 'LEAN BUY'
    elif score >= 0: return 'HOLD'
    else:            return 'PASS'


def compute_ratios_trend(financials):
    bs = financials.get('balance_sheet')
    inc = financials.get('income_statement')

    if bs is None or inc is None or bs.empty or inc.empty:
        return None

    common_years = sorted(bs.columns.intersection(inc.columns))
    if len(common_years) == 0:
        return None

    ratios_by_year = {}
    for year in common_years:
        year_bs = bs[year]
        year_inc = inc[year]
        label = str(year.year) if hasattr(year, 'year') else str(year)

        total_equity = _get(year_bs, ['Stockholders Equity', 'Total Stockholder Equity', 'Common Stock Equity'], allow_zero=False)
        total_assets = _get(year_bs, ['Total Assets'], allow_zero=False)
        total_liabilities = _get(year_bs, ['Total Liabilities Net Minority Interest', 'Total Liab'])
        current_assets = _get(year_bs, ['Current Assets', 'Total Current Assets'])
        current_liabilities = _get(year_bs, ['Current Liabilities', 'Total Current Liabilities'], allow_zero=False)
        net_income = _get(year_inc, ['Net Income', 'Net Income Common Stockholders'])

        year_ratios = {}
        if total_equity and net_income is not None:
            year_ratios['ROE'] = net_income / total_equity
        if total_equity and total_liabilities is not None:
            year_ratios['Debt-to-Equity'] = total_liabilities / total_equity
        if current_assets is not None and current_liabilities:
            year_ratios['Current Ratio'] = current_assets / current_liabilities
        if total_assets and net_income is not None:
            year_ratios['ROA'] = net_income / total_assets

        ratios_by_year[label] = year_ratios

    sorted_years = sorted(ratios_by_year.keys())
    trends = {}
    if len(sorted_years) >= 2:
        first_year = sorted_years[0]
        last_year = sorted_years[-1]
        for ratio_name in ['ROE', 'Debt-to-Equity', 'Current Ratio', 'ROA']:
            first_val = ratios_by_year[first_year].get(ratio_name)
            last_val = ratios_by_year[last_year].get(ratio_name)
            if first_val is not None and last_val is not None:
                diff = last_val - first_val
                if abs(diff) < 0.01:
                    trends[ratio_name] = 'stable'
                elif ratio_name == 'Debt-to-Equity':
                    trends[ratio_name] = 'improving' if diff < 0 else 'declining'
                else:
                    trends[ratio_name] = 'improving' if diff > 0 else 'declining'

    return {'ratios_by_year': ratios_by_year, 'trends': trends}
