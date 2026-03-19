# models/market.py
"""Market-facing data: valuation multiples, analyst consensus, composite rating."""
import pandas as pd
from models.field_keys import _get

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

    # DDM confirmation signal (gentle — only for eligible dividend payers)
    if row.get('ddm_eligible') and row.get('ddm_fv') and row.get('price'):
        ddm_fv = row['ddm_fv']
        price = row['price']
        if ddm_fv > 0:
            ddm_mos = (ddm_fv - price) / ddm_fv
            if ddm_mos > 0.10:
                score += 0.5
            elif ddm_mos < -0.20:
                score -= 0.5

    # EPV floor signal (below zero-growth value = very strong buy signal)
    epv_fv = row.get('epv_fv')
    if epv_fv and row.get('price'):
        if epv_fv > row['price']:
            score += 0.5   # trading below floor value
        elif epv_fv < row['price'] * 0.5:
            score -= 0.5   # floor value is very far below price

    # RIM confirmation signal (gentle)
    rim_mos = row.get('rim_mos')
    if rim_mos is not None:
        if rim_mos > 0.10:
            score += 0.5
        elif rim_mos < -0.20:
            score -= 0.5

    # Altman Z-Score distress signal
    altman_zone = row.get('altman_z_zone')
    if altman_zone == 'distress':
        score -= 1.0
    elif altman_zone == 'safe':
        score += 0.5

    # Beneish M-Score manipulation flag
    if row.get('beneish_flag') is True:
        score -= 2.0

    # 52-week contrarian signal
    range_pos = row.get('range_52w_position')
    if range_pos is not None and range_pos < 25 and mos is not None and mos > 0.10:
        score += 0.5

    if score >= 6:   return 'BUY'
    elif score >= 3: return 'LEAN BUY'
    elif score >= 0: return 'HOLD'
    else:            return 'PASS'
