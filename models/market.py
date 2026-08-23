# models/market.py
"""Market-facing data: valuation multiples, analyst consensus."""
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


def extract_next_earnings(financials):
    """Next scheduled earnings date from the already-fetched yfinance info.

    Yahoo exposes the upcoming earnings window as epoch seconds in
    ``earningsTimestampStart``/``earningsTimestamp`` plus an
    ``isEarningsDateEstimate`` flag — no extra HTTP call needed. Returns
    ``{'earnings_next_date': 'YYYY-MM-DD' | None, 'earnings_date_est': bool | None}``.
    The timestamp can point at the *last* report when Yahoo hasn't scheduled the
    next one yet; consumers should compare against the run date before calling
    it "upcoming".
    """
    info = financials.get('info') or {}
    out = {'earnings_next_date': None, 'earnings_date_est': None}
    ts = info.get('earningsTimestampStart') or info.get('earningsTimestamp')
    try:
        if ts:
            from datetime import datetime, timezone
            out['earnings_next_date'] = datetime.fromtimestamp(
                int(ts), tz=timezone.utc).date().isoformat()
            est = info.get('isEarningsDateEstimate')
            out['earnings_date_est'] = bool(est) if est is not None else None
    except (ValueError, TypeError, OSError, OverflowError):
        pass
    return out


