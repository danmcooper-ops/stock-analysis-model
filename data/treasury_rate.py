# data/fred_client.py
"""
Risk-free rate fetcher.

Uses yfinance ^TNX (CBOE 10-Year Treasury Yield Index) as primary source.
Falls back to a hardcoded 4.0% if the fetch fails.
"""
import yfinance as yf

_cached_rate = None


def fetch_risk_free_rate(fallback=0.04):
    """
    Fetch the current 10-year Treasury yield from yfinance.
    Returns a decimal (e.g. 0.0425 for 4.25%).
    Caches the result for the duration of the session.
    """
    global _cached_rate
    if _cached_rate is not None:
        return _cached_rate

    try:
        # yfinance 1.x manages its own curl_cffi session internally;
        # do NOT pass a custom session — it breaks cookie/crumb auth.
        tnx = yf.Ticker('^TNX')
        price = tnx.info.get('regularMarketPrice')
        if price and 0.5 < price < 20.0:  # sanity: yield between 0.5% and 20%
            _cached_rate = round(price / 100.0, 4)
            return _cached_rate
    except Exception:
        pass

    _cached_rate = fallback
    return _cached_rate
