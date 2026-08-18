# data/treasury_rate.py
"""
Risk-free rate fetcher.

Uses yfinance ^TNX (CBOE 10-Year Treasury Yield Index) as primary source.
Falls back to a hardcoded 4.0% if the fetch fails — loudly, and tracked in
``last_rate_source`` so callers can flag results built on a fabricated rate.
"""
import sys

import yfinance as yf

_cached_rate = None

# 'live' or 'fallback' — set by fetch_risk_free_rate(); None until first call.
last_rate_source = None


def fetch_risk_free_rate(fallback=0.04):
    """
    Fetch the current 10-year Treasury yield from yfinance.
    Returns a decimal (e.g. 0.0425 for 4.25%).
    Caches the result for the duration of the session.

    On failure returns *fallback* and sets ``last_rate_source = 'fallback'``;
    every CAPM/WACC/DCF in the run inherits that rate, so the substitution
    must never be silent.
    """
    global _cached_rate, last_rate_source
    if _cached_rate is not None:
        return _cached_rate

    try:
        # yfinance 1.x manages its own curl_cffi session internally;
        # do NOT pass a custom session — it breaks cookie/crumb auth.
        tnx = yf.Ticker('^TNX')
        price = tnx.info.get('regularMarketPrice')
        if price and 0.5 < price < 20.0:  # sanity: yield between 0.5% and 20%
            _cached_rate = round(price / 100.0, 4)
            last_rate_source = 'live'
            return _cached_rate
    except Exception:
        pass

    print(f"WARNING: ^TNX fetch failed — using hardcoded fallback "
          f"risk-free rate of {fallback:.2%}. All discount rates this run "
          f"are built on this assumption.", file=sys.stderr)
    _cached_rate = fallback
    last_rate_source = 'fallback'
    return _cached_rate
