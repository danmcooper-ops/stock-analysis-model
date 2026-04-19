# data/yf_session.py
"""Shared curl_cffi Session for yfinance with socket-level timeouts.

yfinance >= 1.0 requires a curl_cffi session (not requests.Session).
Pass the session returned by make_yf_session() to yf.Ticker(symbol, session=s)
to enforce a hard timeout on every HTTP request yfinance makes.
"""
from curl_cffi.requests import Session

# Default timeouts (seconds).  Tune here if needed.
_TIMEOUT = 15   # combined connect + read timeout


def make_yf_session(timeout=_TIMEOUT):
    """Return a curl_cffi Session that enforces timeouts.

    Usage:
        session = make_yf_session()
        ticker  = yf.Ticker('AAPL', session=session)
    """
    return Session(timeout=timeout)
