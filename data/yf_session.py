# data/yf_session.py
"""Shared curl_cffi Session for yfinance with socket-level timeouts.

yfinance >= 1.0 requires a curl_cffi session (not requests.Session).
Pass the session returned by make_yf_session() to yf.Ticker(symbol, session=s)
to enforce a hard timeout on every HTTP request yfinance makes.

The browser yfinance impersonates is configurable through ``YF_IMPERSONATE``.
Yahoo 429s any client whose TLS fingerprint is not a real browser, so the
default is curl_cffi's current Chrome profile — the same one yfinance's own
default session uses. Behind a TLS-terminating egress proxy (the Claude Code
cloud environment the nightly routine runs in) that profile's handshake is
reset before Yahoo answers, while the older ``chrome116`` / ``safari17_0`` /
``edge101`` profiles go through; the cloud runbook sets
``YF_IMPERSONATE=chrome116``. ``install_default_session()`` applies the same
choice to the session yfinance builds for itself, so the bare ``yf.Ticker()``
and ``yf.download()`` calls in the data clients follow it too.
"""
import logging
import os

from curl_cffi.requests import Session

logger = logging.getLogger(__name__)

# Default timeouts (seconds).  Tune here if needed.
_TIMEOUT = 15   # combined connect + read timeout

#: Browser profile passed to curl_cffi (env ``YF_IMPERSONATE``, default chrome).
IMPERSONATE = os.environ.get('YF_IMPERSONATE', '').strip() or 'chrome'

_DEFAULT_INSTALLED = False


def make_yf_session(timeout=_TIMEOUT):
    """Return a curl_cffi Session that enforces timeouts.

    Usage:
        session = make_yf_session()
        ticker  = yf.Ticker('AAPL', session=session)
    """
    # Yahoo 429s any client whose TLS fingerprint is not a real browser,
    # so impersonate a browser (yfinance's own default session does the same).
    return Session(timeout=timeout, impersonate=IMPERSONATE)


def install_default_session(timeout=_TIMEOUT):
    """Make yfinance's *own* session (used by ``yf.Ticker(t)`` without a
    ``session=``, and by ``yf.download``) impersonate ``IMPERSONATE`` with
    the same timeout. Idempotent; a no-op when the profile is the default
    one yfinance would pick anyway. Returns True when a session was installed.
    """
    global _DEFAULT_INSTALLED
    if _DEFAULT_INSTALLED or IMPERSONATE == 'chrome':
        return False
    try:
        from yfinance.data import YfData
        # YfData is a singleton; passing session= re-points the existing
        # instance (SingletonMeta calls _set_session) or seeds a new one.
        YfData(session=make_yf_session(timeout))
    except Exception as e:
        logger.warning("yfinance: could not install the %s default session (%s); "
                       "bare yf.Ticker() calls keep yfinance's own profile",
                       IMPERSONATE, e)
        return False
    _DEFAULT_INSTALLED = True
    logger.info("yfinance: default session impersonates %s", IMPERSONATE)
    return True


# Applied at import so every entry point that touches any data client picks
# the override up without a per-script call; harmless when unset.
install_default_session()
