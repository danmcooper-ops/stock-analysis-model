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
import threading

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


# ---------------------------------------------------------------------------
# Crumb poisoning (2026-09-25/26 nights: ~60% of the universe lost .info)
# ---------------------------------------------------------------------------
# yfinance 1.7.0 assigns the /v1/test/getcrumb response text to its cached
# crumb BEFORE validating it (YfData._get_crumb_basic / _get_crumb_csrf). When
# getcrumb is rate-limited the text is "Too Many Requests", and the next call
# reuses it because the only check is `crumb is not None`. Every quoteSummary
# and v7/quote request then carries that string and is answered 401 ("Invalid
# Crumb" / "User is unable to access this feature"), which `.info` logs and
# swallows, returning no price, sector or enterprise value. yfinance's own
# recovery re-mints the crumb on each 401 — one more getcrumb call per ticker,
# which keeps getcrumb rate-limited — so under Phase 1's steady load the state
# persisted for the whole screen while a fresh process worked fine.
#
# YFinanceClient detects the 401 per fetch (auth_errors_this_thread), clears
# the bad crumb (reset_crumb) and backs off; these are the two primitives.

_tls = threading.local()


class _AuthErrorCounter(logging.Handler):
    """Counts yfinance's logged HTTP 401s per thread. `.info` swallows the
    HTTPError and only logs it, so the log record is the one signal that
    reaches us; per-thread so a pool worker's count is its own fetch's."""

    def emit(self, record):
        try:
            if 'HTTP Error 401' in record.getMessage():
                _tls.auth_errors = getattr(_tls, 'auth_errors', 0) + 1
        except Exception:  # a counting handler must never break a fetch
            pass


_AUTH_COUNTER = _AuthErrorCounter(level=logging.ERROR)
logging.getLogger('yfinance').addHandler(_AUTH_COUNTER)


def auth_errors_this_thread():
    """Running count of yfinance HTTP 401s logged on the calling thread."""
    return getattr(_tls, 'auth_errors', 0)


def crumb_is_valid(crumb):
    """A real crumb is a short token with no whitespace or markup;
    "Too Many Requests\r\n" and HTML error pages are not."""
    if not isinstance(crumb, str):
        return False
    c = crumb.strip()
    return 0 < len(c) <= 64 and not any(ch.isspace() for ch in c) and '<' not in c


def reset_crumb():
    """Drop yfinance's cached crumb if it is not a real one, so the next
    request mints a fresh one. Returns True when a bad crumb was cleared.
    Touches yfinance internals, so it degrades to a no-op if they move."""
    try:
        from yfinance.data import YfData
        yd = YfData()
        with yd._cookie_lock:
            if yd._crumb is not None and not crumb_is_valid(yd._crumb):
                yd._crumb = None
                return True
    except Exception as e:
        logger.debug("yfinance: could not inspect the cached crumb (%s)", e)
    return False


# Applied at import so every entry point that touches any data client picks
# the override up without a per-script call; harmless when unset.
install_default_session()
