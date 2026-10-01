# data/treasury_rate.py
"""
Risk-free rate fetcher.

Four sources, tried in order, every one bounds-checked (0.5%-20%):

  1. ``live``            yfinance ^TNX (CBOE 10-Year Treasury Yield Index)
  2. ``fred``            FRED DGS10, the same series from the Fed (keyed when
                         FRED_API_KEY is set, keyless fredgraph.csv otherwise)
  3. ``prior_snapshot``  the rate the newest prior ``results_<date>`` recorded
                         in its meta, if that run measured it (``live`` or
                         ``fred``) within PRIOR_SNAPSHOT_MAX_AGE_DAYS
  4. ``fallback``        a hardcoded 4.0% — loudly, and tracked in
                         ``last_rate_source`` so callers flag results built
                         on a fabricated rate

Until 2026-09-30 there was only 1 and 4. That night Yahoo rate-limited the
host before the first ticker, ^TNX 429'd, and every CAPM/WACC/DCF in a
seven-hour run was built on 4.00% while the market had the 10-year at 5.29%
— 129bp on every discount rate, with FRED_API_KEY set and the prior day's
5.25% sitting in output/. A substitution from 2 or 3 is still logged as a
WARNING: it is a real rate, but not today's print.
"""
import logging
from datetime import date, timedelta

import yfinance as yf

from data.yf_session import make_yf_session

logger = logging.getLogger(__name__)

_cached_rate = None

_YF_SESSION = None

# How old the prior snapshot may be for its rate to stand in for today's.
PRIOR_SNAPSHOT_MAX_AGE_DAYS = 7
# A prior run's rate is only borrowed when that run measured it. Borrowing a
# borrowed (or fabricated) rate would chain the substitution indefinitely.
PRIOR_SOURCES_ACCEPTED = ('live', 'fred')
# Observations further back than this are not "today's" 10-year.
FRED_MAX_LOOKBACK_DAYS = 10


def _yf_session():
    """Shared curl_cffi session so every Yahoo call has a hard 15s timeout."""
    global _YF_SESSION
    if _YF_SESSION is None:
        _YF_SESSION = make_yf_session()
    return _YF_SESSION


# 'live', 'fred', 'prior_snapshot' or 'fallback' — set by
# fetch_risk_free_rate(); None until first call.
last_rate_source = None
# Where a substituted rate came from: the FRED observation date or the
# prior snapshot's date. None for 'live' and 'fallback'.
last_rate_detail = None


def _plausible_pct(pct):
    """Sanity: a 10-year yield between 0.5% and 20%, as a percent figure."""
    try:
        return pct is not None and 0.5 < float(pct) < 20.0
    except (TypeError, ValueError):
        return False


def _from_yahoo():
    # Shared curl_cffi session enforces a hard socket timeout on every
    # request (supported wiring: yf.Ticker(symbol, session=s)).
    tnx = yf.Ticker('^TNX', session=_yf_session())
    price = (tnx.info or {}).get('regularMarketPrice')
    if _plausible_pct(price):
        return round(float(price) / 100.0, 4), None
    return None


def _from_fred(as_of):
    from data.fred_client import FREDClient
    client = FREDClient()
    obs = client.fetch_series('DGS10', start=as_of - timedelta(days=45), end=as_of)
    obs_date, value = FREDClient._as_of_value(obs, as_of, max_lookback_days=FRED_MAX_LOOKBACK_DAYS)
    if obs_date is not None and _plausible_pct(value):
        return round(float(value) / 100.0, 4), obs_date.isoformat()
    return None


def _from_prior_snapshot(run_date, results_dir):
    from data.snapshot_store import prior_snapshot_file, read_snapshot, split_snapshot
    prior = prior_snapshot_file(results_dir, run_date)
    if not prior:
        return None
    prior_date, path = prior
    age = (run_date - date.fromisoformat(prior_date)).days
    if age > PRIOR_SNAPSHOT_MAX_AGE_DAYS:
        logger.info('treasury: prior snapshot %s is %d days old — not borrowing its rate',
                    prior_date, age)
        return None
    meta, _rows = split_snapshot(read_snapshot(path))
    source = meta.get('risk_free_rate_source')
    if source not in PRIOR_SOURCES_ACCEPTED:
        logger.info('treasury: prior snapshot %s recorded a %r rate — not borrowing it',
                    prior_date, source)
        return None
    rate = meta.get('risk_free_rate')
    if rate is not None and _plausible_pct(float(rate) * 100.0):
        return round(float(rate), 4), prior_date
    return None


def fetch_risk_free_rate(fallback=0.04, run_date=None, results_dir='output',
                         refresh=False):
    """
    The current 10-year Treasury yield as a decimal (0.0425 for 4.25%).
    Caches the result for the duration of the session (*refresh* re-fetches).

    Sources are tried in the order in the module docstring; the one that
    answered is in ``last_rate_source`` (and ``last_rate_detail``). On total
    failure returns *fallback* with ``last_rate_source = 'fallback'``; every
    CAPM/WACC/DCF in the run inherits that rate, so the substitution must
    never be silent.
    """
    global _cached_rate, last_rate_source, last_rate_detail
    if _cached_rate is not None and not refresh:
        return _cached_rate
    today = run_date or date.today()

    sources = (
        ('live', _from_yahoo),
        ('fred', lambda: _from_fred(today)),
        ('prior_snapshot', lambda: _from_prior_snapshot(today, results_dir)),
    )
    for name, fetch in sources:
        try:
            got = fetch()
        except Exception as e:
            logger.warning(f"treasury: {name} source failed: {e}")
            got = None
        if not got:
            continue
        rate, detail = got
        _cached_rate, last_rate_source, last_rate_detail = rate, name, detail
        if name != 'live':
            logger.warning(f"treasury: ^TNX unavailable — using the {name} 10-year "
                           f"({detail}) of {rate:.2%} for every discount rate this run")
        return _cached_rate

    logger.warning(f"^TNX, FRED and the prior snapshot all failed — using hardcoded "
                   f"fallback risk-free rate of {fallback:.2%}. All discount rates "
                   f"this run are built on this assumption.")
    _cached_rate = fallback
    last_rate_source = 'fallback'
    last_rate_detail = None
    return _cached_rate
