"""FX rate store: one daily history per currency, kept on disk between runs.

Replaces the in-memory year-end table in data/fx_client.py as the source of
exchange rates. Every rate is **USD per one unit** of the currency (the
``{CCY}USD=X`` convention fx_client already multiplies by), stored as
``<dir>/<CCY>.parquet`` with a ``rate`` and a ``source`` column per day.

Sources. Yahoo's ``{CCY}USD=X`` closes are the primary series: they cover
every currency the universe reports in (32 on 2026-10-26) and are current to
the run date. The Fed's H.10 noon buying rates on FRED (``DEX*``) cover 21 of
them, published weekly, so they lag a few days. H.10 is used two ways:

- a day Yahoo lacks is filled from it;
- a day where the two disagree by more than ``DISAGREE_TOL`` takes H.10,
  the reference rate. The median gap is 0.2-0.5%; the 37 days over 5% on
  2026-10-27 were Yahoo's (EUR repeated 1.557 on the 8th of several 2008
  months, ZAR read 0.067 for 0.055 on 2024-11-15, NOK +51% on 2020-03-20),
  bar CHF on 2015-01-15, where H.10's noon rate is already past the SNB's
  removal of the floor and Yahoo's bar is not.

An isolated bar more than ``SPIKE_TOL`` from both neighbours while the
neighbours agree with each other is a bad print and is dropped. All 23 on
2026-10-27 were decimal shifts (COP 0.27 between two days at 0.00052, VND
1000x). A level shift (ARS fell ~55% on 2023-12-13) moves the neighbours
too and is kept. Two bad bars in a row are not caught; for a currency H.10
covers, the cross-check catches them.

Lookups are as of a date, never "the latest", so a re-run of a past date
converts at that date's rate. A missing rate is ``None``: callers blank the
value rather than pass a native-currency number through as dollars, which
is what fx_client does on a failed fetch.

``refresh()`` refetches every currency in full each run (about 35 small
requests): Yahoo revises a currency's last bar after the fact, and a
freshness check would freeze a live intraday bar. A fetch that fails, or
returns far less history than the file holds, keeps the file as it is.
"""

import logging
import math
import os
from datetime import date

import pandas as pd

logger = logging.getLogger(__name__)

FX_DIR_ENV = 'FX_STORE_DIR'
DEFAULT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output', 'fx')

# Always refreshed, whether or not a company reports in them, so the
# Currencies tab can show the majors.
MAJORS = ('EUR', 'JPY', 'GBP', 'CHF', 'CNY', 'CAD', 'AUD', 'NZD', 'SEK',
          'NOK', 'HKD', 'SGD', 'KRW', 'INR', 'MXN', 'BRL')

# FRED H.10 series per currency: (series id, True when it is quoted as USD
# per unit already; otherwise it is units per USD and is inverted).
H10_SERIES = {
    'EUR': ('DEXUSEU', True), 'GBP': ('DEXUSUK', True),
    'AUD': ('DEXUSAL', True), 'NZD': ('DEXUSNZ', True),
    'JPY': ('DEXJPUS', False), 'CAD': ('DEXCAUS', False),
    'CNY': ('DEXCHUS', False), 'BRL': ('DEXBZUS', False),
    'CHF': ('DEXSZUS', False), 'MXN': ('DEXMXUS', False),
    'KRW': ('DEXKOUS', False), 'SEK': ('DEXSDUS', False),
    'HKD': ('DEXHKUS', False), 'ZAR': ('DEXSFUS', False),
    'INR': ('DEXINUS', False), 'TWD': ('DEXTAUS', False),
    'THB': ('DEXTHUS', False), 'DKK': ('DEXDNUS', False),
    'NOK': ('DEXNOUS', False), 'SGD': ('DEXSIUS', False),
    'MYR': ('DEXMAUS', False),
}

DISAGREE_TOL = 0.05       # Yahoo vs H.10 on the same day
SPIKE_TOL = 0.25          # an isolated bar this far off both neighbours
SPIKE_NEIGHBOUR_TOL = 0.05  # ...while the neighbours agree within this
MAX_LOOKBACK_DAYS = 7     # an as-of rate older than this is no rate
MIN_AVG_COVERAGE = 0.5    # share of weekdays an average needs bars for
MIN_KEEP_RATIO = 0.8      # a refetch shorter than this keeps the old file
FRED_START = date(1995, 1, 1)


def _iso(ccy):
    return (ccy or '').strip().upper()


def _as_date(d):
    if isinstance(d, date):
        return d
    return pd.Timestamp(d).date()


def clean_series(s):
    """*s* (date-indexed rates) with non-finite, non-positive and isolated
    spike bars removed, sorted with one bar per day. Returns
    ``(series, n_dropped)``."""
    if s is None or len(s) == 0:
        return pd.Series(dtype=float), 0
    s = pd.to_numeric(pd.Series(s), errors='coerce')
    s.index = pd.DatetimeIndex(pd.to_datetime(s.index)).normalize()
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    n0 = len(s)
    s = s[s.notna() & (s > 0) & s.map(lambda v: math.isfinite(v))]
    s = s[~s.index.duplicated(keep='last')].sort_index()
    if len(s) >= 3:
        v = s.to_numpy()
        prev, cur, nxt = v[:-2], v[1:-1], v[2:]
        off = ((abs(cur / prev - 1) > SPIKE_TOL)
               & (abs(cur / nxt - 1) > SPIKE_TOL)
               & (abs(nxt / prev - 1) <= SPIKE_NEIGHBOUR_TOL))
        keep = [True] + [not x for x in off] + [True]
        s = s[keep]
    return s.astype(float), n0 - len(s)


def merge_sources(yahoo, fred):
    """One series from the Yahoo and H.10 series (both cleaned, USD per
    unit). Returns ``(frame, stats)``: *frame* has ``rate`` and ``source``
    columns; *stats* counts the days each source supplied and the Yahoo
    days H.10 overrode."""
    yahoo = yahoo if yahoo is not None else pd.Series(dtype=float)
    fred = fred if fred is not None else pd.Series(dtype=float)
    idx = yahoo.index.union(fred.index)
    y = yahoo.reindex(idx)
    f = fred.reindex(idx)
    both = y.notna() & f.notna()
    disagree = both & ((y / f - 1).abs() > DISAGREE_TOL)
    use_fred = y.isna() | disagree
    rate = y.where(~use_fred, f)
    frame = pd.DataFrame({'rate': rate,
                          'source': ['fred' if u else 'yahoo' for u in use_fred]},
                         index=idx)
    frame = frame[frame['rate'].notna()]
    frame.index.name = 'date'
    stats = {'bars': len(frame),
             'yahoo': int((frame['source'] == 'yahoo').sum()),
             'fred': int((frame['source'] == 'fred').sum()),
             'overridden': int(disagree.sum())}
    return frame, stats


def fetch_yahoo(ccy, session=None):
    """Full daily history of ``{ccy}USD=X`` closes, or an empty series."""
    import yfinance as yf
    from data.yf_session import make_yf_session
    # yf.download without session= would replace yfinance's singleton
    # session and drop the YF_IMPERSONATE override (data/yf_session.py).
    df = yf.download(f'{ccy}USD=X', period='max', interval='1d',
                     progress=False, auto_adjust=False,
                     session=session or make_yf_session())
    if df is None or df.empty:
        return pd.Series(dtype=float)
    close = df['Close'] if 'Close' in df.columns.get_level_values(0) else df.iloc[:, [0]]
    if getattr(close, 'ndim', 1) == 2:
        close = close.squeeze('columns')
    return close


_FRED = None


def _default_fred():
    global _FRED
    if _FRED is None:
        from data.fred_client import FREDClient
        _FRED = FREDClient()
    return _FRED


def fetch_fred(ccy, fred_client=None):
    """H.10 history for *ccy* as USD per unit, or an empty series."""
    spec = H10_SERIES.get(ccy)
    if spec is None:
        return pd.Series(dtype=float)
    sid, usd_per_unit = spec
    obs = (fred_client or _default_fred()).fetch_series(sid, start=FRED_START) or {}
    s = pd.Series({pd.Timestamp(k): v for k, v in obs.items()}, dtype=float)
    if not usd_per_unit:
        s = 1.0 / s[s > 0]
    return s


class FXStore:
    """Reads and refreshes the per-currency parquets under *fx_dir*."""

    def __init__(self, fx_dir=None):
        self.fx_dir = fx_dir or os.environ.get(FX_DIR_ENV) or DEFAULT_DIR
        self._frames = {}

    # -- files -------------------------------------------------------------

    def path(self, ccy):
        return os.path.join(self.fx_dir, f'{_iso(ccy)}.parquet')

    def currencies(self):
        """Currencies with a stored history."""
        try:
            names = os.listdir(self.fx_dir)
        except OSError:
            return []
        return sorted(n[:-8] for n in names if n.endswith('.parquet'))

    def frame(self, ccy):
        """The stored ``rate``/``source`` frame for *ccy*, or None."""
        ccy = _iso(ccy)
        if ccy not in self._frames:
            try:
                df = pd.read_parquet(self.path(ccy))
                df.index = pd.DatetimeIndex(df.index)
                self._frames[ccy] = df
            except (OSError, ValueError) as e:
                logger.debug("fx_store: no history for %s: %s", ccy, e)
                self._frames[ccy] = None
        return self._frames[ccy]

    def _write(self, ccy, frame):
        os.makedirs(self.fx_dir, exist_ok=True)
        dest = self.path(ccy)
        tmp = f'{dest}.tmp.{os.getpid()}'
        frame.to_parquet(tmp)
        os.replace(tmp, dest)
        self._frames[_iso(ccy)] = frame

    # -- lookups -----------------------------------------------------------

    def rate_info(self, ccy, as_of, max_lookback_days=MAX_LOOKBACK_DAYS):
        """``(rate, bar_date, source)`` for the last bar on or before
        *as_of*, or None when there is none within *max_lookback_days*.
        USD is ``(1.0, as_of, 'identity')``."""
        ccy = _iso(ccy)
        as_of = _as_date(as_of)
        if ccy == 'USD':
            return 1.0, as_of, 'identity'
        df = self.frame(ccy)
        if df is None or df.empty:
            return None
        hist = df.loc[:pd.Timestamp(as_of)]
        if hist.empty:
            return None
        bar = hist.index[-1].date()
        if (as_of - bar).days > max_lookback_days:
            return None
        row = hist.iloc[-1]
        return float(row['rate']), bar, row['source']

    def rate(self, ccy, as_of, max_lookback_days=MAX_LOOKBACK_DAYS):
        """USD per unit of *ccy* as of *as_of*, or None."""
        info = self.rate_info(ccy, as_of, max_lookback_days)
        return None if info is None else info[0]

    # Balance-sheet items convert at the period-end (closing) rate.
    close_rate = rate

    def avg_rate(self, ccy, start, end):
        """Mean daily rate over [*start*, *end*], the rate income and
        cash-flow items convert at. None when the bars cover under
        ``MIN_AVG_COVERAGE`` of the window's weekdays."""
        ccy = _iso(ccy)
        start, end = _as_date(start), _as_date(end)
        if ccy == 'USD':
            return 1.0
        df = self.frame(ccy)
        if df is None or df.empty or end < start:
            return None
        win = df.loc[pd.Timestamp(start):pd.Timestamp(end), 'rate']
        weekdays = len(pd.bdate_range(start, end))
        if weekdays == 0 or len(win) < MIN_AVG_COVERAGE * weekdays:
            return None
        return float(win.mean())

    # -- refresh -----------------------------------------------------------

    def refresh(self, currencies, yahoo_fetch=None, fred_fetch=None):
        """Refetch *currencies* from both sources and rewrite each file.

        Returns ``{ccy: stats}``; stats carry ``status`` 'ok', 'kept' (the
        fetch failed or came back short, so the old file stands) or
        'missing' (nothing fetched and nothing stored). Never raises for a
        single currency. The fetchers default to this module's, looked up at
        call time so the test suite can stub them out globally."""
        yahoo_fetch = yahoo_fetch or fetch_yahoo
        fred_fetch = fred_fetch or fetch_fred
        out = {}
        for ccy in sorted({_iso(c) for c in currencies} - {'', 'USD'}):
            try:
                out[ccy] = self._refresh_one(ccy, yahoo_fetch, fred_fetch)
            except Exception as e:
                logger.warning("fx_store: refresh failed for %s: %s", ccy, e)
                out[ccy] = {'status': 'kept' if self.frame(ccy) is not None
                            else 'missing', 'error': str(e)}
        return out

    def _refresh_one(self, ccy, yahoo_fetch, fred_fetch):
        try:
            y_raw = yahoo_fetch(ccy)
        except Exception as e:
            logger.warning("fx_store: Yahoo fetch failed for %s: %s", ccy, e)
            y_raw = None
        try:
            f_raw = fred_fetch(ccy)
        except Exception as e:
            logger.warning("fx_store: FRED fetch failed for %s: %s", ccy, e)
            f_raw = None
        y, y_drop = clean_series(y_raw)
        f, f_drop = clean_series(f_raw)
        frame, stats = merge_sources(y, f)
        stats.update(spikes_dropped=y_drop + f_drop)
        old = self.frame(ccy)
        if frame.empty or (old is not None
                           and len(frame) < MIN_KEEP_RATIO * len(old)):
            stats['status'] = 'kept' if old is not None else 'missing'
            if old is not None:
                logger.warning("fx_store: %s refetch returned %d bars against "
                               "%d stored; keeping the stored history",
                               ccy, len(frame), len(old))
            return stats
        self._write(ccy, frame)
        stats.update(status='ok', first=frame.index[0].date().isoformat(),
                     last=frame.index[-1].date().isoformat())
        return stats


def snapshot_currencies(results_dir, before=None):
    """Reporting and quote currencies in the newest snapshot in the store
    for *results_dir* (dated before *before* when given), or an empty set
    when there is no store."""
    try:
        from data.snapshot_store import SnapshotStore
        store = SnapshotStore.for_results_dir(results_dir, allow_db=False)
    except Exception as e:
        logger.warning("fx_store: snapshot store unavailable: %s", e)
        return set()
    if store is None:
        return set()
    with store:
        d = store.latest_date(before=before)
        if d is None:
            return set()
        cols = {c.lower() for c in store.columns()}
        found = set()
        for col in ('currency_financial', 'currency_quote'):
            if col in cols:
                df = store.query(f"SELECT DISTINCT {col} AS c FROM results "
                                 "WHERE date = ?", [d])
                found |= {_iso(c) for c in df['c'] if isinstance(c, str)}
        return found - {'', 'USD'}


