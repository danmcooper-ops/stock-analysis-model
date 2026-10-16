# data/price_store.py
"""Bulk queries over the per-ticker price parquets in ``output/prices``.

``scripts/download_prices.py`` writes one ``{TICKER}.parquet`` of full OHLCV
per ticker (a few thousand files), and readers historically opened them one
at a time with pandas.  The backtester's forward-return pass needs two closes
per ticker per (snapshot, horizon), so a per-ticker open costs thousands of
parquet reads for a single measurement pass.

DuckDB reads the whole set in one scan, so :func:`window_closes` answers
"the close nearest date A and the close nearest date B, for these tickers"
as a single query.  Every function here returns None when it cannot answer
(no duckdb, missing directory, no matching files) so callers keep their
existing per-ticker pandas path as the fallback.

Bar selection (:func:`select_window`, which both the DuckDB query and the
per-ticker pandas path apply, so the two cannot drift):

- Only a *valid* bar counts: a finite, positive close. Yahoo writes a row for
  a session it has not finished with a NaN close (SPY's 2026-10-06 row), and
  a missing print says nothing about the price, so it is skipped, never
  selected and never counted as the file's last bar.
- The start is **as of** the start date: the last valid bar on or before it.
  The nearest bar could be a later one, and a later bar is information the
  snapshot did not have.
- The end is the valid bar nearest the end date, ties to the later bar.
- Neither may be more than *max_gap_days* from its target.
- With *require_reach*, the ticker's own valid bars must reach the end date.
  A file that stops short would otherwise be measured to whatever bar it last
  had within the gap, and that short window is frozen by the return sidecar.
- A selected bar that is an isolated spike (both valid neighbours more than
  SPIKE_RATIO away, in the same direction) is a bad print, not a price:
  GRKZF's $0.001 close on 2026-08-14 between $15.52 closes.
- A window whose valid closes never change over STALE_MIN_BARS or more bars
  is a dead quote, not a 0% return (forward-filled Tiingo series, untraded
  OTC lines).
"""

import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_PRICES_DIR = 'output/prices'

# Same window as scripts/backtest.MAX_SNAP_GAP_DAYS: beyond this the data
# doesn't actually cover the requested date and the observation is dropped
# rather than measured over the wrong span.
DEFAULT_MAX_GAP_DAYS = 7

# A selected bar more than this factor away from BOTH valid neighbours, in the
# same direction, is a one-day bad print.
SPIKE_RATIO = 3.0

# A window with at least this many valid bars and a single distinct close is a
# dead quote.
STALE_MIN_BARS = 5


def _valid(close):
    import math
    return close is not None and math.isfinite(close) and close > 0


def _is_spike(close, prev, nxt):
    if prev is None or nxt is None:
        return False
    return ((close > SPIKE_RATIO * prev and close > SPIKE_RATIO * nxt)
            or (close * SPIKE_RATIO < prev and close * SPIKE_RATIO < nxt))


def window_verdict(start, start_prev, start_next, end, end_prev, end_next,
                   n_bars, lo, hi):
    """Why a selected (start, end) pair is not a measurement, or None.

    The arguments are what bar selection gathered: each end's close and its
    valid neighbours, and the count / min / max of the valid closes from the
    start bar to the end bar. Shared by both selection paths.
    """
    if _is_spike(start, start_prev, start_next) or _is_spike(end, end_prev, end_next):
        return 'spike'
    if n_bars >= STALE_MIN_BARS and lo == hi:
        return 'stale'
    return None


def select_window(series, start, end, max_gap_days=DEFAULT_MAX_GAP_DAYS,
                  require_reach=True):
    """``{'start', 'end'}`` closes for one price series, or None.

    *series* is a pandas Series of closes indexed by date (any order, NaNs
    allowed). The reference implementation of the rules in the module
    docstring; :func:`window_closes` is the same thing in one DuckDB scan.
    """
    import pandas as pd
    s = pd.Series(series, dtype=float)
    s.index = pd.to_datetime(s.index)
    if getattr(s.index, 'tz', None) is not None:
        s.index = s.index.tz_localize(None)
    s = s[[_valid(v) for v in s.values]].sort_index()
    s = s[~s.index.duplicated(keep='last')]
    if s.empty:
        return None
    st, en = pd.Timestamp(start), pd.Timestamp(end)
    if require_reach and s.index[-1] < en:
        return None
    before = s.index[s.index <= st]
    if not len(before) or (st - before[-1]).days > max_gap_days:
        return None
    si = len(before) - 1
    gaps = abs((s.index - en).days)
    best = gaps.min()
    if best > max_gap_days:
        return None
    ei = max(i for i, g in enumerate(gaps) if g == best)   # ties to the later bar

    def nb(i):
        return (float(s.iloc[i - 1]) if i > 0 else None,
                float(s.iloc[i + 1]) if i + 1 < len(s) else None)

    lo_i, hi_i = sorted((si, ei))
    span = s.iloc[lo_i:hi_i + 1]
    if window_verdict(float(s.iloc[si]), *nb(si), float(s.iloc[ei]), *nb(ei),
                      len(span), float(span.min()), float(span.max())):
        return None
    return {'start': float(s.iloc[si]), 'end': float(s.iloc[ei])}


def valid_span(path):
    """``(first, last)`` dates of a price parquet's valid closes, or None.

    The file's index alone overstates it: a trailing NaN-close row (an
    unfinished session) would read as a current file and as a bar on the
    benchmark's eval date. Raises on an unreadable file.
    """
    import pandas as pd
    df = pd.read_parquet(path, columns=['Close'])
    close = df['Close']
    good = close[[_valid(v) for v in close.values]]
    if good.empty:
        return None
    idx = pd.to_datetime(good.index)
    return idx.min().date(), idx.max().date()


def trim_trailing_invalid(df):
    """*df* without its trailing rows whose Close is not a valid price."""
    if 'Close' not in df or df.empty:
        return df
    ok = [_valid(v) for v in df['Close'].values]
    last = max((i for i, v in enumerate(ok) if v), default=-1)
    return df.iloc[:last + 1]


def parquet_paths(prices_dir, tickers=None):
    """Existing ``{ticker}.parquet`` paths under *prices_dir*.

    With *tickers*, only those (silently skipping ones with no file); without,
    every parquet in the directory.  Returns [] when the directory is absent.
    """
    if not prices_dir or not os.path.isdir(prices_dir):
        return []
    if tickers is None:
        return sorted(os.path.join(prices_dir, f) for f in os.listdir(prices_dir)
                      if f.endswith('.parquet'))
    out = []
    for t in dict.fromkeys(tickers):          # de-dupe, keep order
        if not t:
            continue
        p = os.path.join(prices_dir, f"{t}.parquet")
        if os.path.exists(p):
            out.append(p)
    return out


def _connect():
    try:
        import duckdb
    except ImportError as e:
        logger.debug("price store unavailable (no duckdb): %s", e)
        return None
    return duckdb.connect(':memory:')


def window_closes(prices_dir, start, end, tickers=None,
                  max_gap_days=DEFAULT_MAX_GAP_DAYS, require_reach=True):
    """Start and end closes for each ticker, in one scan.

    Args:
        prices_dir: directory of ``{TICKER}.parquet`` files.
        start, end: target dates (``date`` or ``YYYY-MM-DD``).
        tickers: restrict to these (default: every parquet present).
        max_gap_days: a target with no valid bar within this many days is
            unmatched.
        require_reach: the ticker's valid bars must reach *end*.

    Returns:
        ``{ticker: {'start': float, 'end': float}}`` holding only the tickers
        :func:`select_window` would answer for. None when the query cannot
        run, so the caller can fall back.
    """
    paths = parquet_paths(prices_dir, tickers)
    if not paths:
        return None
    con = _connect()
    if con is None:
        return None
    start_s, end_s = str(start)[:10], str(end)[:10]
    sql = """
        WITH raw AS (
            SELECT regexp_extract(filename, '([^/\\\\]+)\\.parquet$', 1) AS ticker,
                   CAST("Date" AS DATE) AS d,
                   CAST("Close" AS DOUBLE) AS close
            FROM read_parquet($paths, filename = true, union_by_name = true)
        ),
        valid AS (
            SELECT ticker, d, last(close) AS close
            FROM raw
            WHERE close IS NOT NULL AND isfinite(close) AND close > 0
            GROUP BY ticker, d
        ),
        nb AS (
            SELECT ticker, d, close,
                   lag(close)  OVER (PARTITION BY ticker ORDER BY d) AS prev,
                   lead(close) OVER (PARTITION BY ticker ORDER BY d) AS nxt
            FROM valid
        ),
        reach AS (SELECT ticker, max(d) AS last_d FROM valid GROUP BY ticker),
        s AS (
            SELECT *, row_number() OVER (PARTITION BY ticker ORDER BY d DESC) AS rn
            FROM nb
            WHERE d <= CAST($start AS DATE)
              AND date_diff('day', d, CAST($start AS DATE)) <= $gap
        ),
        e AS (
            SELECT *, row_number() OVER (
                       PARTITION BY ticker
                       ORDER BY abs(date_diff('day', d, CAST($end AS DATE))), d DESC
                   ) AS rn
            FROM nb
            WHERE abs(date_diff('day', d, CAST($end AS DATE))) <= $gap
        ),
        pick AS (
            SELECT s.ticker, s.d AS sd, s.close AS sc, s.prev AS sp, s.nxt AS sn,
                   e.d AS ed, e.close AS ec, e.prev AS ep, e.nxt AS en
            FROM s JOIN e ON s.ticker = e.ticker AND s.rn = 1 AND e.rn = 1
            JOIN reach r ON r.ticker = s.ticker
            WHERE NOT $reach OR r.last_d >= CAST($end AS DATE)
        )
        SELECT p.ticker, p.sc, p.sp, p.sn, p.ec, p.ep, p.en,
               count(*), min(v.close), max(v.close)
        FROM pick p JOIN valid v
          ON v.ticker = p.ticker
         AND v.d BETWEEN least(p.sd, p.ed) AND greatest(p.sd, p.ed)
        GROUP BY ALL
    """
    try:
        rows = con.execute(sql, {'paths': paths, 'start': start_s,
                                 'end': end_s, 'gap': int(max_gap_days),
                                 'reach': bool(require_reach)}).fetchall()
    except Exception as e:
        logger.warning("price store query failed (%s); falling back", e)
        return None
    finally:
        con.close()
    out = {}
    for ticker, sc, sp, sn, ec, ep, en, n, lo, hi in rows:
        if window_verdict(sc, sp, sn, ec, ep, en, n, lo, hi) is None:
            out[ticker] = {'start': float(sc), 'end': float(ec)}
    return out


def last_valid_dates(prices_dir, tickers=None):
    """``{ticker: date}`` of each file's last valid close, in one scan; None
    when the query cannot run."""
    paths = parquet_paths(prices_dir, tickers)
    if not paths:
        return {} if tickers is not None else None
    con = _connect()
    if con is None:
        return None
    sql = """
        SELECT regexp_extract(filename, '([^/\\\\]+)\\.parquet$', 1) AS ticker,
               max(CAST("Date" AS DATE))
        FROM read_parquet($paths, filename = true, union_by_name = true)
        WHERE "Close" IS NOT NULL AND isfinite(CAST("Close" AS DOUBLE))
          AND CAST("Close" AS DOUBLE) > 0
        GROUP BY 1
    """
    try:
        rows = con.execute(sql, {'paths': paths}).fetchall()
    except Exception as e:
        logger.warning("price store last-bar query failed (%s)", e)
        return None
    finally:
        con.close()
    return {t: d for t, d in rows}


def asof_closes(prices_dir, dates, tickers=None, max_gap_days=DEFAULT_MAX_GAP_DAYS):
    """The last finite, positive close on or before each date, in one scan.

    As-of rather than nearest: a return "as of date D" must never peek at a
    bar after D. Rows with a NULL/NaN/non-positive close are skipped (the
    previous good bar stands), and a bar more than *max_gap_days* before the
    date is no answer — a delisted ticker's last print does not carry on.

    Returns ``{date_str: {ticker: (bar_date_str, close)}}`` holding only the
    pairs that resolved; None when the query cannot run.
    """
    paths = parquet_paths(prices_dir, tickers)
    if not paths or not dates:
        return None if not paths else {}
    con = _connect()
    if con is None:
        return None
    ds = sorted({str(d)[:10] for d in dates})
    sql = """
        WITH bars AS (
            SELECT regexp_extract(filename, '([^/\\\\]+)\\.parquet$', 1) AS ticker,
                   CAST("Date" AS DATE) AS d,
                   CAST("Close" AS DOUBLE) AS close
            FROM read_parquet($paths, filename = true, union_by_name = true)
            WHERE "Close" IS NOT NULL
        ),
        good AS (SELECT * FROM bars WHERE isfinite(close) AND close > 0),
        tk AS (SELECT DISTINCT ticker FROM good),
        targets AS (
            SELECT tk.ticker, CAST(u.t AS DATE) AS target
            FROM tk, (SELECT unnest($dates) AS t) u
        )
        SELECT targets.ticker, targets.target, good.d, good.close
        FROM targets ASOF JOIN good
          ON targets.ticker = good.ticker AND targets.target >= good.d
        WHERE date_diff('day', good.d, targets.target) <= $gap
    """
    try:
        rows = con.execute(sql, {'paths': paths, 'dates': ds,
                                 'gap': int(max_gap_days)}).fetchall()
    except Exception as e:
        logger.warning("price store as-of query failed (%s)", e)
        return None
    finally:
        con.close()
    out = {d: {} for d in ds}
    for ticker, target, bar_d, close in rows:
        out[target.isoformat()][ticker] = (bar_d.isoformat(), float(close))
    return out


def stopped_trading(prices_dir, tickers, as_of, max_lag_bars, spy_max_age_days,
                    benchmark='SPY'):
    """Tickers whose price history shows they stopped trading:
    ``{ticker: (last_bar_iso, lag_bars)}``.

    Yahoo keeps answering ``.info`` for a delisted symbol with its frozen last
    quote, so nothing in the fundamentals notices an acquired company. Its
    parquet does: a fetch of a dead symbol returns only its last few bars.
    A ticker is stopped when its last bar as of *as_of* is more than
    *max_lag_bars* *benchmark* trading days behind the benchmark's own last
    bar. Only meaningful when the caller's parquets were refreshed just
    before (run.sh step 03 for Phase 1, step 05e for the render).

    Lag is counted in benchmark bars and measured against the benchmark's
    last bar rather than *as_of*, so a download that failed wholesale (every
    file a day behind, the benchmark included) stops nothing, and later bars
    are ignored for a past *as_of*. Returns {} — the rule is off — when the
    benchmark parquet is missing or more than *spy_max_age_days* old, or the
    query cannot run; a ticker with no parquet is never reported, since
    absence says nothing about trading.
    """
    if not prices_dir or as_of is None or not tickers:
        return {}
    path = os.path.join(prices_dir, f"{benchmark}.parquet")
    if not os.path.exists(path):
        return {}
    import pandas as pd
    try:
        bench = pd.read_parquet(path)['Close']
        bench.index = pd.to_datetime(bench.index).tz_localize(None)
    except Exception as e:
        logger.warning("stopped-trading check: unreadable %s parquet (%s)", benchmark, e)
        return {}
    as_of_ts = pd.Timestamp(as_of)
    bench = bench[(bench > 0) & (bench.index <= as_of_ts)].dropna().sort_index()
    if bench.empty or (as_of_ts - bench.index[-1]).days > spy_max_age_days:
        return {}
    # No gap limit: the question is exactly how far behind the last bar is.
    bars = asof_closes(prices_dir, [as_of_ts.date().isoformat()],
                       tickers=sorted(set(tickers) - {benchmark}),
                       max_gap_days=100_000)
    if not bars:
        return {}
    days = bench.index
    out = {}
    for tk, (bar_day, _close) in next(iter(bars.values())).items():
        lag = int((days > pd.Timestamp(bar_day)).sum())
        if lag > max_lag_bars:
            out[tk] = (bar_day, lag)
    return out


def mass_stop(n_stopped, n_total, max_share, floor):
    """True when too many names read as stopped for it to be delistings.

    A handful of companies delist on any night; hundreds at once is a price
    refresh that failed, and dropping them would empty the run or report.
    """
    return n_stopped > max(floor, max_share * n_total)
