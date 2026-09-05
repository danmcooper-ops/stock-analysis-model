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

Bar selection matches ``pandas.Index.get_indexer(method='nearest')``: the bar
whose date is closest to the target, ties broken toward the LATER bar, and no
match at all beyond *max_gap_days*.
"""

import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_PRICES_DIR = 'output/prices'

# Same window as scripts/backtest.MAX_SNAP_GAP_DAYS: beyond this the data
# doesn't actually cover the requested date and the observation is dropped
# rather than measured over the wrong span.
DEFAULT_MAX_GAP_DAYS = 7


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
                  max_gap_days=DEFAULT_MAX_GAP_DAYS):
    """Closes nearest *start* and *end* for each ticker, in one scan.

    Args:
        prices_dir: directory of ``{TICKER}.parquet`` files.
        start, end: target dates (``date`` or ``YYYY-MM-DD``).
        tickers: restrict to these (default: every parquet present).
        max_gap_days: a target with no bar within this many days is unmatched.

    Returns:
        ``{ticker: {'start': float, 'end': float}}`` holding only tickers that
        resolved BOTH ends to a finite close with ``start > 0`` — the same rows
        the per-ticker path would keep.  None when the query cannot run, so the
        caller can fall back.
    """
    paths = parquet_paths(prices_dir, tickers)
    if not paths:
        return None
    con = _connect()
    if con is None:
        return None
    start_s, end_s = str(start)[:10], str(end)[:10]
    sql = """
        WITH bars AS (
            SELECT regexp_extract(filename, '([^/\\\\]+)\\.parquet$', 1) AS ticker,
                   CAST("Date" AS DATE) AS d,
                   CAST("Close" AS DOUBLE) AS close
            -- No NULL filter here on purpose: pandas writes a NaN close as a
            -- parquet NULL and reads it back as a NaN row that still occupies
            -- the index, so dropping those rows now would snap to a
            -- neighbouring bar the per-ticker path never picks.
            FROM read_parquet($paths, filename = true, union_by_name = true)
        ),
        targets(tag, target) AS (
            VALUES ('start', CAST($start AS DATE)), ('end', CAST($end AS DATE))
        ),
        ranked AS (
            SELECT b.ticker, t.tag, b.close,
                   row_number() OVER (
                       PARTITION BY b.ticker, t.tag
                       -- nearest bar; ties to the later one, matching
                       -- pandas get_indexer(method='nearest')
                       ORDER BY abs(date_diff('day', b.d, t.target)), b.d DESC
                   ) AS rn
            FROM bars b, targets t
            -- Rank over every bar, empty closes included, so the bar chosen
            -- is the one the pandas path lands on; a missing or non-finite
            -- close is rejected after selection (HAVING), never skipped past.
            WHERE abs(date_diff('day', b.d, t.target)) <= $gap
        )
        SELECT ticker,
               max(close) FILTER (WHERE tag = 'start') AS start_close,
               max(close) FILTER (WHERE tag = 'end')   AS end_close
        FROM ranked WHERE rn = 1 GROUP BY ticker
        HAVING start_close > 0 AND isfinite(start_close) AND isfinite(end_close)
    """
    try:
        rows = con.execute(sql, {'paths': paths, 'start': start_s,
                                 'end': end_s, 'gap': int(max_gap_days)}).fetchall()
    except Exception as e:
        logger.warning("price store query failed (%s); falling back", e)
        return None
    finally:
        con.close()
    return {ticker: {'start': float(s_close), 'end': float(e_close)}
            for ticker, s_close, e_close in rows
            if s_close is not None and e_close is not None}
