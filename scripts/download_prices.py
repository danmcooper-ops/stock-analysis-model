"""scripts/download_prices.py

Bulk download of full price history. Defaults to S&P 500; use --universe us
to backfill the full SEC EDGAR US-listed universe (matches analyze_stock.py's
--universe us flag).

Writes one Parquet file per ticker to --output-dir (default: output/prices/).
By default skips tickers whose file already exists, so the run is safely
resumable.  Use --refresh (or simply --max-age-days N) to re-download files
whose latest bar is stale — needed so forward-return backtests/calibration
have prices that actually reach the evaluation dates. Without a refresh the
skip is unconditional and nothing on disk is ever updated — which is how the
universe drifted three months stale in 2026-07.

Downloads run on a small thread pool (--price-workers) over one shared
Throttle, so the request rate is set by the interval and the workers only
decide how fully it is used. This matters most on a cold cache, where nothing
short-circuits: the stateless cloud container starts every night with an empty
output/prices/ and re-fetches the whole universe.

Usage:
    python scripts/download_prices.py                         # S&P 500 (default)
    python scripts/download_prices.py --universe us           # all US-listed
    python scripts/download_prices.py --output-dir output/prices --delay 0.4
    python scripts/download_prices.py --tickers AAPL MSFT GOOG
    python scripts/download_prices.py --refresh                 # update stale files
    python scripts/download_prices.py --refresh --max-age-days 3
    python scripts/download_prices.py --price-workers 1         # serialise
"""

import argparse
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.analyze_stock import get_sp500_tickers
from data.throttle import Throttle
from data.us_listings import fetch_us_listed_tickers
from data.yf_session import make_yf_session
from scripts.config import (PRICE_IO_WORKERS, YF_REQUEST_DELAY_MAX,
                            YF_THROTTLE_PENALTY, YF_THROTTLE_RELAX)

logger = logging.getLogger(__name__)

_YF_SESSION = None


def _yf_session():
    """Shared curl_cffi session so every Yahoo call has a hard 15s timeout."""
    global _YF_SESSION
    if _YF_SESSION is None:
        _YF_SESSION = make_yf_session()
    return _YF_SESSION


def _parquet_max_date(path):
    """Return the latest bar date in an existing parquet, or None on error."""
    try:
        df = pd.read_parquet(path, columns=[])  # index only — cheap
        idx = pd.to_datetime(df.index)
        return idx.max().date() if len(idx) else None
    except Exception as e:
        logger.debug(f"prices: index-only parquet read failed for {path}: {e}")
        try:
            df = pd.read_parquet(path)
            idx = pd.to_datetime(df.index)
            return idx.max().date() if len(idx) else None
        except Exception as e:
            logger.debug(f"prices: parquet read failed for {path}: {e}")
            return None


def _parquet_is_stub(path):
    """True when the parquet is a Close-only write-through stub.

    YFinanceClient._maybe_persist_prices seeds missing tickers with just the
    Close column of whatever short window the pipeline happened to fetch
    (5y for beta, even 5d from the portfolio path). Those files carry a
    fraction of the listing's real history, and the freshness check alone
    never upgrades them — their latest bar is current from day one. Files
    written by this script always carry full OHLCV, so a missing Volume
    column is a reliable stub marker.
    """
    try:
        import pyarrow.parquet as pq
        return 'Volume' not in pq.read_schema(path).names
    except Exception as e:
        logger.debug(f"prices: schema read failed for {path}: {e}")
        return False


def download_ticker(ticker: str, output_dir: str, delay: float,
                    refresh: bool = False, max_age_days: int = 1,
                    throttle=None) -> str:
    """Download max history for one ticker and save as Parquet.

    By default an existing file is left untouched (resumable bulk download).
    With *refresh*, an existing file is re-downloaded only when its latest bar
    is more than *max_age_days* old — so stale prices (which silently truncate
    forward-return calculations) get refreshed without re-fetching files that
    are already current. Close-only stub files (see _parquet_is_stub) are
    re-downloaded regardless of freshness so they pick up full history.

    *throttle*, when given, is a shared data.throttle.Throttle called in place
    of sleeping *delay*: one interval for every worker, so concurrency changes
    how fully the rate is used, never what the rate is. Without it the
    function sleeps *delay* itself and behaves exactly as the single-threaded
    call always did.

    Note the two early returns below happen before the throttle is touched: a
    file that is already current costs a local parquet read, not a tick.

    Returns 'skipped', 'fresh', 'ok', or an error message string.
    """
    dest = os.path.join(output_dir, f"{ticker}.parquet")
    if os.path.exists(dest):
        if not refresh:
            return "skipped"
        last = _parquet_max_date(dest)
        if last is not None and not _parquet_is_stub(dest):
            cutoff = date.today() - timedelta(days=max_age_days)
            if last >= cutoff:
                return "fresh"  # already current — no re-download needed

    if throttle is not None:
        throttle()
    else:
        time.sleep(delay)
    try:
        df = yf.Ticker(ticker, session=_yf_session()).history(period="max", auto_adjust=True)
        if df.empty:
            return "empty"
        df.index = pd.to_datetime(df.index).tz_localize(None)
        # Write through a temp file: workers write concurrently and the cloud
        # container can be restarted mid-step, and a half-written parquet is
        # worse than a missing one — the freshness check reads its index, so a
        # truncated file could read as current.
        tmp = f"{dest}.tmp.{os.getpid()}.{threading.get_ident()}"
        try:
            df.to_parquet(tmp)
            os.replace(tmp, dest)
        except Exception:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise
        return "ok"
    except Exception as e:
        return f"error: {e}"


class _EmptyStreakGovernor:
    """Turn Yahoo's empty responses into throttle pressure, but only a burst.

    An empty history means one of two unrelated things: Yahoo soft-throttled
    us, or the ticker is dead. The universe carries plenty of dead tickers, so
    penalising every empty would ratchet the interval up on a perfectly
    healthy run — the same reason YFinanceClient's 404s do not penalise.

    A soft throttle arrives as a run of empties; delistings arrive scattered
    between successes. So the interval widens only after *streak* consecutive
    empties, and any success both resets the streak and walks a widened
    interval back toward its base.
    """

    def __init__(self, throttle, streak=3):
        self._throttle = throttle
        self._streak = streak
        self._consecutive = 0
        self._lock = threading.Lock()
        self.penalties = 0

    def record(self, result):
        with self._lock:
            if result == "empty":
                self._consecutive += 1
                if self._consecutive >= self._streak:
                    self._consecutive = 0
                    self.penalties += 1
                    new = self._throttle.penalize(YF_THROTTLE_PENALTY,
                                                  cap=YF_REQUEST_DELAY_MAX)
                    logger.warning(
                        "prices: %d consecutive empty responses — widening the "
                        "request interval to %.2fs", self._streak, new)
                return
            self._consecutive = 0
        if result == "ok":
            self._throttle.relax(YF_THROTTLE_RELAX)


def main():
    parser = argparse.ArgumentParser(description="Bulk download price history")
    parser.add_argument("--output-dir", default="output/prices",
                        help="Directory to write per-ticker Parquet files")
    parser.add_argument("--delay", type=float, default=0.35,
                        help="Seconds to wait between requests (default: 0.35)")
    parser.add_argument("--tickers", nargs="+",
                        help="Override ticker list (default: from --universe)")
    parser.add_argument("--universe", choices=["sp500", "us"], default="sp500",
                        help="Ticker universe when --tickers is not given. "
                             "'sp500' = S&P 500 (default), 'us' = all US-listed "
                             "equities from SEC EDGAR (~7-10k tickers).")
    parser.add_argument("--refresh", action="store_true",
                        help="Re-download existing files whose latest bar is "
                             "older than --max-age-days (default: off — skip "
                             "existing files)")
    parser.add_argument("--price-workers", type=int, default=PRICE_IO_WORKERS,
                        metavar="N",
                        help="Concurrent download workers (default: "
                             f"{PRICE_IO_WORKERS}, env PRICE_IO_WORKERS). The "
                             "shared throttle still bounds the request rate; "
                             "1 serialises the downloads.")
    parser.add_argument("--max-age-days", type=int, default=None,
                        help="A file is re-downloaded only if its latest bar is "
                             "more than this many days old (default: 1). Giving "
                             "this flag implies --refresh, so the daily "
                             "pipeline's bare --max-age-days 7 refreshes.")
    args = parser.parse_args()

    # --max-age-days alone means "refresh anything older than N" (the daily
    # pipeline's invocation style); --refresh alone uses the 1-day default.
    refresh = args.refresh or (args.max_age_days is not None)
    max_age_days = args.max_age_days if args.max_age_days is not None else 1

    os.makedirs(args.output_dir, exist_ok=True)

    if args.tickers:
        tickers = sorted(args.tickers)
    elif args.universe == "us":
        print("Fetching US-listed ticker universe from SEC EDGAR...")
        tickers = sorted(fetch_us_listed_tickers())
    else:
        print("Fetching S&P 500 ticker list...")
        tickers = sorted(get_sp500_tickers())

    total = len(tickers)
    mode = (f"refresh (max-age {max_age_days}d)" if refresh
            else "resume (skip existing)")
    workers = max(1, args.price_workers)
    print(f"{total} tickers to process — output: {args.output_dir} — mode: {mode}")
    print(f"{workers} worker(s), {args.delay:.2f}s shared request interval\n")

    ok = skipped = fresh = empty = errors = 0
    failed = []

    # One throttle for the whole run: the interval is a per-process budget, so
    # workers decide how fully it is used, not what it is.
    throttle = Throttle(args.delay)
    governor = _EmptyStreakGovernor(throttle)

    def _fetch(ticker):
        # The pool does network only. Tallies, printing and the failure list
        # stay in the main thread below, consumed in submission order, so
        # stdout is byte-identical to the sequential run.
        result = download_ticker(ticker, args.output_dir, args.delay,
                                 refresh=refresh,
                                 max_age_days=max_age_days,
                                 throttle=throttle)
        # Back-pressure is applied here rather than on consumption: it has to
        # reach the interval before the next request goes out, not after a
        # slow ticker ahead of it in submission order finally returns.
        governor.record(result)
        return result

    t0 = time.time()
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='prices-io')
    submitted = [(t, pool.submit(_fetch, t)) for t in tickers]
    interrupted = False
    i = 0
    try:
        for i, (ticker, future) in enumerate(submitted, 1):
            result = future.result()
            if result == "ok":
                ok += 1
            elif result == "skipped":
                skipped += 1
            elif result == "fresh":
                fresh += 1
            elif result == "empty":
                empty += 1
                failed.append((ticker, "empty response"))
            else:
                errors += 1
                failed.append((ticker, result))

            print(f"  [{i:>3}/{total}] {ticker:<6} {result}")
    except KeyboardInterrupt:
        # The whole universe is already queued, so the default shutdown would
        # sit through every remaining ticker before exiting — minutes of
        # apparently hung terminal. Drop the queue and report what finished;
        # the writes are atomic, so a half-run leaves no truncated parquet.
        interrupted = True
        pool.shutdown(wait=False, cancel_futures=True)
        print(f"\n  interrupted after {i} of {total} — pending downloads "
              f"cancelled")
    finally:
        if not interrupted:
            pool.shutdown(wait=True)
    elapsed = time.time() - t0

    print(f"\n{'='*50}")
    print(f"Done.  ok={ok}  fresh={fresh}  skipped={skipped}  "
          f"empty={empty}  errors={errors}")
    ts = throttle.stats()
    print(f"Elapsed {elapsed:.0f}s — {ts['calls']} throttled fetches, "
          f"{ts['slept']:.0f}s sleeping, {ts['waited']:.0f}s queued on the "
          f"interval, interval {ts['base_delay']:.2f}s -> {ts['delay']:.2f}s "
          f"({governor.penalties} penalt{'y' if governor.penalties == 1 else 'ies'})")

    if failed:
        print("\nFailed tickers:")
        for t, reason in failed:
            print(f"  {t}: {reason}")

    # Report total size on disk
    files = [f for f in os.listdir(args.output_dir) if f.endswith(".parquet")]
    total_mb = sum(
        os.path.getsize(os.path.join(args.output_dir, f))
        for f in files
    ) / 1_048_576
    print(f"\n{len(files)} files on disk — {total_mb:.1f} MB total")


if __name__ == "__main__":
    main()
