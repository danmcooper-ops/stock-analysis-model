# data/yfinance_client.py
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import date

import yfinance as yf
import pandas as pd


# Module-level executor shared across all timeout calls.  Using a single
# thread avoids the memory/thread leak of creating (and never joining) a
# fresh ThreadPoolExecutor per yfinance call.  max_workers=4 allows light
# concurrency for overlapping timeout calls while capping thread count.
_TIMEOUT_EXECUTOR = ThreadPoolExecutor(max_workers=4)


def _run_with_timeout(func, timeout_seconds):
    """Run *func* in the shared thread pool and raise TimeoutError if it
    exceeds the wall-clock limit.

    Unlike socket.setdefaulttimeout(), this works regardless of the HTTP
    library used internally (urllib3, requests, etc.) because it enforces a
    deadline on the entire call, not just per-socket idle time.
    """
    future = _TIMEOUT_EXECUTOR.submit(func)
    try:
        return future.result(timeout=timeout_seconds)
    except FuturesTimeoutError:
        future.cancel()
        raise TimeoutError(
            f"yfinance call timed out after {timeout_seconds}s"
        )


class YFinanceClient:
    def __init__(self, request_delay=0.25, snapshot_cache=None,
                 fetch_timeout=20):
        self._financials_cache = {}
        self._history_cache = {}
        self._request_delay = request_delay
        self._last_request_time = 0
        self._snapshot_cache = snapshot_cache  # Optional SnapshotCache instance
        self._fetch_timeout = fetch_timeout    # hard wall-clock limit per fetch

    def evict_financials(self, keep_tickers=None):
        """Free cached financial data.  If *keep_tickers* is given, only those
        tickers are retained; otherwise the entire cache is cleared."""
        if keep_tickers is None:
            self._financials_cache.clear()
        else:
            keep = set(keep_tickers)
            for t in list(self._financials_cache):
                if t not in keep:
                    del self._financials_cache[t]

    def clear_history_cache(self):
        """Free all cached price histories and dividend series."""
        self._history_cache.clear()

    def _throttle(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self._request_delay:
            time.sleep(self._request_delay - elapsed)
        self._last_request_time = time.time()

    def _retry(self, func, max_retries=2):
        """Run *func* with retries for transient failures.

        Timeouts are NOT retried — if a call hits the wall-clock limit, we
        accept the failure and propagate immediately.  Retrying a timeout
        only piles up orphaned threads and leaks sockets into CLOSE_WAIT,
        which poisons yfinance's internal connection pool for subsequent
        tickers.  Other exceptions (HTTP errors, parse errors) still retry.
        """
        for attempt in range(max_retries + 1):
            try:
                self._throttle()
                if self._fetch_timeout is not None:
                    return _run_with_timeout(func, self._fetch_timeout)
                else:
                    return func()
            except TimeoutError:
                # Don't retry — Yahoo is unresponsive for this ticker.
                raise
            except Exception:
                if attempt == max_retries:
                    raise
                time.sleep(1.0 * (attempt + 1))

    def fetch_financials(self, ticker, as_of=None):
        """Fetch financial data for *ticker*.

        When *as_of* is provided and a snapshot cache is configured, data is
        loaded from the disk cache and time-sliced to prevent look-ahead bias.
        Otherwise, data is fetched live from yfinance (and optionally
        auto-saved to the disk cache for future replays).

        Args:
            ticker: Stock ticker symbol.
            as_of: Optional historical date.  When set, loads from cache and
                   applies time-slicing.

        Returns:
            dict with keys: balance_sheet, income_statement, cash_flow, info,
            growth_estimates, earnings_history.
        """
        # --- Historical replay path: load from cache + time-slice ---
        if as_of is not None and self._snapshot_cache is not None:
            cached = self._snapshot_cache.load(ticker, as_of)
            if cached is not None:
                from data.time_slice import slice_financials_as_of
                return slice_financials_as_of(cached, as_of)
            # No cache hit for historical date — return None so caller
            # knows this ticker has no data for the requested date.
            return None

        # --- Live fetch path (unchanged behaviour when no cache) ---
        if ticker in self._financials_cache:
            return self._financials_cache[ticker]
        # NOTE: Do NOT pass a custom session — yfinance requires its own
        # curl_cffi session for Yahoo's API.  Connection pool hygiene is
        # handled by the 20s timeout + no-retry-on-timeout policy instead.
        stock = yf.Ticker(ticker)

        def _fetch():
            data = {
                'balance_sheet': stock.balance_sheet,
                'income_statement': stock.financials,
                'cash_flow': stock.cashflow,
                'info': stock.info,
            }
            # Growth estimates and earnings history (may fail for some tickers)
            try:
                data['growth_estimates'] = stock.growth_estimates
            except Exception:
                data['growth_estimates'] = None
            try:
                data['earnings_history'] = stock.earnings_history
            except Exception:
                data['earnings_history'] = None
            return data

        financials = self._retry(_fetch)
        self._financials_cache[ticker] = financials

        # Auto-save to disk cache if configured
        if self._snapshot_cache is not None:
            try:
                self._snapshot_cache.save(ticker, financials, as_of=date.today())
            except Exception:
                pass  # Cache write failures are non-fatal

        return financials

    def fetch_dividends(self, ticker, period="10y"):
        """Fetch historical dividend payments.

        Returns a pandas Series indexed by date with dividend amounts,
        or an empty Series if unavailable.
        """
        cache_key = (ticker, period, 'dividends')
        if cache_key in self._history_cache:
            return self._history_cache[cache_key]
        stock = yf.Ticker(ticker)

        def _fetch():
            return stock.dividends

        try:
            dividends = self._retry(_fetch)
            if dividends is None:
                dividends = pd.Series(dtype=float)
            # yfinance >=1.2 may return a single-column DataFrame instead of
            # a Series.  Normalise to Series so all callers stay consistent.
            if isinstance(dividends, pd.DataFrame):
                if dividends.empty:
                    dividends = pd.Series(dtype=float)
                else:
                    dividends = dividends.iloc[:, 0]
        except Exception:
            dividends = pd.Series(dtype=float)
        self._history_cache[cache_key] = dividends
        return dividends

    def fetch_history(self, ticker, period="5y"):
        cache_key = (ticker, period)
        if cache_key in self._history_cache:
            return self._history_cache[cache_key]
        stock = yf.Ticker(ticker)

        def _fetch():
            hist = stock.history(period=period)
            # Guard against empty DataFrame or renamed column ('close' vs 'Close')
            for col in ('Close', 'close'):
                if col in hist.columns:
                    return hist[col]
            return pd.Series(dtype=float)

        history = self._retry(_fetch)
        self._history_cache[cache_key] = history
        return history
