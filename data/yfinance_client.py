# data/yfinance_client.py
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import date

import yfinance as yf
import pandas as pd


class EmptyYahooResponseError(Exception):
    """Yahoo returned an HTTP-200 response with empty payload — almost
    always a soft rate-limit / throttle. Treated as a retryable failure
    so the caller can either retry or fall back to another data source."""


# Market caps above this are corruption, not data: the largest real market
# cap is ~$5.4T (NVDA, 2026-08), so $20T leaves ~4x headroom while still
# catching Yahoo's preferred-line blowups two orders of magnitude out.
MCAP_MAX_PLAUSIBLE = 2e13


def _sanitize_implausible_mcap(info):
    """Repair or null a corrupt market cap before it flows downstream.

    Yahoo hands preferred / OTC lines the PARENT company's common share
    count: every Fannie/Freddie preferred series reports 5.7B (FNM*) or
    3.2B (FMC*/FRE*) shares. Multiplied by the line's own quote that
    manufactures phantom mega-caps — FNMFO, a $50,000-par preferred quoted
    ~$31,500, reported a $180 TRILLION cap on 2026-08-12 and sailed through
    the mcap-min universe filter, scoring, comparison ratios and the report.

    A cap above ``MCAP_MAX_PLAUSIBLE`` is first re-derived from
    price x sharesOutstanding (repairs a corrupt packaged value whose
    components are sane). When the derived figure is equally absurd the
    share count itself is the poison — price is directly observed — so both
    fields are nulled and the missing-mcap machinery (fast_info backfill
    ran before this, prior-snapshot recovery + miss-rate alert run after)
    handles the row honestly instead of trusting garbage.

    Mutates *info* in place; returns a list describing what changed
    (empty when the cap is plausible).
    """
    changed = []
    mcap = info.get('marketCap')
    if not mcap or mcap <= MCAP_MAX_PLAUSIBLE:
        return changed
    price = info.get('currentPrice') or info.get('regularMarketPrice')
    shares = info.get('sharesOutstanding')
    derived = float(price) * float(shares) if price and shares else None
    if derived and 0 < derived <= MCAP_MAX_PLAUSIBLE:
        info['marketCap'] = derived
        changed.append('marketCap_rederived')
    else:
        info['marketCap'] = None
        changed.append('marketCap_nulled')
        if derived and derived > MCAP_MAX_PLAUSIBLE:
            info['sharesOutstanding'] = None
            changed.append('sharesOutstanding_nulled')
    return changed


def _backfill_shares_and_mcap(stock, info):
    """Backfill ``marketCap`` / ``sharesOutstanding`` from ``fast_info``.

    yfinance 1.3.0's ``.info`` intermittently omits both fields for a subset
    of tickers — roughly 10% of a full-universe run, sticky per ticker rather
    than transient, so retrying does NOT recover them (MA, LLY, GWW and ZTS
    each returned None across four consecutive attempts while ABT succeeded).
    The rest of the same ``info`` dict is fully populated (floatShares,
    currentPrice, trailingPE, all three statement frames), which is why the
    all-empty ``EmptyYahooResponseError`` throttle detector never fires here.

    The data is not actually missing upstream: ``fast_info`` reads the chart /
    quote endpoint rather than quoteSummary's defaultKeyStatistics module and
    returns both values correctly for the affected tickers. ``get_shares_full``
    is the second fallback for share count alone.

    This matters far out of proportion to two fields — everything that divides
    by share count or market cap depends on them, so losing them nulls p_tbv,
    fcf_yield, shareholder_yield, mos, fv_dispersion, pfcf, net_cash_to_mcap,
    tangible_book_per_share, every fair-value model and the Monte Carlo
    confidence fields. The 2026-07-29 run lost them for 265 of 2,244 records
    (11.8% against a 0.1% baseline) and scored those rows as *failing* five
    valuation gates on absent data.

    Mutates *info* in place and returns a list of the fields it recovered so
    the caller can record provenance. Only touches the network when a field is
    actually missing, so the common path costs nothing.
    """
    recovered = []
    if info.get('marketCap') and info.get('sharesOutstanding'):
        return recovered

    fast = None
    try:
        fast = stock.fast_info
    except Exception:
        fast = None

    def _fast(attr):
        if fast is None:
            return None
        try:
            val = getattr(fast, attr, None)
            return float(val) if val else None
        except Exception:
            return None

    if not info.get('sharesOutstanding'):
        shares = _fast('shares')
        if not shares:
            # Last resort: the shares-outstanding time series. Its final row is
            # the same figure fast_info reports, but it survives cases where
            # fast_info itself comes back bare.
            try:
                series = stock.get_shares_full(start='2020-01-01')
                if series is not None and len(series):
                    shares = float(series.iloc[-1])
            except Exception:
                shares = None
        if shares and shares > 0:
            info['sharesOutstanding'] = shares
            recovered.append('sharesOutstanding')

    if not info.get('marketCap'):
        mcap = _fast('market_cap')
        if not mcap:
            # Derive it rather than lose it: fast_info's own market_cap is
            # price x shares, so computing it here is the same number by a
            # different route when only the packaged value is absent.
            price = (info.get('currentPrice') or info.get('regularMarketPrice')
                     or _fast('last_price'))
            shares = info.get('sharesOutstanding')
            if price and shares:
                mcap = float(price) * float(shares)
        if mcap and mcap > 0:
            info['marketCap'] = mcap
            recovered.append('marketCap')

    return recovered


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
    def __init__(self, request_delay=1.0, snapshot_cache=None,
                 fetch_timeout=20, prices_dir="output/prices", run_date=None):
        self._financials_cache = {}
        self._history_cache = {}
        self._request_delay = request_delay
        self._last_request_time = 0
        self._snapshot_cache = snapshot_cache  # Optional SnapshotCache instance
        self._fetch_timeout = fetch_timeout    # hard wall-clock limit per fetch
        self._prices_dir = prices_dir          # Write-through dir for fetch_history
        # Run-START date for snapshot stamping: a 3-6h run crosses midnight,
        # and per-ticker date.today() would date post-midnight tickers run+1,
        # making a same-day replay silently miss them (load requires <= as_of).
        self.run_date = run_date

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
            # Detect Yahoo soft-throttle: HTTP 200 with empty statement frames
            # AND an info dict missing all the standard identifying fields.
            # A real response always carries at least one of symbol/shortName/longName
            # in info, even for OTC / foreign-listed tickers.
            bs = data['balance_sheet']
            inc = data['income_statement']
            cf = data['cash_flow']
            info = data['info'] or {}
            bs_empty = bs is None or (hasattr(bs, 'empty') and bs.empty)
            inc_empty = inc is None or (hasattr(inc, 'empty') and inc.empty)
            cf_empty = cf is None or (hasattr(cf, 'empty') and cf.empty)
            info_empty = not (info.get('symbol') or info.get('shortName')
                              or info.get('longName'))
            if bs_empty and inc_empty and cf_empty and info_empty:
                raise EmptyYahooResponseError(
                    f"yfinance returned empty payload for {ticker} (likely throttled)")
            # Recover marketCap / sharesOutstanding when .info drops them. Not
            # a throttle signal — this response is otherwise complete — so it
            # is repaired in place rather than raised as retryable.
            _recovered = _backfill_shares_and_mcap(stock, info)
            if _recovered:
                data['info'] = info
                data['_info_backfilled'] = _recovered
            # The inverse failure: marketCap present but absurd (preferred /
            # OTC lines carrying the parent's common share count). Runs after
            # the backfill so a backfilled cap is validated too. Sanitizing
            # here also keeps the corruption out of the snapshot cache.
            _sanitized = _sanitize_implausible_mcap(info)
            if _sanitized:
                data['info'] = info
                data['_info_sanitized'] = _sanitized
            # Growth estimates and earnings history (may fail for some tickers)
            try:
                data['growth_estimates'] = stock.growth_estimates
            except Exception:
                data['growth_estimates'] = None
            try:
                data['earnings_history'] = stock.earnings_history
            except Exception:
                data['earnings_history'] = None
            # Capture quote and reporting currencies so the analysis pipeline
            # can normalize foreign-domiciled financials to USD before any
            # valuation model runs. ``currency`` is the quote / price
            # currency; ``financialCurrency`` is the statement reporting
            # currency. They can differ — e.g., NVO (ADR) quotes in USD but
            # reports in DKK. Falls back to ``currency`` when
            # ``financialCurrency`` is absent (common for ADRs that report
            # in USD anyway).
            data['currency_quote'] = info.get('currency')
            data['currency_financial'] = (info.get('financialCurrency')
                                          or info.get('currency'))
            return data

        financials = self._retry(_fetch)
        self._financials_cache[ticker] = financials

        # Auto-save to disk cache if configured
        if self._snapshot_cache is not None:
            try:
                self._snapshot_cache.save(ticker, financials,
                                          as_of=self.run_date or date.today())
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

        fetch_failed = False
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
            fetch_failed = True
        # Only cache real responses: caching after an exception turns a
        # transient Yahoo failure into "this ticker pays no dividends" for
        # the rest of the run (DDM silently disqualified).
        if not fetch_failed:
            self._history_cache[cache_key] = dividends
        return dividends

    def fetch_history(self, ticker, period="5y"):
        cache_key = (ticker, period)
        if cache_key in self._history_cache:
            return self._history_cache[cache_key]
        stock = yf.Ticker(ticker)

        def _fetch():
            return stock.history(period=period)

        fetch_failed = False
        try:
            hist = self._retry(_fetch)
        except Exception:
            hist = None
            fetch_failed = True

        history = pd.Series(dtype=float)
        if hist is not None and not hist.empty:
            for col in ('Close', 'close'):
                if col in hist.columns:
                    history = hist[col]
                    break
            self._maybe_persist_prices(ticker, hist)

        # Only cache real responses: caching the empty Series after an
        # exception turns a transient Yahoo failure into "this ticker has no
        # price history" for the rest of the run (beta silently uncomputable).
        if not fetch_failed:
            self._history_cache[cache_key] = history
        return history

    def _maybe_persist_prices(self, ticker, hist):
        # Write Close series to <prices_dir>/<ticker>.parquet on first encounter,
        # so downstream tools (validate_ratings, portfolio_report, backtest) can
        # use it. Skip if a file already exists (don't stomp richer max-history
        # data from download_prices.py). Failures are silent — never block analysis.
        if not self._prices_dir or hist is None or hist.empty:
            return
        col = 'Close' if 'Close' in hist.columns else ('close' if 'close' in hist.columns else None)
        if col is None:
            return
        path = os.path.join(self._prices_dir, f"{ticker}.parquet")
        if os.path.exists(path):
            return
        try:
            os.makedirs(self._prices_dir, exist_ok=True)
            df = hist[[col]].copy()
            if col == 'close':
                df.columns = ['Close']
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.to_parquet(path)
        except Exception:
            pass
