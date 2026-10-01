# data/yfinance_client.py
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import date

import yfinance as yf
import pandas as pd

from data.throttle import Throttle
from data.yf_session import auth_errors_this_thread, install_default_session, reset_crumb

logger = logging.getLogger(__name__)


class EmptyYahooResponseError(Exception):
    """Yahoo returned an HTTP-200 response with empty payload — almost
    always a soft rate-limit / throttle. Treated as a retryable failure
    so the caller can either retry or fall back to another data source."""


class YahooAuthError(EmptyYahooResponseError):
    """Yahoo answered HTTP 401 ("Invalid Crumb") while fetching `.info`.
    yfinance swallows it and returns an info dict with no price, sector or
    EV, so without this check the ticker went on through the screen with
    those fields missing. It is the auth-token form of a soft throttle
    (see data/yf_session.py), so it subclasses EmptyYahooResponseError: every
    caller that already retries or falls back on a throttle handles it.

    ``data`` carries the fetch's result (statements, but no quote fields) so
    fetch_financials can still keep it once retries are exhausted — the
    worst case is then exactly the pre-check behaviour, never less data."""

    def __init__(self, msg, data=None):
        super().__init__(msg)
        self.data = data


class YahooRateLimitError(EmptyYahooResponseError):
    """Yahoo answered HTTP 429 (yfinance's ``YFRateLimitError``, "Too Many
    Requests. Rate limited. Try after a while.").

    The 2026-09-30 run: 8,030 of 9,186 fetches raised this, and because it was
    a plain Exception to every defence built for the soft throttle it
    penalised nothing, tripped no valve, was never re-queued and never fell
    back to SEC data — Phase 1 spent six hours sending ~24,000 doomed
    requests at the base interval, which is what keeps an IP rate-limited.
    It is the explicit form of the throttle, so it subclasses
    EmptyYahooResponseError: the throttle widens, the Phase-1 alarm counts
    it, and every caller that retries or falls back on a throttle handles it.
    The breaker below is what stops a sustained block from costing the night.
    """


def _is_rate_limited(exc):
    """True for Yahoo's HTTP 429, however it reaches us: yfinance's own
    exception class, or its message text when wrapped by a caller."""
    try:
        from yfinance.exceptions import YFRateLimitError
        if isinstance(exc, YFRateLimitError):
            return True
    except ImportError:  # older yfinance: fall through to the text check
        pass
    msg = str(exc)
    return 'Too Many Requests' in msg or 'Rate limited' in msg or 'HTTP Error 429' in msg


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


# Reported vs implied share count disagreement beyond this fraction means
# the reported figure is not the count the price and market cap are quoted
# on (one share class of a dual-class filer, ordinary shares behind an ADS).
SHARES_MCAP_TOLERANCE = 0.20


def _reconcile_shares_with_mcap(info, tolerance=SHARES_MCAP_TOLERANCE):
    """Replace ``sharesOutstanding`` with the count implied by market cap
    when the two disagree materially.

    Yahoo's ``sharesOutstanding`` is one share class for dual-class filers
    (GOOGL reports Class A only) and the ordinary-share count for many ADRs,
    while ``marketCap`` is the whole company at the quoted price. Every
    per-share model divides a firm-wide numerator (EV, NOPAT, book value)
    by this count, so a one-class figure inflates each fair value by the
    class ratio. ``impliedSharesOutstanding`` is Yahoo's own reconciliation
    of the two; market cap / price is the fallback.

    Mutates *info* in place: the reported count is preserved under
    ``sharesOutstanding_reported``. Returns a list describing what changed
    (empty when the counts agree or either side is unknown).
    """
    shares = info.get('sharesOutstanding')
    price = info.get('currentPrice') or info.get('regularMarketPrice')
    mcap = info.get('marketCap')
    implied = info.get('impliedSharesOutstanding')
    if not implied or implied <= 0:
        implied = (float(mcap) / float(price)) if (mcap and price and price > 0) else None
    if not shares or shares <= 0 or not implied or implied <= 0:
        return []
    if abs(float(shares) - float(implied)) / float(implied) <= tolerance:
        return []
    info['sharesOutstanding_reported'] = shares
    info['sharesOutstanding'] = float(implied)
    return ['sharesOutstanding_implied']


# A shares-outstanding series whose newest observation is older than this is
# no evidence about today's count. Yahoo's series for the Fannie/Freddie and
# Ameren preferred lines stops at 2021-03-17 (FMCCG: 460,190,016 forever), so
# trusting its last row refilled a nulled count with a five-year-old figure
# and rebuilt the phantom caps ($5-11B) this module exists to remove.
SHARES_SERIES_MAX_AGE_DAYS = 540


def _recent_series_shares(stock, max_age_days=SHARES_SERIES_MAX_AGE_DAYS):
    """Last value of ``get_shares_full`` when it is recent, else None.

    Raises whatever ``get_shares_full`` raises; callers already guard it.
    A series without a datetime index cannot be dated and is taken as is.
    """
    series = stock.get_shares_full(start='2020-01-01')
    if series is None or not len(series):
        return None
    series = series.dropna()
    if not len(series):
        return None
    if isinstance(series.index, pd.DatetimeIndex):
        # Yahoo repeats dates in this series; the newest row wins.
        series = series.sort_index()
        newest = series.index[-1]
        now = pd.Timestamp.now(tz=newest.tz)
        if (now - newest).days > max_age_days:
            return None
    return float(series.iloc[-1])


# A packaged share count with no packaged market cap is only trusted when an
# independent Yahoo source agrees with it. The tolerance is loose because the
# sources report different as-of dates; contamination is 100-5000x off.
SHARES_CORROBORATION_TOL = 0.25


def _null_uncorroborated_shares(stock, info):
    """Null a share count that no independent source corroborates.

    Yahoo assigns the PARENT company's common share count to preferred and
    secondary OTC lines: on 2026-08-12 every Fannie Mae preferred series
    (FNM*) carried 5,738,840,064 shares and every Freddie Mac series
    (FMC*/FRE*) 3,221,329,920 — dozens of distinct securities reporting one
    count (same pattern for the Ameren Illinois, AGNC and Valley National
    preferreds). Multiplied by each line's own quote that manufactured
    $24B-$90B phantom caps for ~40 tickers, all BELOW ``MCAP_MAX_PLAUSIBLE``,
    so ``_sanitize_implausible_mcap`` only ever caught FNMFO ($180T).

    The class has a structural signature (verified live 2026-08-14): ``.info``
    packages ``sharesOutstanding`` but NO ``marketCap``, and neither
    ``fast_info`` nor ``get_shares_full`` knows the security at all — Yahoo
    itself refuses to compute a cap for these lines. The phantom caps came
    from OUR own price x shares derivation in ``_backfill_shares_and_mcap``.

    Rule: when the packaged cap is absent, the packaged share count only
    survives if ``fast_info`` (its share count, or the count implied by its
    market cap at the current price) or the ``get_shares_full`` series agrees
    with it within ``SHARES_CORROBORATION_TOL``. Uncorroborated counts are
    nulled — this must run BEFORE the backfill so no cap is ever derived from
    them; the row then goes through the missing-mcap machinery honestly.

    True commons are unaffected: they either package ``marketCap`` (this
    check never engages) or hit the known both-fields-dropped failure
    (``sharesOutstanding`` absent too, so there is nothing to distrust and
    the ``fast_info`` backfill recovers them).

    Mutates *info* in place; returns provenance markers ([] when clean).
    """
    shares = info.get('sharesOutstanding')
    if not shares or info.get('marketCap'):
        return []
    shares = float(shares)

    def _agrees(candidate):
        return (candidate and candidate > 0
                and abs(candidate - shares) <= SHARES_CORROBORATION_TOL * shares)

    price = info.get('currentPrice') or info.get('regularMarketPrice')
    fast_shares = fast_mcap = 0.0
    try:
        fast = stock.fast_info
        fast_shares = float(getattr(fast, 'shares', None) or 0)
        fast_mcap = float(getattr(fast, 'market_cap', None) or 0)
        if not price:
            price = float(getattr(fast, 'last_price', None) or 0)
    except Exception:
        pass
    if _agrees(fast_shares):
        return []
    if fast_mcap and price and _agrees(fast_mcap / float(price)):
        return []
    try:
        series_last = _recent_series_shares(stock) or 0.0
    except Exception:
        series_last = 0.0
    if _agrees(series_last):
        return []
    info['sharesOutstanding'] = None
    return ['sharesOutstanding_uncorroborated']


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
    except Exception as e:
        logger.debug(f"yfinance: fast_info unavailable for {getattr(stock, 'ticker', info.get('symbol'))}: {e}")
        fast = None

    def _fast(attr):
        if fast is None:
            return None
        try:
            val = getattr(fast, attr, None)
            return float(val) if val else None
        except Exception as e:
            logger.debug(f"yfinance: fast_info.{attr} read failed for "
                         f"{getattr(stock, 'ticker', info.get('symbol'))}: {e}")
            return None

    if not info.get('sharesOutstanding'):
        shares = _fast('shares')
        if not shares:
            # Last resort: the shares-outstanding time series. Its final row is
            # the same figure fast_info reports, but it survives cases where
            # fast_info itself comes back bare. A stale series is ignored.
            try:
                shares = _recent_series_shares(stock)
            except Exception as e:
                logger.debug(f"yfinance: get_shares_full failed for "
                             f"{getattr(stock, 'ticker', info.get('symbol'))}: {e}")
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
# Sized above analyze_stock's Phase-2 prefetch threads (default 4) so a few
# orphaned timed-out calls cannot starve them into spurious timeouts.
_TIMEOUT_EXECUTOR = ThreadPoolExecutor(max_workers=8)


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
        ) from None


def probe_yahoo(symbol='SPY', timeout=20):
    """One quoteSummary request, classified: 'ok', 'rate_limited' or 'empty'.

    The startup gate's question is the one Phase 1 asks 9,000 times — can
    this host get an `.info` with an identity in it right now — so the
    probe is exactly that call, crumb included. Attempt 2 on 2026-09-30
    could not (its first log line was the crumb 429) and went on to a
    seven-hour run with a fabricated risk-free rate and 29% of the
    universe. An empty answer clears a poisoned crumb so the next probe
    re-mints one. Anything that is not a rate limit propagates.
    """
    def _fetch():
        info = yf.Ticker(symbol).info or {}
        return bool(info.get('symbol') or info.get('shortName') or info.get('longName'))

    try:
        ok = _run_with_timeout(_fetch, timeout) if timeout else _fetch()
    except Exception as e:
        if _is_rate_limited(e):
            return 'rate_limited'
        raise
    if ok:
        return 'ok'
    reset_crumb()
    return 'empty'


def _is_not_found(exc):
    """True for a definitive "symbol does not exist" answer from Yahoo.

    Retrying a 404 cannot succeed and costs ~3s of sleeps per dead symbol,
    which across a ~9k-ticker universe screen adds up to hours.
    """
    msg = str(exc)
    return ('404' in msg or 'Not Found' in msg
            or 'Quote not found' in msg or 'No fundamentals data found' in msg)


class YFinanceClient:
    def __init__(self, request_delay=1.0, snapshot_cache=None,
                 fetch_timeout=20, prices_dir="output/prices", run_date=None,
                 delay_max=None, penalty=1.5, relax_step=0.98,
                 auth_pause=10.0, auth_pause_max=120.0, auth_pause_budget=1800.0,
                 rate_limit_pause=20.0, rate_limit_pause_max=900.0,
                 rate_limit_budget=3600.0, rate_limit_probe_interval=600.0):
        self._financials_cache = {}
        self._history_cache = {}
        self._throttle = Throttle(request_delay)
        # Adaptive back-off bounds. Yahoo publishes no rate limit, so the
        # interval is a guess; these let a wrong guess correct itself in-run
        # instead of spending the night in a retry storm.
        self._delay_max = delay_max if delay_max is not None else max(request_delay * 7.5, 3.0)
        self._penalty = penalty
        self._relax_step = relax_step
        # Shared pause after a 401. yfinance's recovery re-mints the crumb
        # on every 401, so while getcrumb is rate-limited each throttled
        # ticker is one more getcrumb call keeping it rate-limited. Stopping
        # every thread for a while is what breaks that loop; the pause
        # doubles per consecutive failure (10s -> 120s) and resets on the
        # first healthy fetch. The budget caps the run's total pausing at
        # 30 min: past it an all-night auth outage is not retried at all, so
        # it costs no more time than it did before this check existed.
        self._auth_lock = threading.Lock()
        self._auth_pause_budget = auth_pause_budget
        self._auth_pause_base = auth_pause
        self._auth_pause_max = auth_pause_max
        self._auth_pause_next = auth_pause
        self._auth_pause_until = 0.0
        # Circuit breaker for a hard rate limit (HTTP 429). Same shared pause
        # as the 401 path — one more request from any thread is one more
        # reason for Yahoo to keep the IP limited — but on a longer scale:
        # 20s doubling to 15 min, since a 429 says "try after a while" and a
        # 1-2s retry is just another hit. Consecutive 429s escalate; a
        # successful `.info` fetch (not a history call: the chart endpoint
        # can answer while quoteSummary is blocked, which is what kept the
        # 401 back-off from ever doubling) resets the escalation AND the
        # budget, so the budget (an hour) measures consecutive pausing. The
        # 2026-09-30 re-run is why: refilled only on close, 36 sporadic 429s
        # on a Yahoo answering 98% of requests spent it 60-120s at a time and
        # opened the breaker on a healthy source. When it is spent the
        # breaker OPENS: every fetch raises YahooRateLimitError without a
        # request, so Phase 1 finishes on SEC data in minutes instead of
        # burning 3.7s of doomed retries per ticker for the rest of the
        # night. One probe — any fetch, since Phase 2 only fetches dividends
        # and could never close a breaker that only `.info` might probe — per
        # `rate_limit_probe_interval` is let through; the first success
        # closes the breaker and restores the budget.
        self._rl_pause_base = rate_limit_pause
        self._rl_pause_max = rate_limit_pause_max
        self._rl_pause_next = rate_limit_pause
        self._rl_budget_initial = rate_limit_budget
        self._rl_budget = rate_limit_budget
        self._rl_probe_interval = rate_limit_probe_interval
        self._rl_next_probe = 0.0
        self.rate_limited_out = False   # the breaker is open
        # Per-run call accounting (see stats()). `empty` counts Yahoo's soft
        # throttle, which _is_not_found deliberately does NOT match, so a
        # throttled ticker costs 3 throttle ticks + 3s of backoff before the
        # caller's retry queue even sees it. That amplification is the thing
        # to watch before raising concurrency or cutting the delay.
        # `rate_limited` counts 429 attempts (also in empty_attempts, since a
        # 429 is a throttle); `breaker_*` is the circuit breaker above.
        self.stats = {'calls': 0, 'seconds': 0.0, 'retries': 0, 'timeouts': 0,
                      'not_found': 0, 'empty_attempts': 0, 'errors': 0,
                      'auth_failures': 0, 'rate_limited': 0,
                      'rate_limit_pauses': 0, 'breaker_short_circuits': 0,
                      'breaker_probes': 0, 'breaker_opened': 0}
        # Bare yf.Ticker() calls below use yfinance's own session; honour a
        # YF_IMPERSONATE override for them too (no-op on the default profile).
        install_default_session()
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

    def evict_ticker(self, ticker):
        """Free everything cached for one ticker: its financials and every
        price history and dividend series.  Phase 1 of analyze_stock calls
        this after every ticker's screen: the end-of-phase evict_financials()
        / clear_history_cache() sweep already drops all of it, relying on
        screen_cache holding its own references to what Phase 2 needs, so
        per-ticker eviction reaches that end state without holding ~9k
        tickers' worth (a qualifying ticker's raw yfinance dict is ~4 MB)
        across the sweep."""
        self._financials_cache.pop(ticker, None)
        # list() first: the comprehension runs Python bytecode per item, so
        # iterating the live dict raises "dictionary changed size during
        # iteration" as soon as another thread inserts a history key. list(d)
        # is a single C-level call and is atomic under the GIL. Harmless while
        # Phase 1 was sequential; fatal once it prefetches on a pool.
        for key in [k for k in list(self._history_cache) if k[0] == ticker]:
            self._history_cache.pop(key, None)

    def _wait_auth_pause(self):
        with self._auth_lock:
            wait = self._auth_pause_until - time.monotonic()
        if wait > 0:
            time.sleep(wait)

    def _note_auth_failure(self):
        """Record a 401 and start (or join) the shared pause. Returns False
        once the pause budget is spent: the caller then stops retrying."""
        with self._auth_lock:
            self.stats['auth_failures'] += 1
            now = time.monotonic()
            # Threads failing inside the same pause share it rather than
            # each doubling it.
            if now < self._auth_pause_until:
                return True
            if self._auth_pause_budget <= 0:
                return False
            pause = min(self._auth_pause_next, self._auth_pause_budget)
            self._auth_pause_budget -= pause
            self._auth_pause_until = now + pause
            self._auth_pause_next = min(self._auth_pause_next * 2, self._auth_pause_max)
            if self._auth_pause_budget <= 0:
                logger.warning("yfinance: HTTP 401 (bad crumb) — pausing Yahoo requests "
                               "%.0fs; pause budget now spent, further 401s are not retried",
                               pause)
            else:
                logger.warning("yfinance: HTTP 401 (bad crumb) — pausing Yahoo requests "
                               "%.0fs", pause)
            return True

    def _note_rate_limit(self):
        """Record a 429 and start (or join) the shared pause. Returns False
        once the budget is spent — the breaker is then open and the caller
        stops retrying."""
        with self._auth_lock:
            self.stats['rate_limited'] += 1
            now = time.monotonic()
            if self.rate_limited_out:
                return False
            if now < self._auth_pause_until:
                return True
            if self._rl_budget <= 0:
                self._open_breaker(now)
                return False
            pause = min(self._rl_pause_next, self._rl_budget)
            self._rl_budget -= pause
            self._auth_pause_until = now + pause
            self._rl_pause_next = min(self._rl_pause_next * 2, self._rl_pause_max)
            self.stats['rate_limit_pauses'] += 1
            logger.warning("yfinance: HTTP 429 (rate limited) — pausing Yahoo requests "
                           "%.0fs (%.0fs of pause budget left)", pause, self._rl_budget)
            return True

    def _open_breaker(self, now):
        # Caller holds _auth_lock.
        self.rate_limited_out = True
        self.stats['breaker_opened'] += 1
        self._rl_next_probe = now + self._rl_probe_interval
        logger.warning("yfinance: rate-limit pause budget spent — breaker OPEN: Yahoo "
                       "fetches now fail without a request; one probe every %.0fs",
                       self._rl_probe_interval)

    def _close_breaker(self):
        with self._auth_lock:
            if not self.rate_limited_out:
                return
            self.rate_limited_out = False
            self._rl_budget = self._rl_budget_initial
            self._rl_pause_next = self._rl_pause_base
        logger.warning("yfinance: a probe succeeded — breaker CLOSED, Yahoo fetches resume")

    def _breaker_admits(self):
        """While the breaker is open, one fetch per probe interval may go out
        — whichever call comes first, so a phase that only fetches dividends
        can still close it. Returns True when this call may proceed."""
        with self._auth_lock:
            if not self.rate_limited_out:
                return True
            now = time.monotonic()
            if now >= self._rl_next_probe:
                self._rl_next_probe = now + self._rl_probe_interval
                self.stats['breaker_probes'] += 1
                return True
            self.stats['breaker_short_circuits'] += 1
            return False

    def wait_for_probe(self, sleep=time.sleep):
        """Block until the open breaker will admit a probe (at most one probe
        interval). Returns the seconds waited, 0 when the breaker is closed.

        Phase 1 calls it before its retry pass: the pass re-fetches every
        ticker Yahoo failed, and running it the moment the breaker opens — as
        the 2026-09-30 re-run did, 1,130 re-queued and 0 recovered in under
        a minute — spends the second chance on short-circuits."""
        with self._auth_lock:
            if not self.rate_limited_out:
                return 0.0
            wait = max(0.0, self._rl_next_probe - time.monotonic())
        if wait > 0:
            sleep(wait)
        return wait

    def _retry(self, func, max_retries=2, resets_backoff=False):
        """Run *func* with retries for transient failures.

        Timeouts are NOT retried — if a call hits the wall-clock limit, we
        accept the failure and propagate immediately.  Retrying a timeout
        only piles up orphaned threads and leaks sockets into CLOSE_WAIT,
        which poisons yfinance's internal connection pool for subsequent
        tickers.  Other exceptions (HTTP errors, parse errors) still retry.

        *resets_backoff*: a success resets the 401/429 escalation and may
        close the breaker. Only the `.info` fetch passes it: a history or
        dividends call can succeed while quoteSummary is still blocked, and
        letting those reset the back-off is why it never doubled on
        2026-09-30 (28 pauses, every one 10s).
        """
        t0 = time.perf_counter()
        self.stats['calls'] += 1
        try:
            if not self._breaker_admits():
                self.stats['errors'] += 1
                raise YahooRateLimitError(
                    'yfinance rate-limit breaker open — not requested')
            for attempt in range(max_retries + 1):
                if attempt:
                    self.stats['retries'] += 1
                try:
                    self._wait_auth_pause()
                    self._throttle()
                    if self._fetch_timeout is not None:
                        _out = _run_with_timeout(func, self._fetch_timeout)
                    else:
                        _out = func()
                    # Healthy response: walk a penalised interval back down.
                    # Never below the configured base (relax() floors there).
                    self._throttle.relax(self._relax_step)
                    # Any success proves Yahoo is answering again: close an
                    # open breaker. Only a healthy .info resets the 401/429
                    # escalation and the pause budget, which therefore
                    # measure consecutive pushback, not pushback per night.
                    if self.rate_limited_out:
                        self._close_breaker()
                    if resets_backoff:
                        if self._auth_pause_next != self._auth_pause_base \
                                or self._rl_pause_next != self._rl_pause_base \
                                or self._rl_budget != self._rl_budget_initial:
                            with self._auth_lock:
                                self._auth_pause_next = self._auth_pause_base
                                self._rl_pause_next = self._rl_pause_base
                                self._rl_budget = self._rl_budget_initial
                    return _out
                except TimeoutError:
                    # Don't retry — Yahoo is unresponsive for this ticker.
                    self.stats['timeouts'] += 1
                    raise
                except Exception as exc:
                    e = exc
                    if not isinstance(exc, EmptyYahooResponseError) and _is_rate_limited(exc):
                        # yfinance's own YFRateLimitError: the explicit form
                        # of the throttle, so it takes the throttle's path.
                        e = YahooRateLimitError(str(exc))
                        e.__cause__ = exc
                    if isinstance(e, EmptyYahooResponseError):
                        # Per ATTEMPT, not per call: a soft-throttled ticker
                        # raises on all three, and that 3x is the cost worth
                        # seeing.
                        self.stats['empty_attempts'] += 1
                        # Yahoo pushed back: widen the interval for everyone
                        # sharing this client (the pool's workers included)
                        # before the next attempt goes out.
                        self._throttle.penalize(self._penalty, cap=self._delay_max)
                    fatal = False
                    if isinstance(e, YahooRateLimitError) and not self._note_rate_limit():
                        self.stats['errors'] += 1
                        fatal = True
                    elif isinstance(e, YahooAuthError) and not self._note_auth_failure():
                        self.stats['errors'] += 1
                        fatal = True
                    elif attempt == max_retries or _is_not_found(e):
                        if _is_not_found(e):
                            self.stats['not_found'] += 1
                        else:
                            self.stats['errors'] += 1
                        fatal = True
                    if fatal:
                        if e is exc:
                            raise
                        raise e from exc
                    time.sleep(1.0 * (attempt + 1))
        finally:
            self.stats['seconds'] += time.perf_counter() - t0

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
        # NOTE: no per-call session — yfinance's own curl_cffi session is
        # used (re-pointed by install_default_session() when YF_IMPERSONATE
        # is set).  Connection pool hygiene is handled by the 20s timeout +
        # no-retry-on-timeout policy instead.

        def _fetch():
            # A fresh Ticker per attempt: yfinance marks `.info` fetched
            # before requesting it and caches the (empty) result of a failed
            # request, so a retry on the same object never asked Yahoo again.
            stock = yf.Ticker(ticker)
            # Runs on _run_with_timeout's worker thread, which is also where
            # yfinance logs, so the per-thread 401 count is this fetch's own.
            _auth0 = auth_errors_this_thread()
            data = {
                'balance_sheet': stock.balance_sheet,
                'income_statement': stock.financials,
                'cash_flow': stock.cashflow,
                'info': stock.info,
            }
            _auth_failed = auth_errors_this_thread() > _auth0
            if _auth_failed:
                reset_crumb()
            # Detect Yahoo soft-throttle: HTTP 200 with an info dict missing
            # all the standard identifying fields. A real response always
            # carries at least one of symbol/shortName/longName in info, even
            # for OTC / foreign-listed tickers.
            #
            # The info check stands on its own: Yahoo can throttle the
            # quoteSummary endpoint behind .info while the statement
            # (timeseries) endpoints keep answering. On 2026-09-25 that
            # returned complete statements with an empty .info for 1,548
            # tickers (a contiguous BIPH..MYRG window of the run); requiring
            # empty statements too let every one through as a valid response
            # with no price, sector, industry or name, and 751 of them
            # changed rating on the missing inputs.
            bs = data['balance_sheet']
            inc = data['income_statement']
            cf = data['cash_flow']
            info = data['info'] or {}
            bs_empty = bs is None or (hasattr(bs, 'empty') and bs.empty)
            inc_empty = inc is None or (hasattr(inc, 'empty') and inc.empty)
            cf_empty = cf is None or (hasattr(cf, 'empty') and cf.empty)
            info_empty = not (info.get('symbol') or info.get('shortName')
                              or info.get('longName'))
            if info_empty:
                _all_empty = bs_empty and inc_empty and cf_empty
                what = 'empty payload' if _all_empty else 'statements but an empty .info'
                if _auth_failed:
                    # A poisoned crumb empties .info while the timeseries
                    # endpoints keep answering, so this check fires before the
                    # 401 handler at the end of the fetch. Carry the
                    # statements on the exception (as that handler does) or a
                    # spent retry budget would drop data the fetch did get.
                    data['_info_auth_failed'] = True
                    raise YahooAuthError(
                        f"yfinance returned {what} for {ticker} (HTTP 401, bad crumb)",
                        data=None if _all_empty else data)
                raise EmptyYahooResponseError(
                    f"yfinance returned {what} for {ticker} (likely throttled)")
            # Cross-contamination guard: preferred / secondary OTC lines carry
            # the parent's common share count with no marketCap from any Yahoo
            # source. Must run BEFORE the backfill, which would otherwise
            # manufacture a phantom cap from the poisoned count (the FNM*/FMC*
            # $24B-$90B caps of 2026-08-12 came from exactly that derivation).
            _shares_nulled = _null_uncorroborated_shares(stock, info)
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
            _sanitized = _shares_nulled + _sanitize_implausible_mcap(info)
            if _sanitized:
                data['info'] = info
                data['_info_sanitized'] = _sanitized
            # Dual-class / ADR share counts: the reported count must be the
            # one the price and market cap are quoted on, or every per-share
            # fair value is off by the class ratio. Runs after the repair so
            # a rederived cap (price x shares) is a no-op here.
            _reconciled = _reconcile_shares_with_mcap(info)
            if _reconciled:
                data['info'] = info
                data['_info_shares_reconciled'] = _reconciled
            # Growth estimates and earnings history (may fail for some tickers)
            try:
                data['growth_estimates'] = stock.growth_estimates
            except Exception as e:
                logger.debug(f"yfinance: growth_estimates fetch failed for {ticker}: {e}")
                data['growth_estimates'] = None
            try:
                data['earnings_history'] = stock.earnings_history
            except Exception as e:
                logger.debug(f"yfinance: earnings_history fetch failed for {ticker}: {e}")
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
            if _auth_failed:
                data['_info_auth_failed'] = True
                raise YahooAuthError(
                    f"yfinance got HTTP 401 for {ticker} (bad crumb)", data=data)
            return data

        try:
            financials = self._retry(_fetch, resets_backoff=True)
        except YahooAuthError as e:
            if e.data is None:
                raise
            # Retries (or the pause budget) ran out: keep the statements the
            # last attempt did get, as this client did before the 401 check.
            logger.warning("yfinance: %s — keeping statements without quote data", e)
            financials = e.data
        self._financials_cache[ticker] = financials

        # Auto-save to disk cache if configured
        if self._snapshot_cache is not None:
            try:
                self._snapshot_cache.save(ticker, financials,
                                          as_of=self.run_date or date.today())
            except Exception as e:
                logger.debug(f"yfinance: snapshot cache write failed for {ticker}: {e}")
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
        except Exception as e:
            logger.warning(f"yfinance: dividends fetch failed for {ticker}: {e}")
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
        except Exception as e:
            logger.warning(f"yfinance: history fetch failed for {ticker}: {e}")
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
        except Exception as e:
            logger.debug(f"yfinance: price parquet write failed for {ticker}: {e}")
            pass
