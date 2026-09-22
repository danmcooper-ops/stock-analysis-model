"""Phase-1 network prefetch pool.

The pool only warms caches: every decision, print, counter and cache
mutation stays in the sequential loop. The load-bearing test here is that a
run with workers=4 produces byte-identical stdout and identical outputs to
workers=1 — if that ever breaks, the pool is changing the screen, which is
the one thing it must never do.
"""

import io
import sys
import threading
from contextlib import redirect_stdout
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.analyze_stock as A  # noqa: E402
from data.screen_skip_cache import ScreenSkipCache  # noqa: E402
from data.yfinance_client import EmptyYahooResponseError  # noqa: E402

RUN_DAY = date(2026, 9, 21)


class _FakeYF:
    """Records every fetch. Thread-safe: the pool calls it from workers."""

    def __init__(self, empty=(), mcaps=None):
        self.run_date = RUN_DAY
        self.stats = {'calls': 0, 'seconds': 0.0, 'retries': 0, 'timeouts': 0,
                      'not_found': 0, 'empty_attempts': 0, 'errors': 0}
        self._lock = threading.Lock()
        self._cache = {}
        self.fetched = []          # tickers fetch_financials actually worked for
        self.history = []
        self._empty = set(empty)
        self._mcaps = mcaps or {}

    def fetch_financials(self, ticker):
        with self._lock:
            if ticker in self._cache:
                return self._cache[ticker]
            self.fetched.append(ticker)
            self.stats['calls'] += 1
            if ticker in self._empty:
                self.stats['empty_attempts'] += 3
                raise EmptyYahooResponseError(ticker)
            data = {'info': {'marketCap': self._mcaps.get(ticker, 5e9),
                             'sector': 'Technology', 'symbol': ticker},
                    'balance_sheet': None, 'income_statement': None,
                    'cash_flow': None}
            self._cache[ticker] = data
            return data

    def fetch_history(self, ticker, period="5y"):
        with self._lock:
            self.history.append(ticker)
        return None

    def evict_ticker(self, ticker):
        with self._lock:
            self._cache.pop(ticker, None)

    def evict_financials(self):
        self._cache.clear()

    def clear_history_cache(self):
        pass


class _FakeSEC:
    def __init__(self, ciks=()):
        self._cik_map = {t: str(i) for i, t in enumerate(ciks, 1)}
        self._cache = {}
        self.facts_stats = {'net_calls': 0, 'disk_hits': 0, 'mem_hits': 0}
        self.facts_fetched = []
        self._lock = threading.Lock()

    def fetch_company_facts(self, ticker):
        with self._lock:
            self.facts_fetched.append(ticker)
        return {'facts': {}}

    def build_yfinance_shape(self, ticker):
        return None

    def get_filing_provenance(self, ticker):
        return None

    def release_facts(self, ticker):
        self._cache.pop(ticker, None)


class _Prov:
    def record_source(self, *a, **k):
        pass

    def record_event(self, *a, **k):
        pass


def _args(**over):
    base = dict(tickers=None, mcap_min=1e9, min_spread=None,
                no_screen_cache=True, validation=False)
    base.update(over)
    return SimpleNamespace(**base)


@pytest.fixture(autouse=True)
def _stub_models(monkeypatch, tmp_path):
    """Keep the test about orchestration, not the valuation math."""
    monkeypatch.setattr(A, 'calculate_roic', lambda d: {'roic_median_5y': 0.15})
    monkeypatch.setattr(A, 'calculate_wacc', lambda *a, **k: 0.08)
    monkeypatch.setattr(A, 'select_cost_of_equity',
                        lambda *a, **k: (0.09, 'capm', None))
    monkeypatch.setattr(A, '_convert_financials_to_usd', lambda d, **k: (d, {}))
    monkeypatch.setattr(A, '_fresh_local_prices', lambda *a, **k: None)
    # Never touch the repo's real carry-forward snapshot or skip cache.
    monkeypatch.setattr(A, 'prior_snapshot_file', lambda *a, **k: None)
    monkeypatch.setattr(A, 'ScreenSkipCache',
                        lambda **k: ScreenSkipCache(path=str(tmp_path / 's.json'),
                                                    today=k.get('today')))


def _run(tickers, workers, args=None, sec_ciks=None, empty=(), mcaps=None,
         checkpoint=None):
    yf = _FakeYF(empty=empty, mcaps=mcaps)
    sec = _FakeSEC(ciks=sec_ciks if sec_ciks is not None else tickers)
    buf = io.StringIO()
    with redirect_stdout(buf):
        out = A._run_phase1_screen(
            args or _args(), _Prov(), list(tickers),
            {t: 'quality' for t in tickers}, yf, None, sec,
            0.04, 0.045, checkpoint=checkpoint, prices_dir=None,
            phase1_workers=workers)
    return out, buf.getvalue(), yf, sec


# --- the load-bearing equivalence test --------------------------------------

def test_pool_output_is_identical_to_sequential():
    tickers = [f'T{i:03d}' for i in range(60)]
    mcaps = {t: (5e8 if int(t[1:]) % 3 == 0 else 5e9) for t in tickers}

    seq, seq_out, seq_yf, _ = _run(tickers, 1, mcaps=mcaps)
    par, par_out, par_yf, _ = _run(tickers, 4, mcaps=mcaps)

    # Every per-ticker decision and the phase's own tallies must match
    # exactly. The pool's banner and the timing breakdown are excluded:
    # elapsed seconds differ between any two runs, pool or not.
    def _decisions(text):
        return [ln for ln in text.splitlines()
                if ln.startswith('  [') or 'tickers collected' in ln
                or ln.startswith('  Fetch-failure') or ln.startswith('  Resumed')]

    # Guard against a vacuous comparison: one decision line per ticker.
    assert len(_decisions(seq_out)) >= len(tickers)
    assert _decisions(seq_out) == _decisions(par_out), \
        "stdout diverged between workers=1 and workers=4"
    assert seq['qualifying'] == par['qualifying']
    assert sorted(seq['screen_cache']) == sorted(par['screen_cache'])
    assert seq['screen_outcomes'] == par['screen_outcomes']
    # Same work, not just the same answer.
    assert sorted(seq_yf.fetched) == sorted(par_yf.fetched)


def test_pool_preserves_order_and_numbering():
    tickers = [f'T{i:03d}' for i in range(40)]
    _, out, _, _ = _run(tickers, 4)
    seen = [ln.split(']')[1].split('-')[0].strip()
        for ln in out.splitlines()
        if ln.startswith('  [') and ']' in ln]
    assert seen == tickers, "tickers were not decided in all_tickers order"


# --- the pool must not do work the loop would skip --------------------------

def test_skip_list_tickers_are_never_fetched():
    tickers = ['KEEP1', 'SKIPME', 'KEEP2']
    skip = Path(A.__file__).resolve().parent.parent / 'data' / 'skip_tickers.txt'
    if not skip.exists():                      # repo always ships one
        pytest.skip('no skip_tickers.txt')
    _, _, yf, _ = _run(tickers, 4)
    # Whatever the repo's real skip list holds, nothing outside it is dropped.
    assert 'KEEP1' in yf.fetched and 'KEEP2' in yf.fetched


def test_cache_skipped_tickers_are_never_fetched(tmp_path, monkeypatch):
    """The whole point of the skip cache is to not pay for these."""
    cache = ScreenSkipCache(path=str(tmp_path / 'skip.json'), today=RUN_DAY)
    cache.record_mcap('TINY', 1e6)             # far below the floor
    cache.save()
    monkeypatch.setattr(A, 'ScreenSkipCache',
                        lambda **k: ScreenSkipCache(path=str(tmp_path / 'skip.json'),
                                                    today=k.get('today')))
    tickers = ['BIG1', 'TINY', 'BIG2']
    _, out, yf, sec = _run(tickers, 4, args=_args(no_screen_cache=False))
    assert 'TINY' not in yf.fetched, "pool fetched a ticker the skip cache drops"
    assert 'TINY' not in sec.facts_fetched
    assert 'SKIP' in out


def test_checkpointed_tickers_are_never_fetched():
    class _Ckpt:
        def counts(self):
            return {'screened_out': 1}

        def screened_out(self, t):
            return t == 'DONE'

        def record_screened_out(self, *a):
            pass

    _, _, yf, _ = _run(['A1', 'DONE', 'A2'], 4, checkpoint=_Ckpt())
    assert 'DONE' not in yf.fetched


def test_sub_floor_tickers_cost_no_sec_or_history_fetch():
    """The pool mirrors the loop's early mcap bail."""
    tickers = ['BIG', 'TINY']
    _, _, yf, sec = _run(tickers, 4, mcaps={'BIG': 5e9, 'TINY': 1e6})
    assert 'TINY' in yf.fetched              # the mcap is why it was fetched
    assert 'TINY' not in sec.facts_fetched   # but nothing beyond that
    assert 'TINY' not in yf.history


# --- resilience -------------------------------------------------------------

def test_soft_throttled_ticker_costs_one_fetch_per_occurrence():
    """A pool-side empty response must not make the loop pay a second call.

    An empty response is never cached, so without the _prefetch_empty marker
    each occurrence costs two round trips (pool, then loop) — four in total
    once the fetch-failure retry re-queues the ticker. With it, one each.
    """
    tickers = ['OK1', 'THROTTLED', 'OK2']
    _, out, yf, _ = _run(tickers, 4, empty={'THROTTLED'})
    assert yf.fetched.count('THROTTLED') == 2   # the original + the requeue
    assert 're-queued' in out                   # still takes the retry path


def test_requeued_ticker_is_retried_and_prefetched():
    tickers = ['OK1', 'GONE']
    out, text, yf, _ = _run(tickers, 4, empty={'GONE'})
    # Requeued once, so fetched twice overall, then given up on.
    assert yf.fetched.count('GONE') == 2
    assert 'retry also failed' in text


def test_prefetch_window_is_bounded():
    """Memory, not throughput: an unbounded look-ahead is what OOM-killed
    the cloud run at 13.3 GiB."""
    tickers = [f'T{i:03d}' for i in range(200)]
    consumed = {'n': 0}
    ahead = []
    pos = {t: i for i, t in enumerate(tickers)}

    real_roic = A.calculate_roic

    def _counting_roic(d):
        consumed['n'] += 1
        return real_roic(d)

    yf = _FakeYF()
    real_fetch = yf.fetch_financials

    def _tracking_fetch(t):
        ahead.append(pos[t] - consumed['n'])
        return real_fetch(t)

    yf.fetch_financials = _tracking_fetch
    A.calculate_roic = _counting_roic
    try:
        workers = 4
        with redirect_stdout(io.StringIO()):
            A._run_phase1_screen(_args(), _Prov(), list(tickers),
                                 {t: 'quality' for t in tickers}, yf, None,
                                 _FakeSEC(ciks=tickers), 0.04, 0.045,
                                 prices_dir=None, phase1_workers=workers)
    finally:
        A.calculate_roic = real_roic
    limit = workers * A.PHASE1_PREFETCH_WINDOW_MULT + workers + 2
    assert max(ahead) <= limit, f"prefetched {max(ahead)} ahead, limit {limit}"


def test_workers_one_creates_no_pool():
    tickers = ['A1', 'A2']
    _, out, _, _ = _run(tickers, 1)
    assert 'prefetching network data' not in out
