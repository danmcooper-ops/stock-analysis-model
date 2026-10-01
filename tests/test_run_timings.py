"""Guards for the run/phase instrumentation.

Phase 1 had no timer at all: the only evidence of a slow night was stdout
timestamps. These cover the counters the tuning work is measured against —
if they drift, every later "X minutes faster" claim is unfounded.
"""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.provenance import ProvenanceRecorder  # noqa: E402
from data.throttle import Throttle  # noqa: E402
from data.yfinance_client import (EmptyYahooResponseError, YahooRateLimitError,  # noqa: E402
                                  YFinanceClient)
from scripts.analyze_stock import _PhaseClock, _phase1_timings  # noqa: E402


# --- Throttle ---------------------------------------------------------------

def test_throttle_counts_calls_and_sleep():
    t = Throttle(0.02)
    for _ in range(3):
        t()
    s = t.stats()
    assert s['calls'] == 3
    # First call never sleeps (last=0 is far in the past); the other two do.
    assert s['slept'] >= 0.03
    assert s['delay'] == 0.02


def test_throttle_with_no_delay_sleeps_nothing():
    t = Throttle(0)
    for _ in range(5):
        t()
    assert t.stats()['calls'] == 5
    assert t.stats()['slept'] < 0.05


def test_throttle_counts_lock_contention_across_threads():
    """`waited` is what other threads cost this one — the number that says
    whether more workers would actually help."""
    import threading
    t = Throttle(0.05)
    barrier = threading.Barrier(4)

    def hit():
        barrier.wait()
        t()

    threads = [threading.Thread(target=hit) for _ in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert t.stats()['calls'] == 4
    # Serialized at 0.05s apart, so the last arrivals block on the lock.
    assert t.stats()['waited'] > 0


# --- YFinanceClient._retry --------------------------------------------------

def _client():
    return YFinanceClient(request_delay=0, fetch_timeout=None)


def test_retry_counts_a_successful_call():
    c = _client()
    assert c._retry(lambda: 'ok') == 'ok'
    assert c.stats['calls'] == 1
    assert c.stats['retries'] == 0
    assert c.stats['seconds'] >= 0


def test_retry_counts_not_found_without_retrying(monkeypatch):
    monkeypatch.setattr(time, 'sleep', lambda s: None)
    c = _client()

    def dead():
        raise RuntimeError('HTTP Error 404: Quote not found for symbol: ZZZZ')

    with pytest.raises(RuntimeError):
        c._retry(dead)
    assert c.stats['calls'] == 1
    assert c.stats['not_found'] == 1
    assert c.stats['retries'] == 0        # a 404 is never retried
    assert c.stats['errors'] == 0


def test_retry_counts_empty_attempts_per_attempt(monkeypatch):
    """Yahoo's soft throttle is not a 404, so it burns all three attempts.

    That 3x amplification is the cost the safety valve has to control before
    concurrency goes up, so it is counted per attempt, not per call.
    """
    monkeypatch.setattr(time, 'sleep', lambda s: None)
    c = _client()

    def throttled():
        raise EmptyYahooResponseError('empty payload')

    with pytest.raises(EmptyYahooResponseError):
        c._retry(throttled)
    assert c.stats['calls'] == 1
    assert c.stats['empty_attempts'] == 3
    assert c.stats['retries'] == 2


def test_rate_limit_is_a_throttle_signal(monkeypatch):
    """yfinance's YFRateLimitError (HTTP 429) must take the soft-throttle
    path: penalise the interval, count as an empty attempt, and reach the
    caller as an EmptyYahooResponseError so it is re-queued and falls back.

    On 2026-09-30 it was a plain Exception to all of that: 8,030 of 9,186
    fetches raised it, the interval never widened and nothing fell back.
    """
    monkeypatch.setattr(time, 'sleep', lambda s: None)
    c = YFinanceClient(request_delay=0.2, fetch_timeout=None, delay_max=3.0,
                       rate_limit_pause=0, rate_limit_budget=1e9)
    from yfinance.exceptions import YFRateLimitError

    def limited():
        raise YFRateLimitError()

    with pytest.raises(EmptyYahooResponseError) as ei:
        c._retry(limited)
    assert isinstance(ei.value, YahooRateLimitError)
    assert isinstance(ei.value.__cause__, YFRateLimitError)
    assert c.stats['rate_limited'] == 3
    assert c.stats['empty_attempts'] == 3            # a 429 is a throttle
    assert c._throttle.delay > 0.2                   # the interval widened


def test_rate_limit_text_is_recognised_when_wrapped(monkeypatch):
    monkeypatch.setattr(time, 'sleep', lambda s: None)
    c = YFinanceClient(request_delay=0, fetch_timeout=None,
                       rate_limit_pause=0, rate_limit_budget=1e9)

    def wrapped():
        raise RuntimeError('Too Many Requests. Rate limited. Try after a while.')

    with pytest.raises(YahooRateLimitError):
        c._retry(wrapped)
    assert c.stats['rate_limited'] == 3


def test_retry_counts_a_recovered_transient(monkeypatch):
    monkeypatch.setattr(time, 'sleep', lambda s: None)
    c = _client()
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError('reset by peer')
        return 'ok'

    assert c._retry(flaky) == 'ok'
    assert c.stats['calls'] == 1
    assert c.stats['retries'] == 2
    assert c.stats['errors'] == 0         # it succeeded in the end


# --- Phase clock ------------------------------------------------------------

def test_phase_clock_orders_and_totals():
    c = _PhaseClock()
    c.tick('a')
    c.tick('b')
    d = c.as_dict()
    assert [p['name'] for p in d['phases']] == ['a', 'b']
    assert d['total_seconds'] == pytest.approx(
        sum(p['seconds'] for p in d['phases']))
    assert 'TOTAL' in c.table()


def test_phase_clock_table_is_safe_when_empty():
    assert 'TOTAL' in _PhaseClock().table()


# --- Phase-1 breakdown ------------------------------------------------------

class _FakeThrottle:
    def stats(self):
        return {'calls': 2, 'slept': 1.5, 'waited': 0.1, 'delay': 1.0}


class _FakeYF:
    def __init__(self):
        self.stats = {'calls': 2, 'seconds': 3.0, 'retries': 0, 'timeouts': 0,
                      'not_found': 1, 'empty_attempts': 0, 'errors': 0}
        self._throttle = _FakeThrottle()


class _FakeSEC:
    def __init__(self):
        self.facts_stats = {'net_calls': 1, 'net_seconds': 0.5, 'mem_hits': 3,
                            'disk_hits': 2, 'no_cik': 0, 'failures': 0}
        self._throttle = _FakeThrottle()


def test_phase1_timings_collects_every_leg():
    t = _phase1_timings(120.0, {'yf_fetch': 60.0}, {'yf_fetch': 2},
                        _FakeYF(), _FakeSEC())
    assert t['elapsed'] == 120.0
    assert t['legs']['yf_fetch'] == 60.0
    assert t['yfinance']['not_found'] == 1
    assert t['yfinance_throttle']['slept'] == 1.5
    assert t['sec_facts']['disk_hits'] == 2
    assert t['sec_throttle']['calls'] == 2


def test_phase1_timings_tolerates_clients_without_counters():
    """Never let instrumentation be the thing that fails a 5-hour run."""
    class Bare:
        pass

    t = _phase1_timings(1.0, {}, {}, Bare(), Bare())
    assert t['yfinance'] == {}
    assert t['sec_facts'] == {}
    assert t['yfinance_throttle'] == {}


# --- Provenance -------------------------------------------------------------

def test_provenance_carries_timings_when_set():
    p = ProvenanceRecorder('2026-09-15')
    assert 'timings' not in p.run_block()          # absent until recorded
    p.record_timings({'total_seconds': 42})
    assert p.run_block()['timings'] == {'total_seconds': 42}


def test_provenance_timings_never_raise():
    p = ProvenanceRecorder('2026-09-15')
    p.record_timings(None)
    assert p.run_block()['schema_version']


# --- Adaptive back-off ------------------------------------------------------
# The interval is a guess at Yahoo's undocumented limit, so a wrong guess has
# to correct itself in-run rather than spend the night in a retry storm.

def test_penalize_widens_and_clamps_to_cap():
    t = Throttle(0.4)
    assert t.penalize(1.5, cap=3.0) == pytest.approx(0.6)
    assert t.penalize(1.5, cap=3.0) == pytest.approx(0.9)
    for _ in range(50):
        t.penalize(1.5, cap=3.0)
    assert t.delay == pytest.approx(3.0)       # never runs away
    assert t.stats()['penalties'] > 0


def test_penalize_never_shrinks_the_interval():
    t = Throttle(0.4)
    assert t.penalize(0.5) == pytest.approx(0.4)   # factor < 1 is ignored
    assert t.penalties == 0


def test_relax_walks_back_only_to_the_configured_base():
    t = Throttle(0.4)
    t.penalize(4.0, cap=3.0)                   # 1.6
    for _ in range(500):
        t.relax(0.9)
    assert t.delay == pytest.approx(0.4)       # floors at base, never below
    assert t.base_delay == pytest.approx(0.4)


def test_relax_is_a_noop_at_base():
    t = Throttle(0.4)
    assert t.relax(0.5) == pytest.approx(0.4)


def test_retry_penalizes_on_soft_throttle_and_relaxes_on_success(monkeypatch):
    monkeypatch.setattr(time, 'sleep', lambda s: None)
    c = YFinanceClient(request_delay=0.4, fetch_timeout=None,
                       delay_max=3.0, penalty=1.5, relax_step=0.5)

    def throttled():
        raise EmptyYahooResponseError('empty payload')

    with pytest.raises(EmptyYahooResponseError):
        c._retry(throttled)
    widened = c._throttle.delay
    assert widened > 0.4, "a soft throttle must widen the interval"
    assert widened <= 3.0

    for _ in range(50):                        # healthy traffic resumes
        c._retry(lambda: 'ok')
    assert c._throttle.delay == pytest.approx(0.4)


def test_retry_does_not_penalize_on_a_404(monkeypatch):
    """A dead symbol says nothing about our request rate."""
    monkeypatch.setattr(time, 'sleep', lambda s: None)
    c = YFinanceClient(request_delay=0.4, fetch_timeout=None)

    def dead():
        raise RuntimeError('HTTP Error 404: Quote not found for symbol: ZZZZ')

    with pytest.raises(RuntimeError):
        c._retry(dead)
    assert c._throttle.delay == pytest.approx(0.4)
    assert c._throttle.penalties == 0


def test_valve_break_even_stays_above_observed_empty_rates():
    """penalize and relax are a control loop; they cancel at
    penalty * relax**(n-1) == 1, i.e. one empty per n calls.

    At 1.5 / 0.98 that was one per 21.1 calls (4.75%) while the 2026-09-22..25
    runs sat at 4.66-5.30%, so the delay ratcheted to the cap and stayed
    pinned. The configured pair must keep a real margin over that.
    """
    import math
    from scripts.config import YF_THROTTLE_PENALTY, YF_THROTTLE_RELAX
    n = 1 + math.log(1 / YF_THROTTLE_PENALTY) / math.log(YF_THROTTLE_RELAX)
    break_even_rate = 1 / n
    assert break_even_rate > 0.08, (
        f"valve breaks even at {break_even_rate:.1%} empty; observed runs hit "
        "5.3%, so this leaves no margin and the delay will ratchet to the cap")


def test_a_realistic_empty_rate_does_not_ratchet_to_the_cap():
    """Simulate the observed 5.2% empty rate against the configured pair."""
    from scripts.config import (YF_REQUEST_DELAY, YF_REQUEST_DELAY_MAX,
                                YF_THROTTLE_PENALTY, YF_THROTTLE_RELAX)
    t = Throttle(YF_REQUEST_DELAY)
    for i in range(8000):
        if i % 19 == 0:                    # ~5.3% empty
            t.penalize(YF_THROTTLE_PENALTY, cap=YF_REQUEST_DELAY_MAX)
        else:
            t.relax(YF_THROTTLE_RELAX)
    assert t.delay < YF_REQUEST_DELAY_MAX * 0.5, (
        f"delay settled at {t.delay:.2f}s, heading for the "
        f"{YF_REQUEST_DELAY_MAX}s cap as it did on 2026-09-23/25")


def test_a_genuine_rate_problem_still_reaches_the_cap():
    """Softening the loop must not disarm it."""
    from scripts.config import (YF_REQUEST_DELAY, YF_REQUEST_DELAY_MAX,
                                YF_THROTTLE_PENALTY)
    t = Throttle(YF_REQUEST_DELAY)
    for _ in range(200):
        t.penalize(YF_THROTTLE_PENALTY, cap=YF_REQUEST_DELAY_MAX)
    assert t.delay == pytest.approx(YF_REQUEST_DELAY_MAX)
