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
from data.yfinance_client import (EmptyYahooResponseError,  # noqa: E402
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
