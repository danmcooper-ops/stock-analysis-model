"""Shared request throttling for the HTTP data clients."""

import threading
import time

# SEC EDGAR fair-access limit is 10 requests/second per client across ALL
# endpoints, so every SEC client in a process should share one throttle.
SEC_MIN_INTERVAL = 0.12


class Throttle:
    """Minimum-interval rate limiter shared by the HTTP data clients.

    Calling the instance sleeps just long enough to keep at least
    `delay` seconds between successive calls, then stamps the time.

    Thread-safe: concurrent callers queue on the lock, so a throttle shared
    across worker threads still enforces one interval for all of them.

    Counts its own cost: ``calls``, ``slept`` (seconds actually spent
    sleeping) and ``waited`` (seconds spent blocked on the lock, i.e. what
    other threads cost this one). A phase that is throttle-bound shows
    ``slept`` close to its wall clock; that is the number that decides whether
    the lever is concurrency or the delay itself.
    """

    def __init__(self, delay):
        self.delay = delay
        self._last = 0.0
        self._lock = threading.Lock()
        self.calls = 0
        self.slept = 0.0
        self.waited = 0.0

    def __call__(self):
        t0 = time.time()
        with self._lock:
            acquired = time.time()
            elapsed = acquired - self._last
            if elapsed < self.delay:
                time.sleep(self.delay - elapsed)
            now = time.time()
            self._last = now
            # Counters are only mutated under the lock, so they stay coherent
            # without a second one.
            self.calls += 1
            self.waited += acquired - t0
            self.slept += now - acquired

    def stats(self):
        return {'calls': self.calls, 'slept': self.slept, 'waited': self.waited,
                'delay': self.delay}
