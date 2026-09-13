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
    """

    def __init__(self, delay):
        self.delay = delay
        self._last = 0.0
        self._lock = threading.Lock()

    def __call__(self):
        with self._lock:
            elapsed = time.time() - self._last
            if elapsed < self.delay:
                time.sleep(self.delay - elapsed)
            self._last = time.time()
