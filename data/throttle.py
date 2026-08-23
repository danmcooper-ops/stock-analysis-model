"""Shared request throttling for the HTTP data clients."""

import time


class Throttle:
    """Minimum-interval rate limiter shared by the HTTP data clients.

    Calling the instance sleeps just long enough to keep at least
    `delay` seconds between successive calls, then stamps the time.
    """

    def __init__(self, delay):
        self.delay = delay
        self._last = 0.0

    def __call__(self):
        elapsed = time.time() - self._last
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last = time.time()
