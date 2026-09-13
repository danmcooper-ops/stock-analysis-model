"""Persistent Phase-1 skip list for tickers that cannot pass the screen.

The US universe is ~9k tickers, of which ~6.5k are rejected every night
before Phase 2 — mostly micro-caps under the ``--mcap-min`` floor and dead
symbols that yfinance 404s and SEC has no coverage for. Each rejection still
paid a full fundamentals fetch, which made the screen take ~4.5 hours.

This cache remembers those rejections so the next runs skip them without a
fetch. Entries expire after a per-ticker staggered TTL, so every skipped
ticker is re-checked regularly (a company that grows past the floor is picked
up within a week or two) without the whole list expiring on the same night.

Only a ticker far enough below the floor is skipped: a micro-cap at 80% of
the floor could cross it on a normal move, so it keeps being fetched.
Carry-forward tickers are never skipped by the caller.
"""

import json
import logging
import os
import zlib
from datetime import date

logger = logging.getLogger(__name__)

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), 'cache', 'screen_skip.json')

# A cached mcap must be below this fraction of today's floor to skip.
MCAP_SKIP_FRACTION = 0.5
MCAP_TTL_DAYS = 7          # base TTL; each ticker adds 0..MCAP_TTL_DAYS-1 of jitter
DEAD_TTL_DAYS = 14         # base TTL for symbols no source returned anything for


def _jitter(ticker, span):
    # crc32 rather than hash(): stable across processes (PYTHONHASHSEED).
    return zlib.crc32(ticker.encode('utf-8')) % span if span > 0 else 0


class ScreenSkipCache:
    def __init__(self, path=DEFAULT_PATH, today=None):
        self.path = path
        self.today = today or date.today()
        self._entries = {}
        self._dirty = False
        try:
            with open(path, encoding='utf-8') as f:
                self._entries = json.load(f)
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as e:
            logger.warning('screen skip cache unreadable at %s (%s); starting empty', path, e)

    # ------------------------------------------------------------------
    def _age_ok(self, entry, ticker, base_ttl):
        try:
            seen = date.fromisoformat(entry['date'])
        except (KeyError, TypeError, ValueError):
            return False
        ttl = base_ttl + _jitter(ticker, base_ttl)
        return 0 <= (self.today - seen).days < ttl

    def skip_reason(self, ticker, mcap_min):
        """A short reason string if *ticker* should be skipped, else None."""
        entry = self._entries.get(ticker)
        if not entry:
            return None
        kind = entry.get('kind')
        if kind == 'dead' and self._age_ok(entry, ticker, DEAD_TTL_DAYS):
            return f"no data from any source since {entry['date']}"
        if (kind == 'mcap' and mcap_min
                and self._age_ok(entry, ticker, MCAP_TTL_DAYS)):
            mcap = entry.get('mcap') or 0
            if mcap < mcap_min * MCAP_SKIP_FRACTION:
                return f"mcap ${mcap / 1e6:.0f}M on {entry['date']} (cached)"
        return None

    # ------------------------------------------------------------------
    def record_mcap(self, ticker, mcap):
        self._entries[ticker] = {'kind': 'mcap', 'mcap': float(mcap or 0),
                                 'date': self.today.isoformat()}
        self._dirty = True

    def record_dead(self, ticker):
        self._entries[ticker] = {'kind': 'dead', 'date': self.today.isoformat()}
        self._dirty = True

    def forget(self, ticker):
        """Drop a ticker that was fetched and passed (or at least had data)."""
        if self._entries.pop(ticker, None) is not None:
            self._dirty = True

    def save(self):
        if not self._dirty:
            return
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(self._entries, f, separators=(',', ':'), sort_keys=True)
            os.replace(tmp, self.path)
            self._dirty = False
        except OSError as e:
            logger.warning('screen skip cache save failed at %s: %s', self.path, e)

    def __len__(self):
        return len(self._entries)
