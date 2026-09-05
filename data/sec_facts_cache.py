# data/sec_facts_cache.py
"""Persistent disk cache for SEC XBRL companyfacts blobs.

``SECXBRLClient`` held companyfacts only in a per-process dict, so every
nightly run re-downloaded the whole corpus: ~3.9 MB per ticker uncompressed,
which is ~6 GB over the ~1,500 enriched tickers and ~27 GB over a full US
listing sweep.  This module keeps those blobs on disk (gzipped, ~14x smaller)
so a run fetches only what actually changed.

Freshness is driven by what filers actually did, not by a guessed TTL.  SEC
publishes no ``ETag`` or ``Last-Modified`` on companyfacts, so there is no
conditional GET to lean on — but it does publish a daily filing index, and
``SECXBRLClient.refresh_stale_facts()`` reads it and calls :meth:`invalidate`
with the CIKs that filed.  An entry therefore lives until its filer files
something, rather than until an arbitrary timer expires.

``max_age_days`` is only a backstop for when that sweep cannot run (no
network at sweep time, a gap too long to walk): entries older than it are
treated as missing so the cache can never serve indefinitely-stale
fundamentals.

This module performs no network I/O; the client owns every request.
"""

import gzip
import json
import logging
import os
import time
from datetime import date, datetime

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache', 'sec_facts')

# Backstop only — see the module docstring. Filing-driven invalidation is the
# real freshness mechanism, so this is deliberately generous.
DEFAULT_MAX_AGE_DAYS = 30

_STATE_FILE = '_state.json'


def _norm_cik(cik):
    """CIKs appear zero-padded to 10 in the ticker map and bare in the daily
    index; normalise both to the padded form so they collide correctly."""
    s = str(cik).strip()
    if not s:
        return None
    s = s.lstrip('0') or '0'
    if not s.isdigit():
        return None
    return s.zfill(10)


class SECFactsCache:
    """Gzipped companyfacts blobs on disk, keyed by zero-padded CIK."""

    def __init__(self, cache_dir=None, max_age_days=None):
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        if max_age_days is None:
            try:
                max_age_days = float(os.environ.get(
                    'SEC_FACTS_CACHE_MAX_AGE_DAYS', DEFAULT_MAX_AGE_DAYS))
            except ValueError:
                max_age_days = DEFAULT_MAX_AGE_DAYS
        # Non-positive disables the backstop entirely: entries then live until
        # their filer files. Only sane when the index sweep is known to be
        # running, since nothing else would ever expire a blob.
        self.max_age_days = max_age_days if max_age_days > 0 else None
        self.hits = self.misses = self.writes = self.invalidated = 0

    # -- paths ------------------------------------------------------------

    def path_for(self, cik):
        norm = _norm_cik(cik)
        return None if norm is None else os.path.join(
            self.cache_dir, f'{norm}.json.gz')

    def age_days(self, cik):
        """Age of the cached blob in days, or None when it isn't cached."""
        path = self.path_for(cik)
        if not path or not os.path.exists(path):
            return None
        return (time.time() - os.path.getmtime(path)) / 86400.0

    # -- read / write -----------------------------------------------------

    def get(self, cik):
        """The cached companyfacts blob, or None when absent/expired/corrupt."""
        path = self.path_for(cik)
        if not path or not os.path.exists(path):
            self.misses += 1
            return None
        age = (time.time() - os.path.getmtime(path)) / 86400.0
        if self.max_age_days is not None and age > self.max_age_days:
            logger.debug("sec facts cache: %s is %.1fd old (backstop %.1fd)",
                         cik, age, self.max_age_days)
            self.misses += 1
            return None
        try:
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            # A truncated or corrupt entry must read as a miss, never raise:
            # the caller refetches and overwrites it.
            logger.warning("sec facts cache: unreadable entry %s (%s); refetching",
                           path, e)
            self.misses += 1
            return None
        self.hits += 1
        return data

    def put(self, cik, data):
        """Persist a blob. Returns the path written, or None when it couldn't be."""
        path = self.path_for(cik)
        if not path or data is None:
            return None
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            # Write-then-rename so a crash mid-write can't leave a truncated
            # entry that every later run has to detect and discard.
            tmp = f'{path}.tmp.{os.getpid()}'
            with gzip.open(tmp, 'wt', encoding='utf-8') as f:
                json.dump(data, f)
            os.replace(tmp, path)
        except Exception as e:
            logger.warning("sec facts cache: write failed for %s: %s", cik, e)
            return None
        self.writes += 1
        return path

    def invalidate(self, ciks):
        """Drop the cached blobs for *ciks*. Returns how many were removed."""
        removed = 0
        for cik in ciks or ():
            path = self.path_for(cik)
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    removed += 1
                except OSError as e:
                    logger.warning("sec facts cache: could not evict %s: %s",
                                   path, e)
        self.invalidated += removed
        return removed

    def prune_expired(self):
        """Delete entries past the age backstop. Returns how many were removed.

        Such an entry can never be served (:meth:`get` treats it as a miss), so
        it is pure disk. Entries for tickers still in the universe are rewritten
        on their next fetch and so never reach this; what accumulates here are
        filers that dropped out of the universe entirely (delistings, a
        narrowed run) and would otherwise sit on disk forever.
        """
        if self.max_age_days is None:
            return 0
        cutoff = time.time() - self.max_age_days * 86400
        removed = 0
        try:
            names = os.listdir(self.cache_dir)
        except OSError:
            return 0
        for name in names:
            if not name.endswith('.json.gz'):
                continue
            path = os.path.join(self.cache_dir, name)
            try:
                if os.path.getmtime(path) < cutoff:
                    os.remove(path)
                    removed += 1
            except OSError as e:
                logger.debug("sec facts cache: could not prune %s: %s", path, e)
        if removed:
            logger.info("sec facts cache: pruned %d entr(ies) past the "
                        "%.0f-day backstop", removed, self.max_age_days)
        return removed

    # -- sweep bookkeeping ------------------------------------------------

    def _state_path(self):
        return os.path.join(self.cache_dir, _STATE_FILE)

    def _read_state(self):
        try:
            with open(self._state_path(), encoding='utf-8') as f:
                state = json.load(f)
            return state if isinstance(state, dict) else {}
        except Exception:
            return {}

    def last_sweep(self):
        """Date the filing index was last walked, or None."""
        raw = self._read_state().get('last_index_sweep')
        try:
            return date.fromisoformat(raw) if raw else None
        except (TypeError, ValueError):
            return None

    def record_sweep(self, through):
        """Record that the filing index is walked through *through* (a date)."""
        state = self._read_state()
        state['last_index_sweep'] = (through.isoformat()
                                     if hasattr(through, 'isoformat')
                                     else str(through)[:10])
        state['updated_at'] = datetime.now().isoformat(timespec='seconds')
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            tmp = f'{self._state_path()}.tmp.{os.getpid()}'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(state, f)
            os.replace(tmp, self._state_path())
        except Exception as e:
            logger.warning("sec facts cache: could not record sweep: %s", e)

    # -- reporting --------------------------------------------------------

    def entry_count(self):
        try:
            return sum(1 for f in os.listdir(self.cache_dir)
                       if f.endswith('.json.gz'))
        except OSError:
            return 0

    def disk_bytes(self):
        total = 0
        try:
            for f in os.listdir(self.cache_dir):
                if f.endswith('.json.gz'):
                    total += os.path.getsize(os.path.join(self.cache_dir, f))
        except OSError:
            return 0
        return total

    def stats(self):
        """Counters for the run-quality summary."""
        return {'hits': self.hits, 'misses': self.misses, 'writes': self.writes,
                'invalidated': self.invalidated, 'entries': self.entry_count(),
                'disk_bytes': self.disk_bytes()}
