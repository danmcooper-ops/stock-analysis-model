"""Carry ``data/cache/sec_facts/`` across runs, in Supabase Storage.

The companyfacts blobs die with the stateless container, so every night
re-downloads the corpus SEC already served yesterday — ~3.9 MB per filer
uncompressed. The Phase-1 prefetch pool hides most of that latency behind
other work, so the win here is fewer SEC requests and a smaller failure
surface rather than wall clock; see ``design/cache-persistence.md`` (Phase C).

**Why a restored cache is safe, and what had to change first.** Unlike the
price parquets, these blobs carry no internal date to check, and their
freshness backstop read file mtime — which a restore resets, so it would have
said "fetched today" about a blob of any age and never expired anything again.
Transport alone would have quietly disabled the one guard against
indefinitely-stale fundamentals.

So the backstop now also reads the sweep watermark
(``SECFactsCache.sweep_is_lagging``), which travels with the blobs in
``_state.json`` and measures the thing that actually matters: whether
filing-driven eviction has kept up. A restored cache whose watermark is stale
reads as entirely missing. That is what makes this transport sound, and it is
why the watermark must be restored and saved **with** the blobs, never
separately.

Evictions travel too. ``invalidate()`` removes a blob locally because its
filer filed; if the object stayed in the bucket it would return on the next
restore, and the sweep would not evict it again — the watermark has already
moved past that day. So the save deletes stored objects with no local file.
"""
import logging
import os

from data.sec_facts_cache import DEFAULT_CACHE_DIR, _STATE_FILE
from data.supabase_storage import (DEFAULT_WORKERS, StorageBucket,  # noqa: F401
                                   StorageError, _map)

logger = logging.getLogger(__name__)

DEFAULT_BUCKET = os.environ.get('SEC_CACHE_BUCKET', 'sec-facts-cache')
DEFAULT_PREFIX = os.environ.get('SEC_CACHE_PREFIX', 'sec_facts')
# Refuse to save a local set this much smaller than what is stored: a run that
# half-failed, or whose restore did, must not delete most of a good cache.
MIN_KEEP_FRACTION = 0.8


class SecFactsCacheStore(StorageBucket):
    """Read and write the companyfacts blobs in a Supabase Storage bucket."""

    #: the blobs plus _state.json; the watermark is part of the cache, not
    #: metadata about it, and restoring one without the other is unsound.
    suffix = None

    def __init__(self, url, service_key, bucket=DEFAULT_BUCKET,
                 prefix=DEFAULT_PREFIX, **kwargs):
        super().__init__(url, service_key, bucket, prefix, **kwargs)

    @classmethod
    def from_env(cls, **kwargs):
        """Build from SUPABASE_URL/SUPABASE_SERVICE_ROLE_KEY, or None if unset."""
        url = os.environ.get('SUPABASE_URL')
        key = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
        if not url or not key:
            return None
        return cls(url, key, **kwargs)

    @staticmethod
    def _local(cache_dir):
        """``{name: size}`` for the blobs and the watermark file."""
        try:
            names = os.listdir(cache_dir)
        except OSError:
            return {}
        return {n: os.path.getsize(os.path.join(cache_dir, n)) for n in names
                if n.endswith('.json.gz') or n == _STATE_FILE}

    def restore(self, cache_dir=DEFAULT_CACHE_DIR, workers=DEFAULT_WORKERS):
        """Download the cache, watermark included, skipping what is on disk.

        The watermark goes down with everything else. Restoring blobs without
        it would leave the cache claiming a sweep it cannot evidence; getting
        only the watermark would vouch for blobs that are not there. Either
        way the sweep-lag guard is what catches it, so a partial restore is
        safe — it just costs fetches.
        """
        os.makedirs(cache_dir, exist_ok=True)
        stored = self.list_objects()
        on_disk = self._local(cache_dir)
        wanted = [n for n in sorted(stored) if n not in on_disk]
        counts = {'stored': len(stored), 'present': len(stored) - len(wanted),
                  'restored': 0, 'failed': 0, 'bytes': 0}
        if not wanted:
            return counts

        def _one(name):
            return self.download(name, os.path.join(cache_dir, name))

        for name, result in _map(_one, wanted, workers):
            if isinstance(result, Exception):
                counts['failed'] += 1
                logger.warning('sec cache: restoring %s failed: %s', name, result)
            else:
                counts['restored'] += 1
                counts['bytes'] += result
        return counts

    def save(self, cache_dir=DEFAULT_CACHE_DIR, workers=DEFAULT_WORKERS,
             force=False, all_files=False):
        """Upload new or changed entries and delete the ones evicted locally.

        A blob is only rewritten locally when it was re-fetched, so a size
        difference is a reliable enough signal of "changed"; being wrong
        leaves the bucket one revision behind and costs a refetch next run.
        ``_state.json`` is always uploaded — it is small, it changes every
        sweep, and a watermark behind its blobs is the one combination the
        guard reads as a lagging sweep.
        """
        local = self._local(cache_dir)
        stored = self.list_objects()
        counts = {'local': len(local), 'stored': len(stored), 'uploaded': 0,
                  'unchanged': 0, 'deleted': 0, 'failed': 0, 'bytes': 0}
        if not local:
            raise StorageError(f'{cache_dir} holds no companyfacts blobs; '
                               f'refusing to save')
        floor = int(len(stored) * MIN_KEEP_FRACTION)
        if len(local) < floor and not force:
            raise StorageError(
                f'{len(local)} local entries is under {MIN_KEEP_FRACTION:.0%} of the '
                f'{len(stored)} stored ({floor}); refusing to save. Pass force=True '
                f'if the cache really shrank this much.')

        # The first run of all finds no bucket; make it rather than failing
        # every upload against something nobody created.
        self.ensure_bucket()

        pending = [n for n, size in local.items()
                   if all_files or n == _STATE_FILE or stored.get(n) != size]
        counts['unchanged'] = len(local) - len(pending)

        def _one(name):
            return self.upload(name, os.path.join(cache_dir, name))

        for name, result in _map(_one, pending, workers):
            if isinstance(result, Exception):
                counts['failed'] += 1
                logger.warning('sec cache: uploading %s failed: %s', name, result)
            else:
                counts['uploaded'] += 1
                counts['bytes'] += result

        # Evictions have to reach the bucket or they undo themselves on the
        # next restore. Only safe because the floor above already established
        # that this run holds a substantially complete cache.
        gone = [n for n in stored if n not in local]
        if gone:
            try:
                counts['deleted'] = self.delete(gone)
            except StorageError as e:
                counts['failed'] += len(gone)
                logger.warning('sec cache: deleting %d evicted entr(ies) failed: %s',
                               len(gone), e)
        return counts
