"""Carry ``output/prices/`` across runs, in Supabase Storage.

The cloud container is stateless, so the price parquets die with it and
``run.sh`` step 03 starts cold every night: nothing short-circuits, and the
whole universe is re-fetched from Yahoo one ticker at a time. Phase A made
that concurrent; this makes it unnecessary. See ``design/cache-persistence.md``.

Transport is the Supabase Storage REST API — the same host, service-role key
and HTTPS egress that step ``06a-db-publish`` already exercises nightly
(``data/db/publish.RestTransport``). P0 found the container cannot open raw
TCP to Postgres; Storage is unaffected by that, being HTTPS like the Data API.

One object per ticker, ``<prefix>/<TICKER>.parquet``, replaced in place. There
is no history to accumulate and no per-object size cap to shard around.

**Why a stale or partial cache is safe.** Price freshness is read from parquet
*content* (``download_prices._parquet_max_date`` reads the index), never from
file mtime, so a restored file that is behind simply fails the freshness check
and is re-downloaded. A cache that is old, incomplete or corrupt therefore
degrades to today's cold behaviour; it cannot make the run serve stale prices.
That is what lets the restore be non-blocking and the change-detection below
be a heuristic rather than a proof.
"""
import logging
import os

from data.supabase_storage import (DEFAULT_WORKERS, StorageBucket,  # noqa: F401
                                   StorageError, _map)

logger = logging.getLogger(__name__)

DEFAULT_BUCKET = os.environ.get('PRICE_CACHE_BUCKET', 'price-cache')
DEFAULT_PREFIX = os.environ.get('PRICE_CACHE_PREFIX', 'prices')
# Refuse to save a local set this much smaller than what is already stored —
# a run that half-failed should not overwrite a good cache with its remains.
MIN_KEEP_FRACTION = 0.8

#: Kept as an alias so callers that caught this before the storage plumbing
#: was shared keep working.
PriceCacheError = StorageError


class PriceCacheStore(StorageBucket):
    """Read and write the price parquets in a Supabase Storage bucket."""

    suffix = '.parquet'

    def __init__(self, url, service_key, bucket=DEFAULT_BUCKET,
                 prefix=DEFAULT_PREFIX, **kwargs):
        super().__init__(url, service_key, bucket, prefix, **kwargs)

    @classmethod
    def from_env(cls, **kwargs):
        """Build from SUPABASE_URL/SUPABASE_SERVICE_ROLE_KEY, or None if unset.

        None is a normal outcome, not an error: a developer machine and a
        smoke run have no Supabase credentials and must behave exactly as they
        did before this cache existed.
        """
        url = os.environ.get('SUPABASE_URL')
        key = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
        if not url or not key:
            return None
        return cls(url, key, **kwargs)

    def restore(self, dest_dir, workers=DEFAULT_WORKERS):
        """Download every stored parquet that is not already on disk.

        An existing local file always wins: a resumed run may already hold
        something fresher than the bucket, and re-downloading it would be
        both wasted work and a chance to overwrite it with something older.
        """
        os.makedirs(dest_dir, exist_ok=True)
        stored = self.list_objects()
        wanted = [n for n in sorted(stored)
                  if not os.path.exists(os.path.join(dest_dir, n))]
        counts = {'stored': len(stored), 'present': len(stored) - len(wanted),
                  'restored': 0, 'failed': 0, 'bytes': 0}
        if not wanted:
            return counts

        def _one(name):
            return self.download(name, os.path.join(dest_dir, name))

        for name, result in _map(_one, wanted, workers):
            if isinstance(result, Exception):
                counts['failed'] += 1
                logger.warning('price cache: restoring %s failed: %s', name, result)
            else:
                counts['restored'] += 1
                counts['bytes'] += result
        return counts

    def save(self, src_dir, workers=DEFAULT_WORKERS, force=False, all_files=False):
        """Upload local parquets that are new or whose size changed.

        Size is a heuristic for "changed": a re-downloaded history has gained
        bars, so its parquet is a different size in all but pathological
        cases. Getting it wrong leaves a slightly older copy in the bucket,
        and the freshness check re-downloads that ticker on the next run — the
        failure mode is a wasted fetch, never a stale price. ``all_files``
        skips the comparison.
        """
        local = {n: os.path.getsize(os.path.join(src_dir, n))
                 for n in sorted(os.listdir(src_dir)) if n.endswith('.parquet')}
        stored = self.list_objects()
        counts = {'local': len(local), 'stored': len(stored),
                  'uploaded': 0, 'unchanged': 0, 'failed': 0, 'bytes': 0}
        if not local:
            raise PriceCacheError(f'{src_dir} holds no parquets; refusing to save')
        floor = int(len(stored) * MIN_KEEP_FRACTION)
        if len(local) < floor and not force:
            raise PriceCacheError(
                f'{len(local)} local parquets is under {MIN_KEEP_FRACTION:.0%} of the '
                f'{len(stored)} stored ({floor}); refusing to save. Pass force=True if '
                f'the universe really shrank.')

        pending = [n for n, size in local.items()
                   if all_files or stored.get(n) != size]
        counts['unchanged'] = len(local) - len(pending)
        if not pending:
            return counts

        def _one(name):
            return self.upload(name, os.path.join(src_dir, name))

        for name, result in _map(_one, pending, workers):
            if isinstance(result, Exception):
                counts['failed'] += 1
                logger.warning('price cache: uploading %s failed: %s', name, result)
            else:
                counts['uploaded'] += 1
                counts['bytes'] += result
        return counts
