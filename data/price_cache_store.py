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
import concurrent.futures
import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_BUCKET = os.environ.get('PRICE_CACHE_BUCKET', 'price-cache')
DEFAULT_PREFIX = os.environ.get('PRICE_CACHE_PREFIX', 'prices')
# Storage caps a list page at 1000 rows; the universe is ~2,300 objects.
LIST_PAGE = 1000
# Refuse to save a local set this much smaller than what is already stored —
# a run that half-failed should not overwrite a good cache with its remains.
MIN_KEEP_FRACTION = 0.8
# Our own service, not a third party to be polite to, so there is no minimum
# interval by default; the limit is bandwidth. Workers are still bounded.
DEFAULT_WORKERS = int(os.environ.get('PRICE_CACHE_WORKERS', 8))


class PriceCacheError(Exception):
    """The store could not be read or written."""


class PriceCacheStore:
    """Read and write the price parquets in a Supabase Storage bucket."""

    def __init__(self, url, service_key, bucket=DEFAULT_BUCKET, prefix=DEFAULT_PREFIX,
                 session=None, throttle=None, timeout=(5, 120), retries=3):
        import requests
        self.base = url.rstrip('/') + '/storage/v1'
        self.bucket = bucket
        self.prefix = prefix.strip('/')
        self.session = session or requests.Session()
        self.headers = {
            'apikey': service_key,
            'Authorization': f'Bearer {service_key}',
        }
        self.throttle = throttle
        self.timeout = timeout
        self.retries = retries

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

    # -- HTTP -------------------------------------------------------------

    def _request(self, method, path, *, retry, headers=None, **kwargs):
        """One Storage call, retrying transport errors and 5xx when *retry*.

        *headers* adds to the auth headers rather than replacing them; every
        call needs the service key.
        """
        import requests
        hdrs = {**self.headers, **headers} if headers else self.headers
        attempts = self.retries + 1 if retry else 1
        for attempt in range(attempts):
            if self.throttle:
                self.throttle()
            try:
                resp = self.session.request(method, self.base + path, headers=hdrs,
                                            timeout=self.timeout, **kwargs)
            except requests.RequestException as e:
                if attempt + 1 >= attempts:
                    raise PriceCacheError(f'{method} {path}: {e}') from e
                logger.warning('price cache: %s %s: %s; retrying (%d/%d)',
                               method, path, e, attempt + 1, attempts - 1)
                _backoff(attempt)
                continue
            if resp.status_code >= 500 and attempt + 1 < attempts:
                logger.warning('price cache: %s %s: HTTP %s; retrying (%d/%d)',
                               method, path, resp.status_code, attempt + 1, attempts - 1)
                _backoff(attempt)
                continue
            if resp.status_code >= 400:
                raise PriceCacheError(
                    f'{method} {path}: HTTP {resp.status_code}: {resp.text[:300]}')
            return resp
        raise PriceCacheError(f'{method} {path}: no attempts left')  # pragma: no cover

    def _object_path(self, name):
        return f'/object/{self.bucket}/{self.prefix}/{name}' if self.prefix \
            else f'/object/{self.bucket}/{name}'

    # -- operations -------------------------------------------------------

    def list_objects(self):
        """``{filename: size}`` for every parquet under the prefix."""
        found, offset = {}, 0
        while True:
            body = {'prefix': self.prefix, 'limit': LIST_PAGE, 'offset': offset,
                    'sortBy': {'column': 'name', 'order': 'asc'}}
            rows = self._request('POST', f'/object/list/{self.bucket}',
                                 retry=True, json=body).json()
            if not isinstance(rows, list):
                raise PriceCacheError(f'list: expected a list, got {type(rows).__name__}')
            for row in rows:
                name = (row or {}).get('name')
                if not name or not name.endswith('.parquet'):
                    continue
                meta = row.get('metadata') or {}
                size = meta.get('size')
                found[os.path.basename(name)] = size if isinstance(size, int) else -1
            if len(rows) < LIST_PAGE:
                return found
            offset += len(rows)

    def download(self, name, dest):
        """Fetch one object to *dest*, atomically. Returns bytes written."""
        resp = self._request('GET', self._object_path(name), retry=True)
        tmp = f'{dest}.tmp.{os.getpid()}.{_thread_id()}'
        try:
            with open(tmp, 'wb') as fh:
                fh.write(resp.content)
            os.replace(tmp, dest)
        except Exception:
            _unlink(tmp)
            raise
        return len(resp.content)

    def upload(self, name, src):
        """Write *src* to the object, replacing any existing one."""
        with open(src, 'rb') as fh:
            payload = fh.read()
        # POST creates and 409s on an existing object; x-upsert makes it a
        # replace, which is what a nightly refresh always is.
        # Not retried: a partial upload would be replayed over a good object,
        # and the caller's next run re-uploads anyway.
        self._request('POST', self._object_path(name), retry=False, data=payload,
                      headers={'Content-Type': 'application/octet-stream',
                               'x-upsert': 'true'})
        return len(payload)

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


# -- helpers ---------------------------------------------------------------

def _map(fn, names, workers):
    """Run *fn* over *names* on a pool, yielding ``(name, result-or-exception)``.

    Results come back in submission order, and one object's failure never
    ends the sweep: a cache is worth having partially.
    """
    workers = max(1, int(workers))
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers,
                                                 thread_name_prefix='price-cache')
    try:
        for name, future in [(n, pool.submit(fn, n)) for n in names]:
            try:
                yield name, future.result()
            except Exception as e:      # noqa: BLE001 - reported per object above
                yield name, e
    finally:
        # cancel_futures drops whatever has not started. On the normal path
        # every future is already consumed, so it is a no-op; on an abandoned
        # sweep (Ctrl-C, an exception upstream) it stops the caller waiting
        # out a queue of objects nobody will read.
        pool.shutdown(wait=True, cancel_futures=True)


def _backoff(attempt):
    import time
    time.sleep(2 ** attempt)


def _thread_id():
    import threading
    return threading.get_ident()


def _unlink(path):
    try:
        os.remove(path)
    except OSError as e:
        logger.debug('price cache: could not remove %s: %s', path, e)
