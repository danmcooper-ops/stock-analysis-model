"""A Supabase Storage bucket of cache objects, for what must outlive the container.

The cloud container is stateless, so every on-disk cache dies with it. The
snapshots ride the ``data/snapshots`` git branch, but the bigger caches do not
fit that shape, and Storage is the transport already proven here
(``data/db/storage.py``, P4b): the same host, service-role key and HTTPS
egress the publish RPCs use. P0 found the container cannot open raw TCP to
Postgres; Storage is HTTPS like the Data API, so it is unaffected.

The request layer is ``data/db/storage.StorageClient`` — one Storage client in
the repo, not two. What this adds is what a *cache* needs and a publish does
not: listing a prefix (paginated), deleting evicted objects, and a worker pool,
since these caches move thousands of small objects rather than two big ones.

This module is plumbing only. What is safe to restore, and when, is the
caller's business: ``data/price_cache_store.py`` and
``data/sec_facts_cache_store.py`` each carry their own argument for why a
stale or partial restore cannot harm a run.
"""
import concurrent.futures
import logging
import os

from data.db.storage import StorageClient, StorageError  # noqa: F401

logger = logging.getLogger(__name__)

# Storage caps a list page at 1000 rows; these caches run to a few thousand.
LIST_PAGE = 1000
# Our own service, not a third party to be polite to, so there is no minimum
# interval by default; the limit is bandwidth. Workers are still bounded.
DEFAULT_WORKERS = int(os.environ.get('CACHE_STORE_WORKERS', 8))


class StorageBucket:
    """List, download, upload and delete cache objects under one prefix."""

    #: only objects ending in this are listed; None lists everything.
    suffix = None

    def __init__(self, url, service_key, bucket, prefix, session=None,
                 throttle=None, timeout=(5, 120), retries=3, client=None):
        self.client = client or StorageClient(url, service_key, session=session,
                                              timeout=timeout, retries=retries)
        self.bucket = bucket
        self.prefix = prefix.strip('/')
        self.throttle = throttle

    def _key(self, name):
        return f'{self.prefix}/{name}' if self.prefix else name

    def _json(self, method, path, **kw):
        """A call whose body we read, raising rather than returning a status.

        StorageClient hands back 4xx for its callers to inspect (ensure_bucket
        needs to see a 404); a cache has nothing to do with one but fail.
        """
        if self.throttle:
            self.throttle()
        resp = self.client._request(method, path, **kw)
        if resp.status_code >= 400:
            raise StorageError(f'{method} {path}: HTTP {resp.status_code}: {resp.text[:300]}')
        return resp

    def ensure_bucket(self):
        """Create the bucket if this is the first run. Returns True if created.

        Worth doing rather than documenting as a prerequisite: the first night
        would otherwise fail every upload against a bucket nobody had made.
        """
        return self.client.ensure_bucket(self.bucket)

    def list_objects(self):
        """``{filename: size}`` for every object under the prefix."""
        found, offset = {}, 0
        while True:
            body = {'prefix': self.prefix, 'limit': LIST_PAGE, 'offset': offset,
                    'sortBy': {'column': 'name', 'order': 'asc'}}
            rows = self._json('POST', f'/object/list/{self.bucket}', json=body).json()
            if not isinstance(rows, list):
                raise StorageError(f'list: expected a list, got {type(rows).__name__}')
            for row in rows:
                name = (row or {}).get('name')
                if not name or (self.suffix and not name.endswith(self.suffix)):
                    continue
                meta = row.get('metadata') or {}
                size = meta.get('size')
                found[os.path.basename(name)] = size if isinstance(size, int) else -1
            if len(rows) < LIST_PAGE:
                return found
            offset += len(rows)

    def download(self, name, dest):
        """Fetch one object to *dest*, atomically. Returns bytes written."""
        if self.throttle:
            self.throttle()
        body = self.client.download(self._key(name), bucket=self.bucket)
        tmp = f'{dest}.tmp.{os.getpid()}.{_thread_id()}'
        try:
            with open(tmp, 'wb') as fh:
                fh.write(body)
            os.replace(tmp, dest)
        except Exception:
            _unlink(tmp)
            raise
        return len(body)

    def upload(self, name, src):
        """Write *src* to the object, replacing any existing one."""
        with open(src, 'rb') as fh:
            payload = fh.read()
        if self.throttle:
            self.throttle()
        # x-upsert (StorageClient.upload sets it) makes this a replace, which
        # is what a refresh always is — and what makes a retried upload safe.
        self.client.upload(self._key(name), payload, 'application/octet-stream',
                           bucket=self.bucket)
        return len(payload)

    def delete(self, names):
        """Remove objects under the prefix. Returns how many were requested.

        Evictions have to reach the bucket: an object dropped locally but left
        stored would come back on the next restore, and nothing downstream
        would know it was meant to be gone.
        """
        names = list(names)
        if not names:
            return 0
        self._json('DELETE', f'/object/{self.bucket}',
                   json={'prefixes': [self._key(n) for n in names]})
        return len(names)


# -- helpers ---------------------------------------------------------------

def _map(fn, names, workers):
    """Run *fn* over *names* on a pool, yielding ``(name, result-or-exception)``.

    Results come back in submission order, and one object's failure never
    ends the sweep: a cache is worth having partially.
    """
    workers = max(1, int(workers))
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers,
                                                 thread_name_prefix='cache-store')
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


def _thread_id():
    import threading
    return threading.get_ident()


def _unlink(path):
    try:
        os.remove(path)
    except OSError as e:
        logger.debug('cache store: could not remove %s: %s', path, e)
