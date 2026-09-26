"""A Supabase Storage bucket, for the caches that must survive the container.

The cloud container is stateless, so every on-disk cache dies with it. The
snapshots ride the ``data/snapshots`` git branch, but the bigger caches do not
fit that shape, and Storage is the transport already proven here: the same
host, service-role key and HTTPS egress that step ``06a-db-publish`` uses
nightly (``data/db/publish.RestTransport``). P0 found the container cannot
open raw TCP to Postgres; Storage is HTTPS like the Data API, so it is
unaffected.

This module is the plumbing only — list, download, upload, delete, and a small
worker pool. What is safe to restore, and when, is the caller's business:
``data/price_cache_store.py`` and ``data/sec_facts_cache_store.py`` each carry
their own argument for why a stale or partial restore cannot harm a run.
"""
import concurrent.futures
import logging
import os

logger = logging.getLogger(__name__)

# Storage caps a list page at 1000 rows; these caches run to a few thousand.
LIST_PAGE = 1000
# Our own service, not a third party to be polite to, so there is no minimum
# interval by default; the limit is bandwidth. Workers are still bounded.
DEFAULT_WORKERS = int(os.environ.get('CACHE_STORE_WORKERS', 8))


class StorageError(Exception):
    """The bucket could not be read or written."""


class StorageBucket:
    """List, download, upload and delete objects under one bucket prefix."""

    #: only objects ending in this are listed; None lists everything.
    suffix = None

    def __init__(self, url, service_key, bucket, prefix,
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
                    raise StorageError(f'{method} {path}: {e}') from e
                logger.warning('storage: %s %s: %s; retrying (%d/%d)',
                               method, path, e, attempt + 1, attempts - 1)
                _backoff(attempt)
                continue
            if resp.status_code >= 500 and attempt + 1 < attempts:
                logger.warning('storage: %s %s: HTTP %s; retrying (%d/%d)',
                               method, path, resp.status_code, attempt + 1, attempts - 1)
                _backoff(attempt)
                continue
            if resp.status_code >= 400:
                raise StorageError(
                    f'{method} {path}: HTTP {resp.status_code}: {resp.text[:300]}')
            return resp
        raise StorageError(f'{method} {path}: no attempts left')  # pragma: no cover

    def _object_path(self, name):
        return f'/object/{self.bucket}/{self.prefix}/{name}' if self.prefix \
            else f'/object/{self.bucket}/{name}'

    # -- operations -------------------------------------------------------

    def list_objects(self):
        """``{filename: size}`` for every object under the prefix."""
        found, offset = {}, 0
        while True:
            body = {'prefix': self.prefix, 'limit': LIST_PAGE, 'offset': offset,
                    'sortBy': {'column': 'name', 'order': 'asc'}}
            rows = self._request('POST', f'/object/list/{self.bucket}',
                                 retry=True, json=body).json()
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


    def delete(self, names):
        """Remove objects under the prefix. Returns how many were requested.

        Evictions have to reach the bucket: a blob dropped locally but left
        stored would come back on the next restore, and nothing downstream
        would know it was meant to be gone.
        """
        names = list(names)
        if not names:
            return 0
        prefixes = [f'{self.prefix}/{n}' if self.prefix else n for n in names]
        self._request('DELETE', f'/object/{self.bucket}', retry=True,
                      json={'prefixes': prefixes})
        return len(names)


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
