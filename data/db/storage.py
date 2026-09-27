"""Supabase Storage uploads for published runs (P4b).

After ``publish_run`` commits a day, :func:`upload_run` stores two objects in
the private ``snapshots`` bucket, each verified by downloading it back and
comparing SHA-256:

* ``json/results_<date>.json.gz``: the canonical snapshot, self-contained
  (edgar_history inline) and gzipped deterministically. This is the full row,
  report-only keys included, which the database leaves out.
* ``parquet/results_<date>.parquet``: the analytics export (data/db/parquet.py).

Both are recorded in ``core.snapshot_objects`` through
``pipeline.record_snapshot_objects``. Storage is reached over HTTPS with the
service-role key, like the RPCs (A1).
"""
import gzip
import hashlib
import json
import logging
import os
import time

from data.db.parquet import export_snapshot, parquet_path

logger = logging.getLogger(__name__)

BUCKET = 'snapshots'
_COMPACT = (',', ':')


class StorageError(RuntimeError):
    pass


class StorageClient:
    """The few Storage API calls a publish needs."""

    def __init__(self, url, service_key, session=None, timeout=(5, 300), retries=3):
        import requests
        self.base = url.rstrip('/') + '/storage/v1'
        self.session = session or requests.Session()
        self.headers = {'apikey': service_key, 'Authorization': f'Bearer {service_key}'}
        self.timeout = timeout
        self.retries = retries

    def _request(self, method, path, **kw):
        import requests
        headers = dict(self.headers, **kw.pop('headers', {}))
        for attempt in range(self.retries + 1):
            try:
                resp = self.session.request(method, self.base + path, headers=headers, timeout=self.timeout, **kw)
            except requests.RequestException as e:
                if attempt >= self.retries:
                    raise StorageError(f'{method} {path}: {e}') from e
                logger.warning('storage %s %s: %s; retrying', method, path, e)
                time.sleep(2 ** attempt)
                continue
            if resp.status_code >= 500 and attempt < self.retries:
                logger.warning('storage %s %s: HTTP %s; retrying', method, path, resp.status_code)
                time.sleep(2 ** attempt)
                continue
            return resp
        raise StorageError(f'{method} {path}: no attempts left')   # pragma: no cover

    def ensure_bucket(self, bucket=BUCKET):
        """Create the private *bucket* if it does not exist yet."""
        resp = self._request('GET', f'/bucket/{bucket}')
        if resp.status_code == 200:
            return False
        resp = self._request('POST', '/bucket', json={'id': bucket, 'name': bucket, 'public': False})
        if resp.status_code >= 400 and 'already exists' not in resp.text.lower():
            raise StorageError(f'create bucket {bucket}: HTTP {resp.status_code}: {resp.text[:300]}')
        return True

    def upload(self, path, data, content_type, bucket=BUCKET):
        resp = self._request('POST', f'/object/{bucket}/{path}', data=data,
                             headers={'Content-Type': content_type, 'x-upsert': 'true'})
        if resp.status_code >= 400:
            raise StorageError(f'upload {bucket}/{path}: HTTP {resp.status_code}: {resp.text[:300]}')

    def download(self, path, bucket=BUCKET):
        resp = self._request('GET', f'/object/{bucket}/{path}')
        if resp.status_code >= 400:
            raise StorageError(f'download {bucket}/{path}: HTTP {resp.status_code}: {resp.text[:300]}')
        return resp.content


def snapshot_gzip(data):
    """The canonical snapshot encoding (write_snapshot_file's), gzipped
    deterministically: identical content gives identical bytes."""
    raw = json.dumps(data, separators=_COMPACT, default=str).encode('utf-8')
    return gzip.compress(raw, compresslevel=9, mtime=0)


def _sha256(b):
    return hashlib.sha256(b).hexdigest()


def upload_run(data, run_date, storage, transport, parquet_dir, n_rows=None):
    """Export, upload, verify and record one published run; returns the manifest."""
    storage.ensure_bucket()
    body = snapshot_gzip(data)
    json_key = f'json/results_{run_date}.json.gz'
    pq_file = parquet_path(parquet_dir, run_date)
    n, pq_sha = export_snapshot(data, run_date, pq_file)
    with open(pq_file, 'rb') as f:
        pq_body = f.read()
    pq_key = f'parquet/results_{run_date}.parquet'
    for key, blob, ctype in ((json_key, body, 'application/gzip'), (pq_key, pq_body, 'application/octet-stream')):
        storage.upload(key, blob, ctype)
        got = storage.download(key)
        if _sha256(got) != _sha256(blob):
            raise StorageError(f'{key}: the stored copy does not match what was uploaded')
    manifest = {'p_run_date': run_date, 'p_json_path': json_key, 'p_json_sha256': _sha256(body),
                'p_json_bytes': len(body), 'p_parquet_path': pq_key, 'p_parquet_sha256': pq_sha,
                'p_n_rows': n_rows if n_rows is not None else n}
    transport.call('record_snapshot_objects', manifest, idempotent=True)
    return {k[2:]: v for k, v in manifest.items()}


def storage_from_env():
    """A :class:`StorageClient` from SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY, or None."""
    url, key = os.environ.get('SUPABASE_URL'), os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
    return StorageClient(url, key) if url and key else None
