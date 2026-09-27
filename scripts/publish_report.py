#!/usr/bin/env python3
"""Publish the rendered report to the Cloudflare R2 bucket behind the login.

The report is served by a small Worker (cloudflare/worker/) from an R2
bucket, with Cloudflare Access in front of it, so only allowlisted emails can
open it. Cloudflare Pages was not an option: it caps each file at 25 MiB and
details.json (~32 MB) and index.html (~25 MB, DATA inlined) are both past it.

This syncs a staged docs/ directory (the same file set the old pages-live
branch carried: index.html, the JSON sidecars, px/ and vol/ shards) into the
bucket over R2's S3 API, then checks that the live site serves the run.

  * Only changed files are uploaded: R2's ETag for a single-part PUT is the
    object's MD5, compared against the local file.
  * index.html goes up last and stale keys are deleted after it, so a
    half-finished publish never serves a new page against old sidecars, and
    the previous page never loses a shard it still references.
  * The live check authenticates with an Access service token
    (CF-Access-Client-Id/-Secret); without one Access answers every request
    with its login page.

The S3 signing (SigV4) is done here with the standard library rather than
through boto3, which the pipeline otherwise has no use for; the unit tests
pin it to AWS's published example signature.

Environment:
  R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY   required
  R2_BUCKET                  bucket name (default stock-report)
  REPORT_URL                 the protected site, for the live check
  CF_ACCESS_CLIENT_ID, CF_ACCESS_CLIENT_SECRET            service token

Exit codes: 0 published (and verified, unless --no-verify); 1 the upload or
the live check failed; 2 not configured (R2 credentials unset).
"""

import argparse
import datetime as dt
import hashlib
import hmac
import logging
import os
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import requests

log = logging.getLogger(__name__)

DEFAULT_BUCKET = 'stock-report'
INDEX_KEY = 'index.html'
UPLOAD_WORKERS = 16
EMPTY_SHA256 = hashlib.sha256(b'').hexdigest()
S3_NS = '{http://s3.amazonaws.com/doc/2006-03-01/}'
CONTENT_TYPES = {
    '.html': 'text/html; charset=utf-8',
    '.json': 'application/json; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.txt': 'text/plain; charset=utf-8',
    '.png': 'image/png',
    '.svg': 'image/svg+xml',
    '.ico': 'image/x-icon',
}


def content_type(key):
    return CONTENT_TYPES.get(os.path.splitext(key)[1].lower(), 'application/octet-stream')


# --------------------------------------------------------------------- SigV4
def _hmac(key, msg):
    return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()


def _quote(s, safe='-_.~'):
    return urllib.parse.quote(s, safe=safe)


def sign_request(method, url, headers, payload_sha256, access_key, secret_key,
                 region='auto', service='s3', now=None):
    """Return `headers` plus the SigV4 Authorization, x-amz-date and
    x-amz-content-sha256 headers for this request. Every header passed in is
    signed. `now` is a UTC datetime (tests pin it)."""
    now = now or dt.datetime.now(dt.timezone.utc)
    amz_date = now.strftime('%Y%m%dT%H%M%SZ')
    datestamp = now.strftime('%Y%m%d')
    parts = urllib.parse.urlsplit(url)

    out = dict(headers)
    out['host'] = parts.netloc
    out['x-amz-date'] = amz_date
    out['x-amz-content-sha256'] = payload_sha256
    canon = {k.lower().strip(): ' '.join(str(v).split()) for k, v in out.items()}
    signed_headers = ';'.join(sorted(canon))
    canonical_headers = ''.join(f'{k}:{canon[k]}\n' for k in sorted(canon))

    query = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
    canonical_query = '&'.join(f'{_quote(k)}={_quote(v)}' for k, v in sorted(query))
    canonical_path = _quote(urllib.parse.unquote(parts.path) or '/', safe='-_.~/')

    canonical_request = '\n'.join([method, canonical_path, canonical_query,
                                   canonical_headers, signed_headers, payload_sha256])
    scope = f'{datestamp}/{region}/{service}/aws4_request'
    string_to_sign = '\n'.join(['AWS4-HMAC-SHA256', amz_date, scope,
                                hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()])
    k = _hmac(('AWS4' + secret_key).encode('utf-8'), datestamp)
    k = _hmac(k, region)
    k = _hmac(k, service)
    k = _hmac(k, 'aws4_request')
    signature = hmac.new(k, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()

    out['Authorization'] = (f'AWS4-HMAC-SHA256 Credential={access_key}/{scope}, '
                            f'SignedHeaders={signed_headers}, Signature={signature}')
    del out['host']  # requests sets Host itself, to the same value
    return out


# --------------------------------------------------------------------- R2
class R2Bucket:
    """The three S3 calls the publish needs: list, put, delete."""

    def __init__(self, account_id, access_key, secret_key, bucket, session=None, timeout=120):
        self.base = f'https://{account_id}.r2.cloudflarestorage.com/{bucket}'
        self.access_key = access_key
        self.secret_key = secret_key
        self.session = session or requests.Session()
        self.timeout = timeout

    def _request(self, method, key='', query=None, body=b'', headers=None):
        url = self.base + ('/' + _quote(key, safe='-_.~/') if key else '')
        if query:
            url += '?' + urllib.parse.urlencode(sorted(query.items()), quote_via=urllib.parse.quote)
        payload_hash = hashlib.sha256(body).hexdigest() if body else EMPTY_SHA256
        signed = sign_request(method, url, headers or {}, payload_hash, self.access_key, self.secret_key)
        resp = self.session.request(method, url, data=body or None, headers=signed, timeout=self.timeout)
        if resp.status_code >= 300:
            raise RuntimeError(f'R2 {method} {key or "/"} -> HTTP {resp.status_code}: {resp.text[:300]}')
        return resp

    def list_etags(self):
        """{key: md5 hex} for every object in the bucket."""
        out, token = {}, None
        while True:
            query = {'list-type': '2', 'max-keys': '1000'}
            if token:
                query['continuation-token'] = token
            root = ET.fromstring(self._request('GET', query=query).content)
            for item in root.iter(f'{S3_NS}Contents'):
                out[item.findtext(f'{S3_NS}Key')] = (item.findtext(f'{S3_NS}ETag') or '').strip('"')
            if (root.findtext(f'{S3_NS}IsTruncated') or '').lower() != 'true':
                return out
            token = root.findtext(f'{S3_NS}NextContinuationToken')

    def put(self, key, body):
        self._request('PUT', key, body=body, headers={'content-type': content_type(key)})

    def delete(self, key):
        self._request('DELETE', key)


# --------------------------------------------------------------------- sync
def local_files(docs_dir):
    """{key: absolute path} for every file under docs_dir; keys use '/'."""
    out = {}
    for root, _dirs, files in os.walk(docs_dir):
        for name in files:
            path = os.path.join(root, name)
            out[os.path.relpath(path, docs_dir).replace(os.sep, '/')] = path
    return out


def md5_file(path):
    h = hashlib.md5(usedforsecurity=False)
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def plan_sync(local_md5, remote_etags):
    """(uploads, deletes) turning the bucket into the local tree. Uploads are
    ordered with index.html last; deletes are every remote key not local."""
    uploads = sorted(k for k, md5 in local_md5.items() if remote_etags.get(k) != md5)
    if INDEX_KEY in uploads:
        uploads.remove(INDEX_KEY)
        uploads.append(INDEX_KEY)
    deletes = sorted(set(remote_etags) - set(local_md5))
    return uploads, deletes


def sync(bucket, docs_dir, dry_run=False, workers=UPLOAD_WORKERS):
    files = local_files(docs_dir)
    if INDEX_KEY not in files:
        raise RuntimeError(f'{docs_dir} has no {INDEX_KEY}; refusing to publish a site without a page')
    with ThreadPoolExecutor(workers) as pool:
        local_md5 = dict(zip(files, pool.map(md5_file, files.values()), strict=True))
    remote = bucket.list_etags()
    uploads, deletes = plan_sync(local_md5, remote)
    print(f'publish: {len(files)} local files, {len(remote)} in bucket; '
          f'{len(uploads)} to upload, {len(deletes)} to delete, '
          f'{len(files) - len(uploads)} unchanged')
    if dry_run:
        print('publish: dry run — nothing sent')
        return uploads, deletes

    def put(key):
        with open(files[key], 'rb') as fh:
            bucket.put(key, fh.read())

    body = [k for k in uploads if k != INDEX_KEY]
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(put, body))       # re-raises the first failure
    if INDEX_KEY in uploads:
        put(INDEX_KEY)
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(bucket.delete, deletes))
    print(f'publish: uploaded {len(uploads)}, deleted {len(deletes)}')
    return uploads, deletes


# --------------------------------------------------------------------- verify
def verify_live(url, rundate, client_id, client_secret, attempts=3, delay=10, session=None):
    """True once `url` (fetched with the Access service token) contains
    `rundate`. R2 is read-after-write consistent, so this should pass on the
    first try; the retries cover edge propagation."""
    session = session or requests.Session()
    headers = {'CF-Access-Client-Id': client_id, 'CF-Access-Client-Secret': client_secret}
    status = None
    for i in range(attempts):
        if i:
            time.sleep(delay)
        try:
            resp = session.get(url, headers=headers, timeout=60, allow_redirects=False)
            status = resp.status_code
            if status == 200 and rundate in resp.text:
                print(f'live: {url} serves the {rundate} report')
                return True
        except requests.RequestException as e:
            log.warning('live check of %s failed: %s', url, e)
            status = type(e).__name__
    hint = ' (a 302/403 means the service token was refused by Access)' if status in (302, 403) else ''
    print(f'WARNING: {url} did not serve {rundate} (last status {status}){hint}')
    return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('docs', help='staged site directory (index.html, sidecars, px/, vol/)')
    ap.add_argument('--rundate', required=True, help='YYYY-MM-DD the live page must show')
    ap.add_argument('--dry-run', action='store_true', help='plan the sync, send nothing')
    ap.add_argument('--no-verify', action='store_true', help='skip the live check')
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

    env = os.environ
    missing = [k for k in ('R2_ACCOUNT_ID', 'R2_ACCESS_KEY_ID', 'R2_SECRET_ACCESS_KEY') if not env.get(k)]
    if missing:
        print(f'publish: not configured ({", ".join(missing)} unset)')
        return 2
    bucket = R2Bucket(env['R2_ACCOUNT_ID'], env['R2_ACCESS_KEY_ID'], env['R2_SECRET_ACCESS_KEY'],
                      env.get('R2_BUCKET') or DEFAULT_BUCKET)
    try:
        sync(bucket, args.docs, dry_run=args.dry_run)
    except Exception as e:
        log.warning('publish to R2 failed: %s', e)
        return 1
    if args.dry_run or args.no_verify:
        return 0

    url = env.get('REPORT_URL')
    cid, secret = env.get('CF_ACCESS_CLIENT_ID'), env.get('CF_ACCESS_CLIENT_SECRET')
    if not (url and cid and secret):
        print('WARNING: REPORT_URL / CF_ACCESS_CLIENT_ID / CF_ACCESS_CLIENT_SECRET unset — '
              'uploaded but not verified live')
        return 1
    return 0 if verify_live(url, args.rundate, cid, secret) else 1


if __name__ == '__main__':
    sys.exit(main())
