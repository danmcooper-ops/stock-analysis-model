"""TTL disk cache and retrying HTTP fetch shared by every API client.

Pattern lifted from the sibling stock-analysis-model's fdic_client: URL as
cache key, file mtime as TTL, an optional `meta` out-param the caller can feed
straight into provenance.

Two rules here are load-bearing and were learned the hard way upstream:

1. **Failures are never cached.** A transient 502 must not turn into "this
   indicator has no data" for the remaining lifetime of the TTL. Only a
   successful parse is written to disk.

2. **A stale hit beats a hard failure.** When the network attempt fails and an
   *expired* cache entry exists, serve the expired entry and flag it via
   `meta['stale']`. The World Bank API returns 502 on a rotating subset of
   indicators (verified: FP.CPI.TOTL.ZG served fine while BN.CAB.XOKA.GD.ZS
   502'd on every retry), so without this a daily build would randomly lose
   whole pillars. Serving 400-day-old annual macro data with a provenance
   flag is strictly better than serving nothing.
"""

import gzip
import json
import os
import time
import urllib.error
import urllib.request

_CACHE_ROOT = os.path.join(os.path.dirname(__file__), 'cache')
_UA = 'currency-screener/1.0 (+https://github.com/danmcooper-ops/Currency-Arbitrage-)'

DEFAULT_TIMEOUT = 45
DEFAULT_RETRIES = 4


def cache_path(namespace, key):
    d = os.path.join(_CACHE_ROOT, namespace)
    os.makedirs(d, exist_ok=True)
    safe = ''.join(c if (c.isalnum() or c in '-_.') else '_' for c in key)
    # Keep the tail: query params discriminate more than the shared URL prefix.
    return os.path.join(d, safe[-180:] + '.cache')


def _read_cache(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def _write_cache(path, text):
    """Write atomically so a crash mid-write can't leave a truncated file that
    the next run would happily parse as a short/empty dataset."""
    tmp = '%s.tmp.%d' % (path, os.getpid())
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(text)
    os.replace(tmp, path)


def fetch(url, namespace, ttl_days=1.0, headers=None, timeout=DEFAULT_TIMEOUT,
          retries=DEFAULT_RETRIES, meta=None):
    """GET `url` as text, through a TTL disk cache, with retry and stale-fallback.

    Args:
        url: absolute URL.
        namespace: cache subdirectory, e.g. 'ecb'.
        ttl_days: how long a cached body stays fresh. Annual macro series use
            30; daily FX uses ~0.5 so a re-run the same evening still refreshes.
        meta: optional dict; populated with cache_hit / cache_age_days / stale
            for provenance recording.

    Returns:
        The response body as str, or None if the fetch failed and no cache
        entry (fresh or stale) exists. Never raises on network failure —
        callers degrade per-field rather than aborting a whole run.
    """
    path = cache_path(namespace, url)
    if meta is None:
        meta = {}

    if os.path.exists(path):
        age_days = (time.time() - os.path.getmtime(path)) / 86400.0
        if age_days < ttl_days:
            meta.update(cache_hit=True, cache_age_days=round(age_days, 3), stale=False)
            return _read_cache(path)

    req_headers = {'User-Agent': _UA, 'Accept-Encoding': 'gzip'}
    if headers:
        req_headers.update(headers)

    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=req_headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
                if r.headers.get('Content-Encoding') == 'gzip':
                    raw = gzip.decompress(raw)
                text = raw.decode('utf-8', errors='replace')
            _write_cache(path, text)
            meta.update(cache_hit=False, cache_age_days=0.0, stale=False,
                        attempts=attempt + 1)
            return text
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # 1s, 2s, 4s

    # Network exhausted. Fall back to an expired entry rather than losing the
    # field entirely — see module docstring.
    if os.path.exists(path):
        age_days = (time.time() - os.path.getmtime(path)) / 86400.0
        meta.update(cache_hit=True, cache_age_days=round(age_days, 3), stale=True,
                    error=str(last_err))
        return _read_cache(path)

    meta.update(cache_hit=False, stale=False, error=str(last_err), failed=True)
    return None


def fetch_json(url, namespace, **kw):
    """fetch() + json.loads. Returns None on fetch failure or unparseable body.

    A body that fails to parse is treated as a failure, not an empty result:
    the World Bank serves an HTML error page with HTTP 200 in some states, and
    silently reading that as "no data" would zero out a pillar.
    """
    text = fetch(url, namespace, **kw)
    if text is None:
        return None
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        meta = kw.get('meta')
        if isinstance(meta, dict):
            meta.update(failed=True, error='non-json response')
        return None
