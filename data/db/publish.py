"""Publish a snapshot to Supabase: stage it in chunks, then publish atomically.

See ``design/supabase-migration.md`` (P2) and
``supabase/migrations/*_publish_rpc.sql``. The same two RPCs are reached over
either transport:

* :class:`RestTransport`: HTTPS to the Data API with the service-role key.
  This is the only path the cloud pipeline has (amendment A1).
* :class:`DirectTransport`: a psycopg connection (dev machines, CI, admin).

``stage_chunk`` is idempotent per ``(load_id, chunk_no)``, so a chunk is
retried on network errors. ``publish_run`` is not retried: when its outcome is
unknown, the whole publish is re-run under a new ``load_id``. That is safe
because the database replaces the date in one transaction.
"""
import hashlib
import json
import logging
import math
import time
import uuid

from data.db.codec import blob_sha, cast, CastError, encode, split_row
from data.db.columns import COLUMNS
from data.snapshot_store import split_snapshot

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_BYTES = 1_000_000
# Share of typed values allowed to miss their column before a publish is
# refused without --force (R9). New keys that land in extra never count.
MAX_CAST_FAILURE_RATE = 0.01
# A run with fewer rows than this share of the previous complete run is
# refused without --force: more likely a broken night than a real change.
MIN_ROW_RATIO = 0.7
_COMPACT = (',', ':')
_NONFINITE_JSON = {math.inf: 'Infinity', -math.inf: '-Infinity'}


class PublishError(RuntimeError):
    """The snapshot was refused before or by ``publish_run``."""


def canonical_sha256(data):
    """SHA-256 of the snapshot in the canonical encoding (write_snapshot_file's).

    It equals the file's own hash for a plain ``results_<date>.json``. Readers
    compare it with ``core.runs.source_sha256`` to spot a rewritten file (R2).
    """
    text = json.dumps(data, separators=_COMPACT, default=str)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _json_typed(v):
    """A typed column value as JSON that ``float8in`` reads back exactly."""
    if isinstance(v, float) and not math.isfinite(v):
        return 'NaN' if math.isnan(v) else _NONFINITE_JSON[v]
    return v


class Load:
    """A snapshot in staged form: payload rows, blobs and client-side stats."""

    def __init__(self, run_date, rows, blobs, run, stats):
        self.run_date = run_date
        self.rows = rows            # [{"ticker", "cols", "extra", "edgar_history_sha"}]
        self.blobs = blobs          # {sha: encoded value}
        self.run = run              # p_run for publish_run
        self.stats = stats

    def chunks(self, max_bytes=DEFAULT_CHUNK_BYTES):
        """``[(rows, blobs), ...]`` of about *max_bytes* each; a blob rides with
        the first row that references it."""
        out, rows, blobs, size, sent = [], [], [], 0, set()
        for r in self.rows:
            extra_blobs = []
            sha = r.get('edgar_history_sha')
            if sha and sha not in sent:
                sent.add(sha)
                extra_blobs.append({'sha': sha, 'value': self.blobs[sha]})
            n = len(json.dumps(r, separators=_COMPACT)) + sum(
                len(json.dumps(b, separators=_COMPACT)) for b in extra_blobs)
            if rows and size + n > max_bytes:
                out.append((rows, blobs))
                rows, blobs, size = [], [], 0
            rows.append(r)
            blobs.extend(extra_blobs)
            size += n
        if rows:
            out.append((rows, blobs))
        return out


def build_load(data, run_date, columns=COLUMNS):
    """A :class:`Load` for the resolved snapshot *data* (``read_snapshot``)."""
    meta, rows = split_snapshot(data)
    by_ticker = {}
    for r in rows:                          # last one wins, as the stores dedupe
        if r.get('ticker'):
            by_ticker[r['ticker']] = r
    payload, blobs = [], {}
    failures, typed_values = {}, 0
    for ticker, r in by_ticker.items():
        sr = split_row(r, columns)
        cols = {}
        for k, v in zip(columns, sr.typed, strict=True):
            if v is not None:
                cols[k] = _json_typed(v)
                typed_values += 1
        for k in sr.cast_failures:
            failures[k] = failures.get(k, 0) + 1
        sha = None
        if sr.blob:
            body = json.dumps(encode(sr.blob), separators=_COMPACT, allow_nan=False, ensure_ascii=False)
            sha = blob_sha(body)
            blobs.setdefault(sha, json.loads(body))
        payload.append({'ticker': ticker, 'cols': cols,
                        'extra': encode(sr.extra) if sr.extra else None,
                        'edgar_history_sha': sha})
    n_failures = sum(failures.values())
    stats = {'rows': len(payload), 'duplicate_rows': len(rows) - len(by_ticker),
             'typed_values': typed_values, 'cast_failures': n_failures,
             'cast_failures_by_key': dict(sorted(failures.items(), key=lambda kv: -kv[1])[:20]),
             'blobs': len(blobs)}
    run = {
        'risk_free_rate': _risk_free(meta.get('risk_free_rate')),
        'risk_free_rate_source': meta.get('risk_free_rate_source'),
        'source_sha256': canonical_sha256(data),
        'meta': encode({k: v for k, v in meta.items()
                        if k not in ('date', 'risk_free_rate', 'risk_free_rate_source')}),
    }
    return Load(run_date, payload, blobs, run, stats)


def _risk_free(v):
    try:
        return _json_typed(cast(v, 'double precision'))
    except CastError:
        return None


class RestTransport:
    """The ``pipeline`` RPCs over HTTPS (Supabase Data API)."""

    def __init__(self, url, service_key, session=None, throttle=None, timeout=(5, 300), retries=4):
        import requests
        self.base = url.rstrip('/') + '/rest/v1/rpc/'
        self.session = session or requests.Session()
        self.headers = {
            'apikey': service_key,
            'Authorization': f'Bearer {service_key}',
            'Content-Type': 'application/json',
            'Content-Profile': 'pipeline',     # the RPCs live outside `public`
            'Accept-Profile': 'pipeline',
        }
        self.throttle = throttle
        self.timeout = timeout
        self.retries = retries

    def call(self, fn, args, idempotent=False):
        import requests
        body = json.dumps(args, separators=_COMPACT, allow_nan=False, ensure_ascii=False).encode('utf-8')
        attempts = self.retries + 1 if idempotent else 1
        for attempt in range(attempts):
            if self.throttle:
                self.throttle()
            try:
                resp = self.session.post(self.base + fn, data=body, headers=self.headers, timeout=self.timeout)
            except requests.RequestException as e:
                if attempt + 1 >= attempts:
                    raise PublishError(f'{fn}: {e}') from e
                logger.warning('%s: %s; retrying (%d/%d)', fn, e, attempt + 1, attempts - 1)
                time.sleep(2 ** attempt)
                continue
            if resp.status_code >= 500 and attempt + 1 < attempts:
                logger.warning('%s: HTTP %s; retrying (%d/%d)', fn, resp.status_code, attempt + 1, attempts - 1)
                time.sleep(2 ** attempt)
                continue
            if resp.status_code >= 400:
                raise PublishError(f'{fn}: HTTP {resp.status_code}: {resp.text[:500]}')
            return resp.json()
        raise PublishError(f'{fn}: no attempts left')   # pragma: no cover - loop always returns/raises


class DirectTransport:
    """The same RPCs over a psycopg connection, one transaction per call."""

    def __init__(self, con):
        self.con = con

    def call(self, fn, args, idempotent=False):
        from psycopg.types.json import Jsonb
        names = ', '.join(f'{k} => %s' for k in args)
        values = [Jsonb(v) if isinstance(v, (dict, list)) else v for v in args.values()]
        try:
            with self.con.transaction():
                return self.con.execute(f'SELECT pipeline.{fn}({names})', values).fetchone()[0]
        except Exception as e:
            raise PublishError(f'{fn}: {e}') from e


def publish(load, transport, force=False, reason=None, chunk_bytes=DEFAULT_CHUNK_BYTES,
            max_cast_failure_rate=MAX_CAST_FAILURE_RATE, min_row_ratio=MIN_ROW_RATIO, pipeline_version=None):
    """Stage *load* chunk by chunk, then publish it; returns publish_run's result.

    *min_row_ratio*: refuse (without *force*) a run with fewer rows than this
    share of the previous complete run's.
    """
    if force and not reason:
        raise PublishError('force needs a reason; it is recorded in core.runs.meta')
    rate = load.stats['cast_failures'] / max(1, load.stats['typed_values'] + load.stats['cast_failures'])
    if rate > max_cast_failure_rate and not force:
        raise PublishError(f"{load.stats['cast_failures']} values ({rate:.2%}) do not fit their columns; "
                           f"top keys {load.stats['cast_failures_by_key']}. Regenerate the registry "
                           'and add a migration, or pass force with a reason.')
    load_id = str(uuid.uuid4())
    chunks = load.chunks(chunk_bytes)
    t0 = time.time()
    for i, (rows, blobs) in enumerate(chunks):
        transport.call('stage_chunk', {'p_load_id': load_id, 'p_chunk_no': i, 'p_rows': rows, 'p_blobs': blobs},
                       idempotent=True)
    staged_s = time.time() - t0
    run = dict(load.run, pipeline_version=pipeline_version)
    expect = {'n_rows': len(load.rows), 'n_chunks': len(chunks), 'force': bool(force), 'reason': reason,
              'min_row_ratio': min_row_ratio, 'client_stats': load.stats}
    result = transport.call('publish_run', {'p_load_id': load_id, 'p_run_date': load.run_date,
                                            'p_run': run, 'p_expect': expect})
    result = dict(result or {}, chunks=len(chunks), staged_s=round(staged_s, 1),
                  total_s=round(time.time() - t0, 1), load_id=load_id)
    return result
