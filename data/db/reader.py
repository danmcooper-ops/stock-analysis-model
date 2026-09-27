"""The snapshot-store readers backed by the Supabase database (P4).

:class:`DbStore` offers the reader half of ``data.snapshot_store.SnapshotStore``
(``dates``, ``has_date``, ``rows``, ``prior_rows``, ``last_known_rows``,
``rating_history``, ``run_meta``, ``columns``) over the ``pipeline`` read RPCs,
so the existing call sites switch without change: ``SnapshotStore.
for_results_dir()`` returns a DbStore when the database backend is selected.

Selection is explicit (plan item R12). ``SNAPSHOT_STORE_BACKEND=postgres``
must be set, plus a transport:

* ``SUPABASE_URL`` + ``SUPABASE_SERVICE_ROLE_KEY``: the Data API over HTTPS,
  the only path from the cloud container (A1);
* otherwise ``SUPABASE_READER_URL`` (or ``SUPABASE_DB_URL``): a direct
  connection, for dev machines and CI.

Every failure degrades to the callers' existing JSON fallback: opening the
store returns None, and the first failure disables the backend for the rest of
the process, so an outage costs one timeout rather than one per call site
(R10). Under pytest only a local database is accepted.

A date is served only when its run is complete and, if a plain
``results_<date>.json`` sits in the results directory, that file's SHA-256
equals the published ``source_sha256``. A file rewritten after publishing
therefore falls back to the JSON instead of serving stale rows (R2).
"""
import hashlib
import logging
import os

from data.db.codec import decode
from data.db.columns import COLUMNS, EXTRA_KEYS

logger = logging.getLogger(__name__)

BACKEND_ENV = 'SNAPSHOT_STORE_BACKEND'
_BLOB_KEYS = ('edgar_history',)
_disabled_reason = None           # set on the first failure; see open_db_store
_hash_cache = {}                  # path -> ((mtime_ns, size), sha256)


def db_backend_requested():
    return os.environ.get(BACKEND_ENV, '').strip().lower() in ('postgres', 'supabase')


def _cast(k, v):
    t = COLUMNS.get(k)
    if v is None:
        return None
    if t == 'double precision':
        return float(v)                  # a JSON number, or "NaN"/"Infinity"/"-Infinity"
    if t == 'bigint':
        return int(v)
    return v


def _file_sha256(path):
    st = os.stat(path)
    key = (st.st_mtime_ns, st.st_size)
    hit = _hash_cache.get(path)
    if hit and hit[0] == key:
        return hit[1]
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    _hash_cache[path] = (key, h.hexdigest())
    return _hash_cache[path][1]


class DbStore:
    """Read-only snapshot store over the ``pipeline`` read RPCs."""

    # Holds the whole published history, not just the files staged locally.
    # report_html's rating-history check accepts a superset of the file dates
    # from such a store (plan item R3).
    authoritative = True

    def __init__(self, transport, results_dir=None, closer=None):
        self._t = transport
        self._results_dir = results_dir
        self._closer = closer
        self._runs_cache = None

    # -- lifecycle -----------------------------------------------------------
    def close(self):
        if self._closer is not None:
            self._closer.close()
            self._closer = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _call(self, fn, **args):
        return self._t.call(fn, args, idempotent=True)

    # -- runs ----------------------------------------------------------------
    def _runs(self):
        if self._runs_cache is None:
            self._runs_cache = {r['run_date']: r for r in (self._call('list_runs') or [])}
        return self._runs_cache

    def _stale(self, run_date, run):
        if not self._results_dir:
            return False
        path = os.path.join(self._results_dir, f'results_{run_date}.json')
        if not os.path.exists(path):
            return False        # .gz archive copies are verified at publish time
        stale = _file_sha256(path) != run.get('source_sha256')
        if stale:
            logger.warning('snapshot %s changed since it was published; reading the JSON instead', path)
        return stale

    def dates(self, before=None):
        before = str(before) if before is not None else None
        return sorted(d for d, r in self._runs().items()
                      if r.get('status') == 'complete' and (before is None or d < before)
                      and not self._stale(d, r))

    def has_date(self, run_date):
        return str(run_date) in self.dates()

    def latest_date(self, before=None):
        ds = self.dates(before=before)
        return ds[-1] if ds else None

    def run_meta(self, run_date):
        """The snapshot's top-level metadata, shaped like DuckDB's run_meta."""
        r = self._call('run_meta', p_run_date=str(run_date))
        if not r or r.get('status') != 'complete':
            return None
        meta = decode(r.get('meta') or {})
        meta.update({'date': str(run_date), 'risk_free_rate': _cast_float(r.get('risk_free_rate')),
                     'risk_free_rate_source': r.get('risk_free_rate_source')})
        meta.setdefault('count', r.get('n_rows'))
        return meta

    def columns(self):
        return ['date', 'ticker', *COLUMNS, *sorted(EXTRA_KEYS), *_BLOB_KEYS]

    # -- rows ----------------------------------------------------------------
    @staticmethod
    def _request(columns):
        """``(typed columns or None, with_extra, with_blob)`` for a column list."""
        if columns is None:
            return None, True, True
        want = [c for c in columns if c not in ('date', 'ticker')]
        typed = [c for c in want if c in COLUMNS]
        with_blob = any(c in _BLOB_KEYS for c in want)
        with_extra = any(c not in COLUMNS and c not in _BLOB_KEYS for c in want)
        return typed, with_extra, with_blob

    @staticmethod
    def _rebuild(rec):
        row = {'ticker': rec['ticker']}
        for k, v in (rec.get('cols') or {}).items():
            row[k] = _cast(k, v)
        if rec.get('extra'):
            row.update(decode(rec['extra']))
        if rec.get('blob'):
            row.update(decode(rec['blob']))
        return row

    @staticmethod
    def _project(row, columns):
        if columns is None:
            return row
        out = {'ticker': row['ticker']}
        for c in columns:
            out[c] = row.get(c)
        return out

    def rows(self, run_date, columns=None):
        """Rows for *run_date*, ordered by ticker, with the requested columns
        (unknown ones as None; every stored key when *columns* is None)."""
        typed, with_extra, with_blob = self._request(columns)
        recs = self._call('read_rows', p_run_date=str(run_date), p_columns=typed,
                          p_with_extra=with_extra, p_with_blob=with_blob) or []
        return [self._project(self._rebuild(r), columns) for r in recs]

    def prior_rows(self, before, columns=None):
        d = self.latest_date(before=before)
        if d is None:
            return None, []
        return d, self.rows(d, columns)

    def last_known_rows(self, before, columns, max_lookback=7, require='rating'):
        """``(primary_date, {ticker: row}, n_fallback)`` as in the DuckDB store."""
        wanted = [c for c in columns if c not in ('date', 'ticker')]
        typed, with_extra, _ = self._request(wanted)
        if any(c in _BLOB_KEYS for c in wanted):
            raise NotImplementedError('last_known_rows does not serve edgar_history')
        res = self._call('last_known_rows', p_before=str(before), p_columns=typed,
                         p_max_lookback=max_lookback, p_require=require, p_with_extra=with_extra)
        primary = res.get('primary_date')
        if primary is None:
            return None, {}, 0
        out, n_fallback = {}, 0
        for rec in res.get('rows') or []:
            row = self._rebuild(rec)
            if rec['date'] != primary:
                n_fallback += 1
            out[rec['ticker']] = {c: row.get(c) for c in wanted}
        return primary, out, n_fallback

    def rating_history(self, before=None, column='rating'):
        """``{ticker: [[date, rating], ...]}`` change points before *before*."""
        if column != 'rating':
            raise NotImplementedError('the database keeps change points for rating only')
        out = {}
        for ticker, d, rating in self._call('rating_history', p_before=str(before) if before else None) or []:
            out.setdefault(ticker, []).append([d, rating])
        return out

    def query(self, sql, params=None):
        raise NotImplementedError('raw SQL is not available over the Data API; use the DuckDB store')


def _cast_float(v):
    return None if v is None else float(v)


def open_db_store(results_dir=None):
    """A :class:`DbStore`, or None when the backend is not selected or fails."""
    global _disabled_reason
    if not db_backend_requested() or _disabled_reason:
        return None
    closer = None
    try:
        from data.db.publish import transport_from_env
        transport, closer, _ = transport_from_env(('SUPABASE_READER_URL', 'SUPABASE_DB_URL'))
        store = DbStore(transport, results_dir, closer)
        store._runs()                                 # probe once; cached for the store's life
        return store
    except Exception as e:
        _disabled_reason = str(e) or type(e).__name__
        logger.warning('database snapshot store unavailable (%s); readers use the JSON files '
                       'for the rest of this run', _disabled_reason)
        if closer is not None:
            closer.close()
        return None


def reset_backend_state():
    """Forget a cached failure (tests, long-lived processes)."""
    global _disabled_reason
    _disabled_reason = None
    _hash_cache.clear()
