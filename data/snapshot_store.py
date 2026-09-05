# data/snapshot_store.py
"""Embedded DuckDB store for the daily ``results_YYYY-MM-DD.json`` snapshots.

The nightly pipeline writes one ~66 MB JSON snapshot per run (~2,300 rows x
~270 keys plus a nested ``edgar_history``), and a dozen scripts each re-parse
whole files to answer small questions: yesterday's ticker set and share
counts, each ticker's last known rating, rating change-points across every
run, per-gate N/A deltas.  This module keeps the same rows in a single
DuckDB file (``output/snapshots.duckdb`` by default) so those readers can
select the handful of columns they need in milliseconds.

The JSON snapshot stays the canonical artifact (it is what the backtester
reads and what the ``data/snapshots`` git branch archives); the store is a
derived index that is rebuilt from the files by ``scripts/ingest_snapshots.py``
and kept current by the pipeline's write hook.  Every reader falls back to the
JSON files when the store is missing or does not yet hold the date it needs.

A snapshot exists in two interchangeable forms: ``results_<date>.json`` in
``output/``, and the gzipped ``results_<date>.json.gz`` that
``scripts/archive_snapshot.py`` pushes to the ``data/snapshots`` branch (plain
JSON had reached 97.2 MiB against GitHub's 100 MiB per-blob cap, and one day
was rejected and lost).  The helpers below — :func:`list_snapshot_files`,
:func:`snapshot_date_from_path`, :func:`read_snapshot`,
:func:`write_snapshot_file` — handle both, so a directory may hold a mix; the
weekly backtest reads the archive worktree directly and relies on this.

Schema (``schema_version`` = :data:`SCHEMA_VERSION`):

``runs``
    One row per snapshot date: ``date`` (PK), ``risk_free_rate``,
    ``risk_free_rate_source``, ``count``, ``meta`` (JSON: every other
    top-level key, e.g. provenance / macro_regime), ``n_rows``,
    ``source_path``, ``ingested_at``.
``results``
    One row per (``date``, ``ticker``) (PK).  Every scalar key of a result
    row becomes a column (BOOLEAN / BIGINT / DOUBLE / VARCHAR, widened as
    later snapshots require); dict and list values are stored as JSON.
    Columns are added on the fly as new keys appear, so schema drift between
    snapshot versions never blocks an ingest.  The nested ``edgar_history``
    block is stored as a slim projection (:data:`DEFAULT_PROJECTIONS`).

Identifiers are always double-quoted: row keys such as ``_composite_score``
or ``pe`` are stored verbatim.
"""

import glob
import gzip
import io
import json
import logging
import math
import os
import re
from datetime import date, datetime

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2
DB_FILENAME = 'snapshots.duckdb'
DEFAULT_RESULTS_DIR = 'output'

# json.dump's defaults put a space after every comma and colon.  On a
# ~2,500-row snapshot that is ~5 MB of nothing; indent=2 costs far more.
_COMPACT = (',', ':')

_SNAPSHOT_RE = re.compile(r'^results_(\d{4}-\d{2}-\d{2})\.json(?:\.gz)?$')

# Column-type lattice for scalar row values.  A column is widened (never
# narrowed) when a later snapshot carries a wider type; JSON absorbs anything.
_TYPE_RANK = {'BOOLEAN': 0, 'BIGINT': 1, 'DOUBLE': 2, 'VARCHAR': 3, 'JSON': 4}
_RESERVED_COLUMNS = ('date',)

# Row keys left out of the store entirely.  Nothing is dropped wholesale by
# default; pass ``exclude=('key', ...)`` to skip one.
DEFAULT_EXCLUDE_KEYS = ()

# Nested blocks stored as a slim projection rather than whole.
# ``edgar_history`` carries 54 fundamentals series per ticker and is what
# makes a snapshot ~66 MB; storing it intact would grow the store faster
# than the JSON files it indexes.  But the re-scoring path reads two things
# out of it — ``years_available`` (the thin-history rating cap in
# scripts/scoring._rating_cap_for_row) and ``operating_income_history`` (the
# pool-share CAGR signal) — so keeping exactly those makes a store row a
# drop-in for re-scoring while costing a few values per ticker.  The
# snapshot JSON remains the source for the other 52 series.
DEFAULT_PROJECTIONS = {
    'edgar_history': ('years_available', 'operating_income_history'),
}


# ---------------------------------------------------------------------------
# Snapshot-file helpers (shared by every script that discovers snapshots)
# ---------------------------------------------------------------------------

def snapshot_date_from_path(path):
    """``'YYYY-MM-DD'`` for a canonical ``results_YYYY-MM-DD.json`` path, else
    None.  Suffixed variants such as ``results_X_replay.json`` are re-scored
    copies and are rejected here so no caller mistakes them for live runs."""
    m = _SNAPSHOT_RE.match(os.path.basename(path))
    if not m:
        return None
    try:
        date.fromisoformat(m.group(1))
    except ValueError:
        return None
    return m.group(1)


def list_snapshot_files(results_dir=DEFAULT_RESULTS_DIR):
    """``[(date_str, path), ...]`` of canonical snapshots, ascending by date.

    Both the plain ``results_<date>.json`` and the gzipped archive form
    ``results_<date>.json.gz`` are discovered.  A date carried by both yields
    the plain file: that is the live working copy the pipeline rewrites, while
    the ``.gz`` is an archive copy of the same run.  A date is never emitted
    twice, so callers keyed by date (the store's ``runs`` PK, the backtest
    corpus) cannot double-count one run.
    """
    by_date = {}
    for pattern in ('results_*.json', 'results_*.json.gz'):
        for p in glob.glob(os.path.join(results_dir, pattern)):
            d = snapshot_date_from_path(p)
            if not d or (d in by_date and p.endswith('.gz')):
                continue
            by_date[d] = p
    return sorted(by_date.items())


def prior_snapshot_file(results_dir, before):
    """``(date_str, path)`` of the newest snapshot dated strictly before
    *before* (ISO string or date), or None."""
    before = _iso(before)
    prior = [(d, p) for d, p in list_snapshot_files(results_dir) if d < before]
    return prior[-1] if prior else None


def split_snapshot(data):
    """``(meta, rows)`` from a loaded snapshot; old files are a bare list."""
    if isinstance(data, dict):
        rows = data.get('results') or []
        meta = {k: v for k, v in data.items() if k != 'results'}
        return meta, rows
    return {}, list(data or [])


def open_snapshot(path):
    """Open a snapshot for text reading, transparently handling ``.json.gz``.

    The ``data/snapshots`` archive branch holds the gzipped form, and the
    weekly backtest reads that directory directly, so every reader of a
    snapshot must go through here rather than a bare ``open()``.
    """
    if path.endswith('.gz'):
        return gzip.open(path, 'rt', encoding='utf-8')
    return open(path, encoding='utf-8')


def read_snapshot(path):
    """The decoded snapshot structure at *path* (a dict, or a bare list on
    older files).  Use :func:`load_snapshot_file` for ``(meta, rows)``."""
    with open_snapshot(path) as f:
        return json.load(f)


def load_snapshot_file(path):
    """``(meta, rows)`` for a snapshot path, plain ``.json`` or ``.json.gz``."""
    return split_snapshot(read_snapshot(path))


def write_snapshot_file(path, data):
    """Write a snapshot to *path* in the canonical compact encoding.

    Every writer of a snapshot goes through here so the encoding cannot drift
    per call site.  It has drifted before: ``indent=2`` inflated the
    2026-08-03 file from 67 MB to 107 MB, past GitHub's 100 MiB blob cap,
    which is what makes a snapshot unpushable to the ``data/snapshots``
    archive branch.

    ``default=str`` matches what the pipeline's canonical writer has always
    used.  It is deliberately *not* widened to a numpy-aware encoder here:
    that would render numpy scalars as JSON numbers where every archived
    snapshot renders them as strings, silently shifting values under a
    backtest corpus whose whole purpose is cross-run comparability.

    A ``.gz`` path is written gzipped and deterministically (``mtime=0``), so
    re-archiving identical content produces an identical blob rather than
    churning the archive branch's history.

    The write is atomic: the canonical snapshot is rewritten in place by
    rescore_and_render and all four enrich_* scripts, and a crash partway
    through a direct write would truncate the run's only copy.
    """
    tmp = '%s.tmp.%d' % (path, os.getpid())
    try:
        if path.endswith('.gz'):
            # filename='' as well as mtime=0: GzipFile otherwise stores the
            # temp file's name (which carries the pid) in the gzip header,
            # so identical content would produce a different blob per run.
            with open(tmp, 'wb') as _raw:
                with gzip.GzipFile(filename='', fileobj=_raw, mode='wb',
                                   compresslevel=9, mtime=0) as _gz:
                    with io.TextIOWrapper(_gz, encoding='utf-8') as f:
                        json.dump(data, f, separators=_COMPACT, default=str)
        else:
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(data, f, separators=_COMPACT, default=str)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def db_path_for(results_dir=DEFAULT_RESULTS_DIR):
    return os.path.join(results_dir, DB_FILENAME)


def _iso(d):
    if d is None:
        return None
    if isinstance(d, (date, datetime)):
        return d.date().isoformat() if isinstance(d, datetime) else d.isoformat()
    return str(d)


def _quote(name):
    return '"' + str(name).replace('"', '""') + '"'


# ---------------------------------------------------------------------------
# Value normalisation / type inference
# ---------------------------------------------------------------------------

def _scalar(v):
    """Coerce numpy / pandas scalars to plain Python; NaN and inf become None."""
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, str)):
        return v
    if isinstance(v, float):
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(v, (dict, list, tuple)):
        return v
    item = getattr(v, 'item', None)  # numpy scalars
    if callable(item):
        try:
            return _scalar(item())
        except (ValueError, TypeError):
            pass
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    return str(v)


def _value_type(v):
    if isinstance(v, bool):
        return 'BOOLEAN'
    if isinstance(v, int):
        return 'BIGINT'
    if isinstance(v, float):
        return 'DOUBLE'
    if isinstance(v, str):
        return 'VARCHAR'
    return 'JSON'


def _widen(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return a if _TYPE_RANK[a] >= _TYPE_RANK[b] else b


def _json_default(o):
    item = getattr(o, 'item', None)
    if callable(item):
        return item()
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


def _cast(v, col_type):
    """Render a normalised Python value for a column of *col_type*."""
    if v is None:
        return None
    if col_type == 'JSON':
        return json.dumps(v, default=_json_default)
    if col_type == 'VARCHAR':
        if isinstance(v, (dict, list, tuple)):
            return json.dumps(v, default=_json_default)
        return v if isinstance(v, str) else json.dumps(v)
    if col_type == 'DOUBLE':
        return float(v)
    if col_type == 'BIGINT':
        return int(v)
    if col_type == 'BOOLEAN':
        return bool(v)
    return v


def _pa_type(col_type):
    import pyarrow as pa
    return {'BOOLEAN': pa.bool_(), 'BIGINT': pa.int64(), 'DOUBLE': pa.float64(),
            'VARCHAR': pa.string(), 'JSON': pa.string()}[col_type]


def _canonical_type(duck_type):
    """Map a DuckDB column type back onto the lattice (unknown -> VARCHAR)."""
    t = str(duck_type).upper()
    if t in _TYPE_RANK:
        return t
    if t in ('INTEGER', 'SMALLINT', 'TINYINT', 'HUGEINT', 'UBIGINT', 'UINTEGER'):
        return 'BIGINT'
    if t in ('FLOAT', 'REAL', 'DECIMAL') or t.startswith('DECIMAL'):
        return 'DOUBLE'
    if t in ('BOOL',):
        return 'BOOLEAN'
    return 'VARCHAR'


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------

class SnapshotStore:
    """Read/write access to the DuckDB snapshot index.

    Use as a context manager (``with SnapshotStore(path) as s:``) or call
    :meth:`close`.  Readers that must not create an empty file should use
    :meth:`open_existing`, which returns None when the database is absent.
    """

    def __init__(self, path=None, read_only=False):
        import duckdb
        self.path = path or db_path_for()
        self.read_only = read_only
        parent = os.path.dirname(self.path)
        if parent and not read_only:
            os.makedirs(parent, exist_ok=True)
        self._con = duckdb.connect(self.path, read_only=read_only)
        if not read_only:
            self._ensure_schema()

    @classmethod
    def open_existing(cls, path=None, read_only=True):
        """Open an existing store or return None (never creates the file)."""
        path = path or db_path_for()
        if not os.path.exists(path):
            return None
        try:
            store = cls(path, read_only=read_only)
        except Exception as e:
            logger.warning("snapshot store %s could not be opened: %s", path, e)
            return None
        # A store built under an older schema holds columns projected by
        # different rules; readers must fall back to the JSON files rather
        # than trust it. A writable open rebuilds it instead (_ensure_schema).
        found = store.schema_version()
        if found != SCHEMA_VERSION:
            logger.warning(
                "snapshot store %s is schema v%s, expected v%s — ignoring it "
                "(re-run scripts/ingest_snapshots.py to rebuild)",
                path, found, SCHEMA_VERSION)
            store.close()
            return None
        return store

    @classmethod
    def for_results_dir(cls, results_dir, read_only=True):
        return cls.open_existing(db_path_for(results_dir), read_only=read_only)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        con, self._con = self._con, None
        if con is not None:
            con.close()

    # -- schema -----------------------------------------------------------

    def schema_version(self):
        """Version the store on disk was built at, or None when unstamped."""
        try:
            row = self._con.execute(
                "SELECT max(version) FROM schema_version").fetchone()
        except Exception:
            return None
        return row[0] if row else None

    def _ensure_schema(self):
        con = self._con
        con.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER)")
        found = self.schema_version()
        if found is not None and found != SCHEMA_VERSION:
            # The store is a derived index, so an older layout is rebuilt
            # rather than migrated: the columns it holds were projected under
            # different rules and cannot be reinterpreted in place. Readers
            # fall back to the JSON files until scripts/ingest_snapshots.py
            # refills it.
            logger.warning(
                "snapshot store %s built at schema v%s, rebuilding at v%s "
                "(re-run scripts/ingest_snapshots.py to refill history)",
                self.path, found, SCHEMA_VERSION)
            con.execute("DROP TABLE IF EXISTS results")
            con.execute("DROP TABLE IF EXISTS runs")
            con.execute("DELETE FROM schema_version")
            found = None
        if found is None:
            con.execute("INSERT INTO schema_version VALUES (?)", [SCHEMA_VERSION])
        con.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                date DATE PRIMARY KEY,
                risk_free_rate DOUBLE,
                risk_free_rate_source VARCHAR,
                count INTEGER,
                n_rows INTEGER,
                meta JSON,
                source_path VARCHAR,
                ingested_at TIMESTAMP
            )""")
        con.execute("""
            CREATE TABLE IF NOT EXISTS results (
                date DATE,
                ticker VARCHAR,
                PRIMARY KEY (date, ticker)
            )""")

    def column_types(self):
        """``{column: lattice type}`` for the ``results`` table."""
        rows = self._con.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = 'results'").fetchall()
        return {name: ('DATE' if name == 'date' else _canonical_type(t))
                for name, t in rows}

    def columns(self):
        return list(self.column_types())

    # -- ingest -----------------------------------------------------------

    def ingest_json(self, path, replace=False, exclude=None, projections=None):
        """Ingest one canonical snapshot file.  Returns True when rows were
        written, False when the date was already present (and *replace* is
        False) or the path is not a canonical snapshot."""
        d = snapshot_date_from_path(path)
        if d is None:
            logger.debug("snapshot store: skipping non-canonical %s", path)
            return False
        meta, rows = load_snapshot_file(path)
        return self.ingest_rows(meta, rows, run_date=meta.get('date') or d,
                                source_path=path, replace=replace, exclude=exclude,
                                projections=projections)

    def ingest_rows(self, meta, rows, run_date=None, source_path=None,
                    replace=False, exclude=None, projections=None):
        """Ingest in-memory ``(meta, rows)``.  *run_date* defaults to
        ``meta['date']``; *exclude* names row keys to leave out (default
        :data:`DEFAULT_EXCLUDE_KEYS`) and *projections* maps a nested key to
        the sub-keys to keep (default :data:`DEFAULT_PROJECTIONS`).  Returns
        True when rows were written."""
        if self.read_only:
            raise RuntimeError("snapshot store opened read-only")
        run_date = _iso(run_date or (meta or {}).get('date'))
        if not run_date:
            raise ValueError("snapshot has no date (pass run_date=)")
        if self.has_date(run_date):
            if not replace:
                logger.debug("snapshot store: %s already ingested", run_date)
                return False
            self.delete_date(run_date)

        exclude = DEFAULT_EXCLUDE_KEYS if exclude is None else tuple(exclude)
        projections = (DEFAULT_PROJECTIONS if projections is None
                       else dict(projections))
        table, col_types = self._build_batch(run_date, rows, exclude, projections)
        self._reconcile_columns(col_types)

        con = self._con
        meta = dict(meta or {})
        extra = {k: v for k, v in meta.items()
                 if k not in ('date', 'risk_free_rate', 'risk_free_rate_source',
                              'count')}
        rf = _scalar(meta.get('risk_free_rate'))
        rf = float(rf) if isinstance(rf, (int, float)) and not isinstance(rf, bool) else None
        count = _scalar(meta.get('count'))
        count = int(count) if isinstance(count, int) and not isinstance(count, bool) else None
        con.execute("BEGIN")
        try:
            con.register('_snapshot_batch', table)
            con.execute("INSERT INTO results BY NAME SELECT * FROM _snapshot_batch")
            con.unregister('_snapshot_batch')
            con.execute(
                "INSERT INTO runs (date, risk_free_rate, risk_free_rate_source, "
                "count, n_rows, meta, source_path, ingested_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [run_date, rf, _cast(_scalar(meta.get('risk_free_rate_source')), 'VARCHAR'),
                 count, table.num_rows,
                 json.dumps(extra, default=_json_default),
                 source_path,
                 datetime.now()])
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        logger.info("snapshot store: ingested %d rows for %s", table.num_rows, run_date)
        return True

    def _build_batch(self, run_date, rows, exclude=(), projections=None):
        """Flatten *rows* into a typed pyarrow table plus ``{col: type}``."""
        import pyarrow as pa
        # Dedupe on ticker (last occurrence wins) and drop ticker-less rows.
        by_ticker = {}
        for r in rows or []:
            if not isinstance(r, dict):
                continue
            tk = r.get('ticker')
            if not tk:
                continue
            by_ticker[str(tk)] = r
        clean = []
        col_types = {}
        for tk, r in by_ticker.items():
            row = {}
            for k, v in r.items():
                if k == 'ticker' or k in exclude:
                    continue
                k = str(k)
                if projections and k in projections and isinstance(v, dict):
                    keep = projections[k]
                    v = {sk: v[sk] for sk in keep if sk in v}
                if k in _RESERVED_COLUMNS:
                    k = '_row_' + k
                v = _scalar(v)
                if v is None:
                    row[k] = None
                    col_types.setdefault(k, None)
                    continue
                row[k] = v
                col_types[k] = _widen(col_types.get(k), _value_type(v))
            row['ticker'] = tk
            clean.append(row)
        # A key that is None in every row carries no type evidence: leave it
        # out of this batch so a later snapshot can create the column with
        # its real type (readers get NULL for unknown columns either way).
        col_types = {k: t for k, t in col_types.items() if t is not None}
        # Existing column types take part in widening decisions so the batch
        # is rendered in the type the table will end up with.
        existing = self.column_types()
        for k in col_types:
            if k in existing and existing[k] != 'DATE':
                col_types[k] = _widen(col_types[k], existing[k])
        arrays = {
            'date': pa.array([date.fromisoformat(run_date)] * len(clean), pa.date32()),
            'ticker': pa.array([r['ticker'] for r in clean], pa.string()),
        }
        for k, t in col_types.items():
            arrays[k] = pa.array([_cast(r.get(k), t) for r in clean], _pa_type(t))
        return pa.table(arrays), col_types

    def _reconcile_columns(self, col_types):
        """Add missing columns and widen narrower ones (never in a txn: DuckDB
        DDL on an indexed table commits immediately)."""
        existing = self.column_types()
        for k, t in col_types.items():
            cur = existing.get(k)
            if cur is None:
                self._con.execute(f"ALTER TABLE results ADD COLUMN {_quote(k)} {t}")
            elif cur != 'DATE' and _TYPE_RANK[t] > _TYPE_RANK[cur]:
                logger.debug("snapshot store: widening %s %s -> %s", k, cur, t)
                self._con.execute(
                    f"ALTER TABLE results ALTER COLUMN {_quote(k)} SET DATA TYPE {t}")

    def delete_date(self, run_date):
        run_date = _iso(run_date)
        self._con.execute("DELETE FROM results WHERE date = ?", [run_date])
        self._con.execute("DELETE FROM runs WHERE date = ?", [run_date])

    # -- reads ------------------------------------------------------------

    def dates(self, before=None):
        sql = "SELECT date FROM runs"
        params = []
        if before is not None:
            sql += " WHERE date < ?"
            params.append(_iso(before))
        return [r[0].isoformat() for r in
                self._con.execute(sql + " ORDER BY date", params).fetchall()]

    def has_date(self, run_date):
        return self._con.execute("SELECT 1 FROM runs WHERE date = ?",
                                 [_iso(run_date)]).fetchone() is not None

    def latest_date(self, before=None):
        ds = self.dates(before=before)
        return ds[-1] if ds else None

    def run_meta(self, run_date):
        """The snapshot's top-level metadata dict (as written to JSON)."""
        row = self._con.execute(
            "SELECT risk_free_rate, risk_free_rate_source, count, meta FROM runs "
            "WHERE date = ?", [_iso(run_date)]).fetchone()
        if row is None:
            return None
        meta = json.loads(row[3]) if row[3] else {}
        meta.update({'date': _iso(run_date), 'risk_free_rate': row[0],
                     'risk_free_rate_source': row[1], 'count': row[2]})
        return meta

    def _select_list(self, columns):
        """``(select_parts, out_keys, json_keys)`` for a column request.
        Unknown columns are selected as NULL so callers get every key back."""
        types = self.column_types()
        if columns is None:
            columns = [c for c in types if c != 'date']
        parts, json_keys = [], set()
        for c in columns:
            if c in types:
                parts.append(f"{_quote(c)} AS {_quote(c)}")
                if types[c] == 'JSON':
                    json_keys.add(c)
            else:
                parts.append(f"NULL AS {_quote(c)}")
        return parts, list(columns), json_keys

    def _rows_from_cursor(self, cur, keys, json_keys):
        out = []
        for rec in cur.fetchall():
            row = dict(zip(keys, rec, strict=True))
            for k in json_keys:
                v = row.get(k)
                if isinstance(v, str):
                    try:
                        row[k] = json.loads(v)
                    except ValueError:
                        pass
            out.append(row)
        return out

    def rows(self, run_date, columns=None):
        """Rows for *run_date* as a list of dicts with the requested columns
        (all columns when None).  JSON columns come back as Python objects."""
        parts, keys, json_keys = self._select_list(columns)
        if 'ticker' not in keys:
            parts, keys = ['"ticker" AS "ticker"'] + parts, ['ticker'] + keys
        cur = self._con.execute(
            f"SELECT {', '.join(parts)} FROM results WHERE date = ? ORDER BY ticker",
            [_iso(run_date)])
        return self._rows_from_cursor(cur, keys, json_keys)

    def prior_rows(self, before, columns=None):
        """``(date_str, rows)`` for the newest snapshot strictly before
        *before*, or ``(None, [])``."""
        d = self.latest_date(before=before)
        if d is None:
            return None, []
        return d, self.rows(d, columns)

    def last_known_rows(self, before, columns, max_lookback=7,
                        require='rating'):
        """Each ticker's most recent row among the *max_lookback* snapshots
        strictly before *before*, restricted to rows whose *require* column
        is non-null / non-empty.

        Returns ``(primary_date, {ticker: row}, n_fallback)`` where
        ``primary_date`` is the newest prior snapshot and ``n_fallback`` counts
        tickers whose row came from an older snapshot (they were missing from
        the primary one).  Mirrors ``report_html._load_prev_ratings``.
        """
        before = _iso(before)
        dates = self.dates(before=before)[-max_lookback:]
        if not dates:
            return None, {}, 0
        primary = dates[-1]
        parts, keys, json_keys = self._select_list(
            [c for c in columns if c not in ('date', 'ticker')])
        sel = ', '.join(['"date"', '"ticker"'] + parts)
        types = self.column_types()
        where = "date >= ? AND date < ?"
        if require and require in types:
            where += f" AND {_quote(require)} IS NOT NULL"
            if types[require] == 'VARCHAR':
                where += f" AND {_quote(require)} <> ''"
        cur = self._con.execute(f"""
            SELECT {sel} FROM (
                SELECT *, row_number() OVER (PARTITION BY ticker ORDER BY date DESC) AS _rn
                FROM results WHERE {where}
            ) WHERE _rn = 1 ORDER BY ticker""", [dates[0], before])
        out, n_fallback = {}, 0
        for row in self._rows_from_cursor(cur, ['date', 'ticker'] + keys, json_keys):
            d, tk = row.pop('date'), row.pop('ticker')
            if _iso(d) != primary:
                n_fallback += 1
            out[tk] = row
        return primary, out, n_fallback

    def rating_history(self, before=None, column='rating'):
        """``{ticker: [[date, rating], ...]}`` change-points (first observation
        plus every transition) ascending by date, over snapshots strictly
        before *before* (all snapshots when None)."""
        if column not in self.column_types():
            return {}
        col = _quote(column)
        where = f"{col} IS NOT NULL AND {col} <> ''"
        params = []
        if before is not None:
            where += " AND date < ?"
            params.append(_iso(before))
        cur = self._con.execute(f"""
            SELECT ticker, date, {col} FROM (
                SELECT ticker, date, {col},
                       lag({col}) OVER (PARTITION BY ticker ORDER BY date) AS _prev
                FROM results WHERE {where}
            ) WHERE _prev IS NULL OR _prev <> {col}
            ORDER BY ticker, date""", params)
        out = {}
        for tk, d, rt in cur.fetchall():
            out.setdefault(tk, []).append([d.isoformat(), rt])
        return out

    def query(self, sql, params=None):
        """Run arbitrary SQL against the store; returns a pandas DataFrame."""
        return self._con.execute(sql, params or []).df()

    def counts(self):
        """``{date: n_rows}`` for every ingested snapshot."""
        return {r[0].isoformat(): r[1] for r in self._con.execute(
            "SELECT date, n_rows FROM runs ORDER BY date").fetchall()}


def sync_snapshot_file(path, data=None, db_path=None):
    """Mirror a just-written snapshot file into the store (replacing the
    date).  Pass *data* (the full snapshot structure: a dict with
    ``results`` or a bare row list) when it is already in memory to skip
    re-parsing the file.  Never raises: the store is a derived index and a
    failure here must not block the pipeline.  Returns True on success."""
    run_date = snapshot_date_from_path(path)
    if run_date is None:
        logger.debug("snapshot store: not syncing non-canonical %s", path)
        return False
    db_path = db_path or db_path_for(os.path.dirname(path) or DEFAULT_RESULTS_DIR)
    try:
        meta, rows = split_snapshot(data) if data is not None else load_snapshot_file(path)
        with SnapshotStore(db_path) as store:
            store.ingest_rows(meta, rows, run_date=run_date, source_path=path,
                              replace=True)
        return True
    except Exception as e:
        logger.warning("snapshot store sync failed for %s (%s): %s", run_date, db_path, e)
        return False
