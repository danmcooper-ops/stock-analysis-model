"""Per-run Parquet exports of the results rows (P4b).

Backtests and research read these instead of the database (the plan's
analytics path: Postgres is never scanned for a backtest). One file per run,
``results_<date>.parquet``:

* ``run_date``, ``ticker``, then one column per registry key
  (``data/db/columns.COLUMNS``) with its type. Doubles keep NaN and ±Inf.
* ``extra``: every other kept key, as codec-encoded JSON text (``data/db/codec``).
* ``edgar_history``: the slim projection the DuckDB store also keeps
  (``DEFAULT_PROJECTIONS``: the only sub-keys any scoring path reads), as
  codec JSON text. The full history stays in the archived snapshot.

The snapshot's top-level metadata travels in the file's key-value metadata,
so a file stands alone. Report-only keys (``DEFAULT_EXCLUDE_KEYS``) are left
out, as in the database and the DuckDB store.
"""
import hashlib
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq

from data.db.codec import decode, dumps, split_row
from data.db.columns import COLUMNS
from data.snapshot_store import BLOB_KEYS, DEFAULT_PROJECTIONS, split_snapshot

FORMAT = 'stock-analysis/results-parquet-v1'
_ARROW = {'double precision': pa.float64(), 'bigint': pa.int64(), 'boolean': pa.bool_(), 'text': pa.string()}


def parquet_path(parquet_dir, run_date):
    return os.path.join(parquet_dir, f'results_{run_date}.parquet')


def _slim_edgar(blob):
    if not blob:
        return None
    out = {}
    for key, value in blob.items():
        keep = DEFAULT_PROJECTIONS.get(key)
        if isinstance(value, dict) and keep:
            value = {k: value[k] for k in keep if k in value}
        out[key] = value
    return out


def export_snapshot(data, run_date, path, columns=COLUMNS):
    """Write the resolved snapshot *data* for *run_date* to *path* (atomic).

    Returns ``(n_rows, sha256 of the file)``.
    """
    meta, rows = split_snapshot(data)
    by_ticker = {}
    for r in rows:                                   # last one wins, as everywhere
        if isinstance(r, dict) and r.get('ticker'):
            by_ticker[r['ticker']] = r
    tickers = sorted(by_ticker)
    typed = {k: [] for k in columns}
    extra, edgar = [], []
    for t in tickers:
        sr = split_row(by_ticker[t], columns)
        for k, v in zip(columns, sr.typed, strict=True):
            typed[k].append(v)
        extra.append(dumps(sr.extra) if sr.extra else None)
        slim = _slim_edgar(sr.blob)
        edgar.append(dumps(slim[BLOB_KEYS[0]]) if slim and BLOB_KEYS[0] in slim else None)
    arrays = [pa.array([run_date] * len(tickers), pa.string()).cast(pa.date32()), pa.array(tickers, pa.string())]
    names = ['run_date', 'ticker']
    for k, t in columns.items():
        arrays.append(pa.array(typed[k], _ARROW[t]))
        names.append(k)
    arrays += [pa.array(extra, pa.string()), pa.array(edgar, pa.string())]
    names += ['extra', 'edgar_history']
    table = pa.Table.from_arrays(arrays, names=names)
    table = table.replace_schema_metadata({
        'format': FORMAT, 'run_date': run_date,
        'snapshot_meta': dumps(meta),
    })
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = f'{path}.tmp.{os.getpid()}'
    try:
        pq.write_table(table, tmp, compression='zstd')
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return len(tickers), h.hexdigest()


def read_snapshot_parquet(path):
    """The snapshot a Parquet export stands for: ``{**meta, 'results': rows}``.

    Rows carry the non-NULL typed values, the decoded ``extra`` keys and the
    slim ``edgar_history``, like the DuckDB store's rows.
    """
    table = pq.read_table(path)
    md = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()}
    if md.get('format') != FORMAT:
        raise ValueError(f'{path} is not a {FORMAT} file')
    fixed = {'run_date', 'ticker', 'extra', 'edgar_history'}
    typed_names = [n for n in table.column_names if n not in fixed]
    rows = []
    for rec in table.to_pylist():
        row = {'ticker': rec['ticker']}
        for k in typed_names:
            if rec[k] is not None:
                row[k] = rec[k]
        if rec.get('extra'):
            row.update(decode(json.loads(rec['extra'])))
        if rec.get('edgar_history'):
            row[BLOB_KEYS[0]] = decode(json.loads(rec['edgar_history']))
        rows.append(row)
    meta = decode(json.loads(md.get('snapshot_meta') or '{}'))
    meta['date'] = md.get('run_date', meta.get('date'))
    return {**meta, 'results': rows}


def export_dir(results_dir, parquet_dir, dates=None, replace=False):
    """Export every snapshot in *results_dir* (or *dates*) that has no Parquet
    file yet; returns ``[(date, n_rows)]``. Used to backfill the analytics
    corpus from the archive."""
    from data.snapshot_store import list_snapshot_files, read_snapshot
    done = []
    for d, path in list_snapshot_files(results_dir):
        if dates is not None and d not in dates:
            continue
        out = parquet_path(parquet_dir, d)
        if os.path.exists(out) and not replace:
            continue
        n, _ = export_snapshot(read_snapshot(path), d, out)
        done.append((d, n))
    return done

