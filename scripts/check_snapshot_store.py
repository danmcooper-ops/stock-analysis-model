# scripts/check_snapshot_store.py
"""Fail when the DuckDB snapshot store did not keep up with a run's snapshot.

``sync_snapshot_file`` never raises: the store is a derived index and a sync
failure must not block the pipeline. The cost is that a store which stops
updating says nothing -- an ingest error that repeats every night (a column
type the store cannot widen, a key DuckDB treats as a duplicate) is only a
log warning per step, and every reader quietly falls back to parsing JSON.

This checks one run date against its ``results_<date>.json``:

* the store is readable and at the current ``SCHEMA_VERSION``;
* it holds the date, with one row per distinct ticker in the snapshot;
* it was synced no earlier than the file's last rewrite -- the enrichment
  steps and rescore_and_render rewrite the snapshot and re-sync it, so a
  failed re-sync leaves the date present but holding stale rows.

Other snapshot dates missing from the store are listed but do not fail the
check: the cloud routine stages only recent history, and readers already
fall back per date.

With ``--database`` it checks the Supabase database instead (the post-publish
check after ``scripts/db_publish.py``, design/supabase-migration.md): the run is
``complete``, holds one row per distinct ticker, and its ``source_sha256`` equals
the file's, so the file has not been rewritten since it was published. The
transport comes from the environment, as for db_publish.

Usage:
    python scripts/check_snapshot_store.py --date 2026-09-11
    python scripts/check_snapshot_store.py --date 2026-09-11 --database
    python scripts/check_snapshot_store.py --date 2026-09-11 --results-dir output --db x.duckdb

Exit status: 0 the store matches the snapshot, 1 it does not, 2 the snapshot
itself could not be found or read (an unknown, not a verdict).
"""
import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.snapshot_store import (SCHEMA_VERSION, SnapshotStore, db_path_for,  # noqa: E402
                                 list_snapshot_files, load_snapshot_file)

# File mtimes and the store's ingested_at are both local wall-clock times; a
# little slack absorbs filesystem timestamp granularity.
_MTIME_SLACK_SECONDS = 2


def snapshot_path(results_dir, run_date):
    """The snapshot path for *run_date* (plain or .gz, as discovered), or None."""
    return dict(list_snapshot_files(results_dir)).get(run_date)


def check_store(results_dir, run_date, db_path=None):
    """``(problems, info)``: lists of human-readable lines. No problems = OK.

    Raises ``FileNotFoundError`` when there is no snapshot for *run_date*.
    """
    path = snapshot_path(results_dir, run_date)
    if path is None:
        raise FileNotFoundError(f'no results_{run_date}.json[.gz] in {results_dir}')
    db_path = db_path or db_path_for(results_dir)
    problems, info = [], []
    if not os.path.exists(db_path):
        return [f'no snapshot store at {db_path}'], info

    try:
        store = SnapshotStore(db_path, read_only=True)
    except Exception as e:
        return [f'snapshot store {db_path} could not be opened: {e}'], info
    with store:
        found = store.schema_version()
        if found != SCHEMA_VERSION:
            return [f'snapshot store is schema v{found}, expected v{SCHEMA_VERSION} '
                    f'(readers ignore it; run scripts/ingest_snapshots.py)'], info
        row = store._con.execute(
            "SELECT n_rows, ingested_at, "
            "(SELECT count(*) FROM results WHERE date = ?) FROM runs WHERE date = ?",
            [run_date, run_date]).fetchone()
        have = set(store.dates())

    if row is None:
        problems.append(f'store has no rows for {run_date} -- the sync after the run failed '
                        f'(see the "snapshot store sync failed" warnings in the step logs)')
    else:
        n_rows, ingested_at, n_results = row
        _, rows = load_snapshot_file(path)
        tickers = {str(r['ticker']) for r in rows if isinstance(r, dict) and r.get('ticker')}
        if n_rows != len(tickers) or n_results != len(tickers):
            problems.append(f'store holds {n_results} rows for {run_date} (runs.n_rows={n_rows}); '
                            f'the snapshot has {len(tickers)} tickers')
        modified = datetime.fromtimestamp(os.path.getmtime(path))
        if ingested_at is None or (modified - ingested_at).total_seconds() > _MTIME_SLACK_SECONDS:
            problems.append(f'store last synced {run_date} at {ingested_at}, but {os.path.basename(path)} '
                            f'was rewritten at {modified:%Y-%m-%d %H:%M:%S} -- a later re-sync failed '
                            f'and the store holds stale rows')
        if not problems:
            info.append(f'store holds {n_results} rows for {run_date}, synced {ingested_at:%Y-%m-%d %H:%M:%S}')

    missing = [d for d, _ in list_snapshot_files(results_dir) if d not in have and d != run_date]
    if missing:
        shown = ', '.join(missing[-10:]) + (' ...' if len(missing) > 10 else '')
        info.append(f'{len(missing)} other snapshot date(s) not in the store (informational): {shown}')
    return problems, info


def check_database(results_dir, run_date, transport):
    """``(problems, info)`` for the published run of *run_date*."""
    from data.db.publish import canonical_sha256
    from data.snapshot_store import read_snapshot, split_snapshot
    path = snapshot_path(results_dir, run_date)
    if path is None:
        raise OSError(f'no results_{run_date}.json[.gz] in {results_dir}')
    data = read_snapshot(path)
    tickers = {r['ticker'] for r in split_snapshot(data)[1] if isinstance(r, dict) and r.get('ticker')}
    runs = {r['run_date']: r for r in (transport.call('list_runs', {}, idempotent=True) or [])}
    run = runs.get(run_date)
    if run is None:
        return [f'the database has no run for {run_date}'], []
    problems = []
    if run.get('status') != 'complete':
        problems.append(f"run {run_date} is {run.get('status')}, not complete")
    if run.get('n_rows') != len(tickers):
        problems.append(f"database holds {run.get('n_rows')} rows for {run_date}, the snapshot {len(tickers)}")
    if run.get('source_sha256') != canonical_sha256(data):
        problems.append(f'{path} changed since it was published (source_sha256 differs); re-run db_publish')
    missing = sorted(d for d, _ in list_snapshot_files(results_dir) if d not in runs)
    info = [f"database holds {run_date}: {run.get('n_rows')} rows, status {run.get('status')}"]
    if missing:
        info.append(f"{len(missing)} other snapshot date(s) not published: {', '.join(missing[:5])}"
                    + (' ...' if len(missing) > 5 else ''))
    return problems, info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--date', required=True, help='run date YYYY-MM-DD')
    ap.add_argument('--results-dir', default='output')
    ap.add_argument('--db', default=None, help='store path (default: <results-dir>/snapshots.duckdb)')
    ap.add_argument('--database', action='store_true', help='check the Supabase database instead')
    args = ap.parse_args(argv)
    try:
        if args.database:
            from data.db.publish import PublishError, transport_from_env
            try:
                transport, closer, _ = transport_from_env(('SUPABASE_READER_URL', 'SUPABASE_DB_URL'))
            except PublishError as e:
                print(f'check_snapshot_store: {e}', file=sys.stderr)
                return 2
            try:
                problems, info = check_database(args.results_dir, args.date, transport)
            finally:
                if closer is not None:
                    closer.close()
        else:
            problems, info = check_store(args.results_dir, args.date, args.db)
    except (OSError, ValueError) as e:
        print(f'check_snapshot_store: could not read the {args.date} snapshot: {e}', file=sys.stderr)
        return 2
    for line in problems:
        print(f'PROBLEM: {line}')
    for line in info:
        ok = not problems and line.startswith(('store holds', 'database holds'))
        print(f'{"OK" if ok else "note"}: {line}')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
