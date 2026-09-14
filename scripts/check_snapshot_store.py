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

Usage:
    python scripts/check_snapshot_store.py --date 2026-09-11
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--date', required=True, help='run date YYYY-MM-DD')
    ap.add_argument('--results-dir', default='output')
    ap.add_argument('--db', default=None, help='store path (default: <results-dir>/snapshots.duckdb)')
    args = ap.parse_args(argv)
    try:
        problems, info = check_store(args.results_dir, args.date, args.db)
    except (OSError, ValueError) as e:
        print(f'check_snapshot_store: could not read the {args.date} snapshot: {e}', file=sys.stderr)
        return 2
    for line in problems:
        print(f'PROBLEM: {line}')
    for line in info:
        print(f'{"OK" if not problems and line.startswith("store holds") else "note"}: {line}')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
