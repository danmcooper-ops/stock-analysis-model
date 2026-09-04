# scripts/ingest_snapshots.py
"""Load results_YYYY-MM-DD.json snapshots into the DuckDB snapshot store.

The store (output/snapshots.duckdb by default) is a derived index over the
JSON snapshots — see data/snapshot_store.py.  The nightly pipeline ingests
each run as it is written; this script backfills history (e.g. the
data/snapshots branch checkout) and re-syncs a date whose JSON was rewritten
by the enrichment scripts.

Usage:
    python scripts/ingest_snapshots.py                       # every new snapshot in output/
    python scripts/ingest_snapshots.py --results-dir path/to/snapshots-data
    python scripts/ingest_snapshots.py --since 2026-08-01    # only dates >= this
    python scripts/ingest_snapshots.py --replace output/results_2026-09-03.json
    python scripts/ingest_snapshots.py --db /tmp/x.duckdb --results-dir output

Idempotent: dates already present are skipped unless --replace is given.
Exit code is non-zero only when a snapshot fails to ingest.
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.snapshot_store import (SnapshotStore, db_path_for,  # noqa: E402
                                 list_snapshot_files, snapshot_date_from_path)

logger = logging.getLogger(__name__)


def ingest_dir(results_dir='output', db_path=None, replace=False, since=None,
               paths=None):
    """Ingest snapshots into the store.  Returns ``(ingested, skipped, failed)``
    as lists of date strings."""
    db_path = db_path or db_path_for(results_dir)
    if paths:
        files = []
        for p in paths:
            d = snapshot_date_from_path(p)
            if d is None:
                print(f"[ingest] skipping non-canonical snapshot: {p}")
                continue
            files.append((d, p))
        files.sort()
    else:
        files = list_snapshot_files(results_dir)
    if since:
        files = [(d, p) for d, p in files if d >= since]
    ingested, skipped, failed = [], [], []
    with SnapshotStore(db_path) as store:
        have = set(store.dates())
        for d, p in files:
            if d in have and not replace:
                skipped.append(d)
                continue
            try:
                if store.ingest_json(p, replace=replace):
                    ingested.append(d)
                    print(f"[ingest] {d}: {store.counts().get(d, 0)} rows"
                          f"{' (replaced)' if d in have else ''}")
                else:
                    skipped.append(d)
            except Exception as e:
                logger.warning("ingest failed for %s (%s): %s", d, p, e)
                print(f"[ingest] {d}: FAILED — {e}")
                failed.append(d)
    return ingested, skipped, failed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('paths', nargs='*',
                    help='specific results_YYYY-MM-DD.json files (default: all in --results-dir)')
    ap.add_argument('--results-dir', default='output')
    ap.add_argument('--db', default=None,
                    help='store path (default: <results-dir>/snapshots.duckdb)')
    ap.add_argument('--replace', action='store_true',
                    help='re-ingest dates already in the store')
    ap.add_argument('--since', default=None, help='only snapshots dated >= YYYY-MM-DD')
    args = ap.parse_args(argv)
    ingested, skipped, failed = ingest_dir(
        args.results_dir, db_path=args.db, replace=args.replace,
        since=args.since, paths=args.paths)
    print(f"[ingest] {len(ingested)} ingested, {len(skipped)} already present, "
          f"{len(failed)} failed -> {args.db or db_path_for(args.results_dir)}")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
