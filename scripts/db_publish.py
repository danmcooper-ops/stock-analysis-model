# scripts/db_publish.py
"""Publish one day's snapshot to the Supabase database.

The snapshot is staged in ~1 MB chunks (``pipeline.stage_chunk``), then
published in a single validated transaction (``pipeline.publish_run``). That
transaction replaces the date, recomputes rating change points, and moves the
latest pointers forward. See ``design/supabase-migration.md`` (P2).

Transport:
    default    HTTPS to the Data API. Needs SUPABASE_URL and
               SUPABASE_SERVICE_ROLE_KEY; this is the cloud pipeline's only path.
    --dsn      a direct Postgres connection (dev machine, CI, admin work).

Usage:
    python scripts/db_publish.py --run-date 2026-09-25            # output/results_<date>.json[.gz]
    python scripts/db_publish.py output/results_2026-09-25.json.gz
    python scripts/db_publish.py --run-date 2026-09-25 --dsn postgresql://...
    python scripts/db_publish.py --run-date 2026-09-25 --dry-run  # build and report, send nothing
    python scripts/db_publish.py --run-date 2026-09-25 --force --reason "methodology change"

Exit code: 0 published (or dry run), 1 refused or failed, 2 bad arguments or
missing configuration.
"""
import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.db.publish import (DEFAULT_CHUNK_BYTES, DirectTransport, PublishError,  # noqa: E402
                             RestTransport, build_load, publish)
from data.snapshot_store import (DEFAULT_RESULTS_DIR, list_snapshot_files, read_snapshot,  # noqa: E402
                                 snapshot_date_from_path)
from data.throttle import Throttle  # noqa: E402

logger = logging.getLogger('db_publish')


def _resolve(args):
    if args.snapshot:
        path = args.snapshot
        run_date = args.run_date or snapshot_date_from_path(path)
    else:
        found = dict(list_snapshot_files(args.results_dir))
        path = found.get(args.run_date)
        run_date = args.run_date
        if not path:
            raise SystemExit(f'no results_{args.run_date}.json[.gz] in {args.results_dir}')
    if not run_date:
        raise SystemExit(f'cannot tell the run date of {path}; pass --run-date')
    return path, run_date


def _git_version():
    try:
        import subprocess
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return subprocess.check_output(['git', '-C', root, 'rev-parse', '--short', 'HEAD'],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception as e:
        logger.debug('no git version: %s', e)
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('snapshot', nargs='?', help='snapshot file (default: --run-date in --results-dir)')
    ap.add_argument('--run-date', help='YYYY-MM-DD')
    ap.add_argument('--results-dir', default=DEFAULT_RESULTS_DIR)
    ap.add_argument('--dsn', help='direct Postgres connection instead of the Data API')
    ap.add_argument('--force', action='store_true', help='override the soft checks (needs --reason)')
    ap.add_argument('--reason', help='recorded in core.runs.meta with --force')
    ap.add_argument('--chunk-kb', type=int, default=DEFAULT_CHUNK_BYTES // 1000)
    ap.add_argument('--dry-run', action='store_true', help='build the payload and report; send nothing')
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
    if not a.snapshot and not a.run_date:
        ap.error('give a snapshot path or --run-date')
    if a.force and not a.reason:
        ap.error('--force needs --reason')

    path, run_date = _resolve(a)
    load = build_load(read_snapshot(path), run_date)
    chunks = load.chunks(a.chunk_kb * 1000)
    print(f"{run_date}: {load.stats['rows']} rows, {load.stats['blobs']} edgar blobs, "
          f"{load.stats['cast_failures']} cast failures, {len(chunks)} chunks")
    if a.dry_run:
        return 0

    try:
        if a.dsn:
            from data.db.connect import connect
            with connect(a.dsn, autocommit=True) as con:
                result = publish(load, DirectTransport(con), force=a.force, reason=a.reason,
                                 chunk_bytes=a.chunk_kb * 1000, pipeline_version=_git_version())
        else:
            url, key = os.environ.get('SUPABASE_URL'), os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
            if not url or not key:
                print('SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set (or pass --dsn)', file=sys.stderr)
                return 2
            transport = RestTransport(url, key, throttle=Throttle(0.2))
            result = publish(load, transport, force=a.force, reason=a.reason,
                             chunk_bytes=a.chunk_kb * 1000, pipeline_version=_git_version())
    except PublishError as e:
        print(f'publish refused: {e}', file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
