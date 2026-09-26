# scripts/db_backfill.py
"""Backfill the Supabase database from the snapshot archive (P3).

Publishes every archived snapshot in ascending date order through the same
path the nightly run uses: ``stage_chunk`` followed by one ``publish_run``
transaction per date (``data/db/publish.py``). Going in date order means each
publish extends the rating change points incrementally, so no special bulk
recompute is needed.

The script is resumable and idempotent. ``pipeline.list_runs()`` reports what
is already published, and a date whose ``source_sha256`` still matches its file
is skipped. If the file changed since it was published, the date is published
again. ``--replace`` republishes every date.

Sources:
    --archive-git DIR   a blob-less clone of the data/snapshots branch (made
                        like run.sh step 02). Each snapshot and the
                        edgar_history blobs it references are staged into
                        --work one at a time; the snapshot file is removed
                        after publishing unless --keep-files is given.
    --results-dir DIR   snapshots already on disk, with their blobs/.

Transport: HTTPS to the Data API by default (SUPABASE_URL and
SUPABASE_SERVICE_ROLE_KEY), or --dsn for a direct connection.

    python scripts/db_backfill.py --archive-git /tmp/snaparch --dry-run
    python scripts/db_backfill.py --archive-git /tmp/snaparch --keep-files --work output/archive
    python scripts/db_backfill.py --results-dir output/archive --dsn postgresql://...

A date the database refuses (e.g. a row drop against the previous run) stops
the backfill unless --keep-going is given. --force-reason publishes such dates
anyway, and the reason is recorded in core.runs.meta for each one.
Exit code 1 if any date failed.
"""
import argparse
import logging
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.db.publish import (DirectTransport, PublishError, RestTransport, build_load,  # noqa: E402
                             canonical_sha256, publish)
from data.snapshot_store import list_snapshot_files, read_snapshot  # noqa: E402
from data.throttle import Throttle  # noqa: E402

logger = logging.getLogger('db_backfill')
_SNAP_RE = re.compile(r'^results_(\d{4}-\d{2}-\d{2})\.json(\.gz)?$')


def archive_dates(repo):
    """``{date: file name}`` on the archive branch, plain form preferred."""
    names = subprocess.check_output(['git', '-C', repo, 'ls-tree', '--name-only', 'HEAD'], text=True).split()
    by_date = {}
    for nm in sorted(names):
        m = _SNAP_RE.match(nm)
        if m and (m.group(1) not in by_date or not nm.endswith('.gz')):
            by_date[m.group(1)] = nm
    return by_date


def stage_from_archive(repo, name, work):
    """Write *name* from the archive into *work*, with its blobs; returns the path."""
    from scripts.stage_snapshot_blobs import stage_blobs
    os.makedirs(work, exist_ok=True)
    path = os.path.join(work, name)
    with open(path, 'wb') as fh:
        subprocess.check_call(['git', '-C', repo, 'show', f'HEAD:{name}'], stdout=fh)
    stage_blobs(repo, work, [path])
    return path


def iter_snapshots(args):
    """``(date, path, cleanup)`` in ascending date order, within --since/--until."""
    if args.archive_git:
        items = sorted(archive_dates(args.archive_git).items())
    else:
        items = list_snapshot_files(args.results_dir)
    for d, ref in items:
        if (args.since and d < args.since) or (args.until and d > args.until):
            continue
        if args.archive_git:
            existing = os.path.join(args.work, ref)
            if os.path.exists(existing):
                yield d, existing, False
            else:
                yield d, (lambda ref=ref: stage_from_archive(args.archive_git, ref, args.work)), not args.keep_files
        else:
            yield d, ref, False


def make_transport(args):
    if args.dsn:
        from data.db.connect import connect
        con = connect(args.dsn, autocommit=True)
        return DirectTransport(con), con
    url, key = os.environ.get('SUPABASE_URL'), os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
    if not url or not key:
        raise SystemExit('SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set (or pass --dsn)')
    return RestTransport(url, key, throttle=Throttle(0.2)), None


def published_runs(transport):
    """``{date: {status, source_sha256, n_rows}}`` from pipeline.list_runs()."""
    return {r['run_date']: r for r in (transport.call('list_runs', {}, idempotent=True) or [])}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--archive-git', help='blob-less clone of the data/snapshots branch')
    src.add_argument('--results-dir', help='directory of results_<date>.json[.gz] with blobs/')
    ap.add_argument('--work', default='output/archive', help='staging dir for --archive-git (default output/archive)')
    ap.add_argument('--keep-files', action='store_true', help='keep staged snapshots (e.g. for db_parity_check)')
    ap.add_argument('--dsn', help='direct Postgres connection instead of the Data API')
    ap.add_argument('--since', help='first date (YYYY-MM-DD)')
    ap.add_argument('--until', help='last date (YYYY-MM-DD)')
    ap.add_argument('--replace', action='store_true', help='republish dates that are already published')
    ap.add_argument('--force-reason', help='publish dates the soft checks refuse; recorded per date')
    ap.add_argument('--keep-going', action='store_true', help='continue past a refused date')
    ap.add_argument('--dry-run', action='store_true', help='list what would be published; send nothing')
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

    transport, con = make_transport(a)
    done = published_runs(transport)
    counts = {'published': 0, 'skipped': 0, 'failed': 0}
    t_start = time.time()
    try:
        for d, path, cleanup in iter_snapshots(a):
            if callable(path):
                if a.dry_run and d not in done:
                    print(f'{d}: would stage and publish')
                    continue
                path = path()
            data = read_snapshot(path)
            sha = canonical_sha256(data)
            prior = done.get(d)
            if prior and prior['status'] == 'complete' and prior['source_sha256'] == sha and not a.replace:
                counts['skipped'] += 1
                print(f'{d}: already published, unchanged')
            elif a.dry_run:
                print(f"{d}: would {'re' if prior else ''}publish ({'changed file' if prior else 'new'})")
            else:
                t0 = time.time()
                try:
                    load = build_load(data, d)
                    result = publish(load, transport, force=bool(a.force_reason), reason=a.force_reason)
                except PublishError as e:
                    counts['failed'] += 1
                    print(f'{d}: REFUSED {e}', file=sys.stderr)
                    if not a.keep_going:
                        break
                else:
                    counts['published'] += 1
                    warn = f" warnings={result['warnings']}" if result.get('warnings') else ''
                    print(f"{d}: {result['rows']} rows, {result['rating_changes']} rating changes, "
                          f"{result['new_blobs']} new blobs, {time.time() - t0:.1f}s{warn}", flush=True)
            if cleanup:
                os.remove(path)
    finally:
        if con is not None:
            con.close()
    print(f"\n{counts['published']} published, {counts['skipped']} unchanged, {counts['failed']} refused "
          f'in {time.time() - t_start:.0f}s')
    return 1 if counts['failed'] else 0


if __name__ == '__main__':
    sys.exit(main())
