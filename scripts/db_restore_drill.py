#!/usr/bin/env python3
"""Restore drill: rebuild the database from its backups and prove it (P6).

The plan's stability check 8 times three restores. Point-in-time recovery is
a dashboard operation on a Pro project (``scheduled-tasks/RECOVERY.md``).
This script covers the two rebuilds that need no Supabase feature at all:

* ``--source storage``: the canonical ``.json.gz`` per run in the private
  ``snapshots`` bucket, as listed in the live database's
  ``core.snapshot_objects`` and checked against its SHA-256;
* ``--source archive``: the git archive alone (``--archive-git``, a
  blob-less clone of ``data/snapshots``, or ``--results-dir``).

Each drill:

1. creates a scratch database (``--drill-db``) beside the live one and
   applies ``supabase/migrations`` in order;
2. publishes every snapshot in date order through the nightly path;
3. compares it with the live database (``--compare-dsn``): per run, the
   status, row count and source SHA, and a digest of every row keyed by
   ticker; the rating change points after the first restored day; and the
   latest pointers when the drill covers the live history's end.

It prints one JSON report with the timings and exits 1 on any difference.

Usage (local stack; for a hosted project, the session pooler DSNs):
    python scripts/db_restore_drill.py --source storage \\
        --server-dsn postgresql://postgres:postgres@127.0.0.1:54322/postgres
    python scripts/db_restore_drill.py --source archive --archive-git /tmp/snaparch \\
        --since 2026-09-14 --server-dsn ...

``--compare-dsn`` defaults to ``--server-dsn`` (the live database on the
same server). Storage needs ``SUPABASE_URL`` and ``SUPABASE_SERVICE_ROLE_KEY``.
The scratch database is dropped afterwards unless ``--keep``.
"""
import argparse
import glob
import gzip
import hashlib
import json
import os
import sys
import time
import types
from urllib.parse import urlparse, urlunparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIGRATIONS = os.path.join(REPO, 'supabase', 'migrations')


def with_dbname(dsn, name):
    u = urlparse(dsn)
    return urlunparse(u._replace(path='/' + name))


def create_drill_db(server_dsn, name, replace=False):
    """Create *name* on the server and apply every migration; returns seconds."""
    from data.db.connect import connect
    if not name.replace('_', '').isalnum():
        raise SystemExit(f'--drill-db must be a plain identifier, not {name!r}')
    # The scratch database is dropped and recreated: never the live one.
    live_name = (urlparse(server_dsn).path or '/postgres').lstrip('/') or 'postgres'
    if name in ('postgres', 'template0', 'template1', live_name):
        raise SystemExit(f'--drill-db {name!r} is a live or system database; pick a scratch name')
    t0 = time.time()
    admin = connect(server_dsn, autocommit=True)
    try:
        exists = admin.execute('SELECT 1 FROM pg_database WHERE datname = %s', (name,)).fetchone()
        if exists and not replace:
            raise SystemExit(f'database {name} exists; pass --replace-db to drop it first')
        if exists:
            admin.execute(f'DROP DATABASE "{name}" WITH (FORCE)')
        admin.execute(f'CREATE DATABASE "{name}"')
    finally:
        admin.close()
    con = connect(with_dbname(server_dsn, name), autocommit=True)
    try:
        for path in sorted(glob.glob(os.path.join(MIGRATIONS, '*.sql'))):
            with open(path, encoding='utf-8') as f, con.transaction():
                con.execute(f.read())
    finally:
        con.close()
    return time.time() - t0


def drop_drill_db(server_dsn, name):
    from data.db.connect import connect
    admin = connect(server_dsn, autocommit=True)
    try:
        admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
    finally:
        admin.close()


# --- sources: (date, loader) in date order ------------------------------------

def storage_snapshots(compare_con, since=None, until=None):
    """Snapshots from the Storage bucket, per the live manifest, SHA-checked."""
    from data.db.storage import storage_from_env
    client = storage_from_env()
    if client is None:
        raise SystemExit('--source storage needs SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY')
    rows = compare_con.execute(
        'SELECT run_date::text, json_path, json_sha256 FROM core.snapshot_objects ORDER BY run_date').fetchall()
    for d, path, sha in rows:
        if (since and d < since) or (until and d > until):
            continue

        def load(path=path, sha=sha, d=d):
            blob = client.download(path)
            if hashlib.sha256(blob).hexdigest() != sha:
                raise SystemExit(f'{path}: SHA-256 differs from core.snapshot_objects')
            return json.loads(gzip.decompress(blob))
        yield d, load


def archive_snapshots(archive_git=None, results_dir=None, work=None, since=None, until=None):
    """Snapshots from the git archive (or a directory), via db_backfill."""
    from data.snapshot_store import read_snapshot
    from scripts.db_backfill import iter_snapshots
    args = types.SimpleNamespace(archive_git=archive_git, results_dir=results_dir, work=work,
                                 since=since, until=until, keep_files=False)
    for d, ref, cleanup in iter_snapshots(args):
        def load(ref=ref, cleanup=cleanup):
            path = ref() if callable(ref) else ref
            try:
                return read_snapshot(path)
            finally:
                if cleanup and os.path.exists(path):
                    os.remove(path)
        yield d, load


# --- comparison ------------------------------------------------------------------

_DIGEST = """
    SELECT r.run_date::text, count(*),
           md5(string_agg(t.ticker || '=' || (to_jsonb(r) - 'ticker_id')::text, '|' ORDER BY t.ticker))
      FROM core.results r JOIN core.tickers t ON t.ticker_id = r.ticker_id
     WHERE r.run_date = ANY (%s::date[])
     GROUP BY r.run_date"""
_RUNS = """SELECT run_date::text, status::text, n_rows, source_sha256 FROM core.runs
            WHERE run_date = ANY (%s::date[])"""
_CHANGES = """
    SELECT md5(coalesce(string_agg(t.ticker || '|' || rc.run_date || '|' || rc.rating || '|' ||
                                   coalesce(rc.prev_rating, ''), ';' ORDER BY t.ticker, rc.run_date), '')),
           count(*)
      FROM core.rating_changes rc JOIN core.tickers t ON t.ticker_id = rc.ticker_id
     WHERE rc.run_date > %s AND rc.run_date <= %s"""
_LATEST = """SELECT t.ticker, l.run_date::text FROM core.latest_results l
               JOIN core.tickers t ON t.ticker_id = l.ticker_id WHERE t.ticker = ANY (%s)"""


def compare(drill, live, dates):
    """``(problems, stats)`` of the drill database against the live one."""
    problems = []
    runs = [{r[0]: r[1:] for r in c.execute(_RUNS, (dates,)).fetchall()} for c in (drill, live)]
    digests = [{r[0]: r[1:] for r in c.execute(_DIGEST, (dates,)).fetchall()} for c in (drill, live)]
    for d in dates:
        if runs[0].get(d) != runs[1].get(d):
            problems.append(f'{d}: run differs: drill {runs[0].get(d)} vs live {runs[1].get(d)}')
        if digests[0].get(d) != digests[1].get(d):
            problems.append(f'{d}: rows differ (drill {digests[0].get(d, (0,))[0]} rows, '
                            f'live {digests[1].get(d, (0,))[0]})')
    first, last = min(dates), max(dates)
    changes = [c.execute(_CHANGES, (first, last)).fetchone() for c in (drill, live)]
    if changes[0] != changes[1]:
        problems.append(f'rating change points after {first} differ: drill {changes[0][1]}, live {changes[1][1]}')
    live_last = live.execute("SELECT max(run_date)::text FROM core.runs WHERE status = 'complete'").fetchone()[0]
    latest_checked = live_last == last
    if latest_checked:                       # for the tickers the drill restored
        tickers = [r[0] for r in drill.execute('SELECT ticker FROM core.tickers').fetchall()]
        latest = [dict(c.execute(_LATEST, (tickers,)).fetchall()) for c in (drill, live)]
        diff = sorted(t for t in tickers if latest[0].get(t) != latest[1].get(t))
        if diff:
            problems.append(f'latest pointers differ for {len(diff)} ticker(s), e.g. {diff[:5]}')
    return problems, {'dates': len(dates), 'rows': sum(v[0] for v in digests[0].values()),
                      'change_points_compared': changes[0][1], 'latest_checked': latest_checked}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--source', choices=['storage', 'archive'], required=True)
    ap.add_argument('--server-dsn', required=True, help='a DSN on the server that gets the scratch database')
    ap.add_argument('--compare-dsn', help='the live database (default: --server-dsn)')
    ap.add_argument('--drill-db', default='restore_drill')
    ap.add_argument('--replace-db', action='store_true', help='drop an existing --drill-db first')
    ap.add_argument('--keep', action='store_true', help='keep the scratch database afterwards')
    ap.add_argument('--archive-git', help='--source archive: blob-less clone of data/snapshots')
    ap.add_argument('--results-dir', help='--source archive: a directory of results_<date>.json[.gz]')
    ap.add_argument('--work', default='output/drill', help='staging dir for --archive-git')
    ap.add_argument('--since')
    ap.add_argument('--until')
    a = ap.parse_args(argv)
    if a.source == 'archive' and not (a.archive_git or a.results_dir):
        ap.error('--source archive needs --archive-git or --results-dir')
    from data.db.connect import connect
    from data.db.publish import DirectTransport, build_load, publish
    live = connect(a.compare_dsn or a.server_dsn, autocommit=True)
    report = {'source': a.source, 'drill_db': a.drill_db}
    try:
        report['create_and_migrate_s'] = round(create_drill_db(a.server_dsn, a.drill_db, a.replace_db), 1)
        drill = connect(with_dbname(a.server_dsn, a.drill_db), autocommit=True)
        try:
            items = (storage_snapshots(live, a.since, a.until) if a.source == 'storage'
                     else archive_snapshots(a.archive_git, a.results_dir, a.work, a.since, a.until))
            fetch_s = publish_s = 0.0
            dates = []
            for d, load in items:
                t0 = time.time()
                data = load()
                t1 = time.time()
                publish(build_load(data, d), DirectTransport(drill), min_row_ratio=0)
                t2 = time.time()
                fetch_s, publish_s = fetch_s + t1 - t0, publish_s + t2 - t1
                dates.append(d)
                print(f'  restored {d} (fetch {t1 - t0:.1f}s, publish {t2 - t1:.1f}s)', file=sys.stderr, flush=True)
            if not dates:
                raise SystemExit('nothing to restore in the selected range')
            report.update(fetch_s=round(fetch_s, 1), publish_s=round(publish_s, 1),
                          first=dates[0], last=dates[-1])
            t0 = time.time()
            problems, stats = compare(drill, live, dates)
            report.update(compare_s=round(time.time() - t0, 1), **stats, problems=problems[:20],
                          ok=not problems)
        finally:
            drill.close()
    finally:
        live.close()
        if not a.keep:
            drop_drill_db(a.server_dsn, a.drill_db)
    report['total_s'] = round(report.get('create_and_migrate_s', 0) + report.get('fetch_s', 0)
                              + report.get('publish_s', 0) + report.get('compare_s', 0), 1)
    print(json.dumps(report, indent=2))
    return 0 if report.get('ok') else 1


if __name__ == '__main__':
    sys.exit(main())
