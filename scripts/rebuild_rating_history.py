#!/usr/bin/env python3
"""Rebuild the report's rating-history cache over a whole snapshot archive.

``run.sh`` step 02 runs this when the ``data/snapshots`` branch carries no
``rating_history.json``: the cloud container stages only the newest
``SNAPSHOT_HISTORY`` snapshots, so the render alone would rebuild the cache
over those few days.

The snapshots are scanned one at a time, oldest first, in a scratch
directory that holds nothing else. That isolation is the point. The inline
rebuild this replaced ran in ``output/``, where step 02 had already staged
the newest snapshots, so its first call scanned the oldest day *and* every
staged day, moved ``last_scanned`` to the newest, and skipped every older
day after it as already seen. The 2026-09-14 run built the cache that way
(2026-04-20, then nothing until the first staged day, 08-26), and the gap
surfaced only on 09-28, as 1,107 parity mismatches against the database
(scripts/db_night_check.py).

After every snapshot the cache's ``last_scanned`` must equal that snapshot's
date, or the rebuild fails rather than leave a hole: an unreadable day stops
the scan (report_html._advance_rating_history_cache), and nothing would say
the days after it were never folded in.

    python scripts/rebuild_rating_history.py <snapshots-clone> <output-dir>

Exit code 1 on a hole or an unreadable archive.
"""
import argparse
import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.report_html import _advance_rating_history_cache  # noqa: E402
from scripts.stage_snapshot_blobs import stage_blobs  # noqa: E402

CACHE_NAME = 'rating_history.json'
_SNAPSHOT_RE = re.compile(r'results_(\d{4}-\d{2}-\d{2})\.json(\.gz)?')


class RebuildError(RuntimeError):
    """The archive could not be folded in without leaving a hole."""


def archive_snapshots(names):
    """``{date: name}`` for the top-level ``results_<date>.json[.gz]`` names;
    the plain form wins over ``.gz``, as in ``list_snapshot_files``."""
    by_date = {}
    for nm in sorted(names):
        m = _SNAPSHOT_RE.fullmatch(nm)
        if m and (m.group(1) not in by_date or not nm.endswith('.gz')):
            by_date[m.group(1)] = nm
    return by_date


def _archive_names(snap, rev):
    return subprocess.check_output(['git', '-C', snap, 'ls-tree', '--name-only', rev],
                                   text=True).split()


def _last_scanned(path):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f).get('last_scanned')
    except (OSError, ValueError, AttributeError):
        return None


def rebuild(snap, out, rev='HEAD', log=print):
    """Write ``out/rating_history.json`` from every snapshot in *snap* (a
    clone of the archive branch, blob-less is fine) at *rev*.

    Returns ``(snapshots, tickers)``. Raises :class:`RebuildError` if any
    snapshot was not folded in; ``out`` is then left untouched.
    """
    by_date = archive_snapshots(_archive_names(snap, rev))
    if not by_date:
        raise RebuildError(f'no results_<date>.json[.gz] files at {rev} of {snap}')
    os.makedirs(os.path.join(out, 'blobs'), exist_ok=True)
    work = tempfile.mkdtemp(prefix='rating-history-rebuild-')
    try:
        # History blobs land in output/blobs, where the staged snapshots
        # already brought most of them, instead of being fetched twice.
        os.symlink(os.path.abspath(os.path.join(out, 'blobs')), os.path.join(work, 'blobs'))
        cache = os.path.join(work, CACHE_NAME)
        hist = {}
        for d in sorted(by_date):
            nm = by_date[d]
            dest = os.path.join(work, nm)
            with open(dest, 'wb') as fh:
                subprocess.run(['git', '-C', snap, 'show', f'{rev}:{nm}'], stdout=fh, check=True)
            said = io.StringIO()
            try:
                stage_blobs(snap, work, [dest], rev=rev, log=lambda *_: None)
                with contextlib.redirect_stdout(said):
                    hist = _advance_rating_history_cache(work, [(d, dest)], None, CACHE_NAME)
            finally:
                os.remove(dest)
            got = _last_scanned(cache)
            if got != d:
                raise RebuildError(f'{nm} was not folded in (cache stops at {got}); '
                                   f'refusing to write a cache with a hole. '
                                   + said.getvalue().strip())
        tmp = os.path.join(out, f'.{CACHE_NAME}.tmp')
        shutil.copyfile(cache, tmp)
        os.replace(tmp, os.path.join(out, CACHE_NAME))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return len(by_date), len(hist)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('snap', help='clone of the data/snapshots branch')
    ap.add_argument('out', help='results directory to write rating_history.json into')
    ap.add_argument('--rev', default='HEAD')
    a = ap.parse_args(argv)
    t0 = time.time()
    try:
        n, tickers = rebuild(a.snap, a.out, a.rev)
    except (RebuildError, OSError, ValueError, subprocess.CalledProcessError) as e:
        print(f'rating-history rebuild failed: {e}', file=sys.stderr)
        return 1
    print(f'rating history rebuilt over {n} snapshots, {tickers} tickers, {time.time() - t0:.0f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
