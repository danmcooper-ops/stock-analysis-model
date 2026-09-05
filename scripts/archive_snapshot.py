#!/usr/bin/env python3
"""Archive a daily snapshot to the ``data/snapshots`` branch, gzipped.

The nightly runbook used to ``cp`` ``output/results_<date>.json`` straight into
the snapshots worktree.  That file crossed GitHub's 100 MiB per-blob cap:
``results_2026-09-01.json`` reached 97.2 MiB and ``results_2026-08-11.json``
was rejected outright at 102.4 MB and lost — quietly, because the runbook step
was non-blocking.  Gzipping takes a ~87 MiB snapshot to ~27 MiB, which is also
what keeps the archive branch's checkout from growing ~90 MB a night.

The copy is byte-exact and memory-flat: the source is streamed into the gzip
writer and never parsed, then the archive is streamed back out and compared by
SHA-256 against the source.  A truncated or corrupt archive therefore fails
here rather than entering the backtest corpus.

Usage::

    python scripts/archive_snapshot.py --dest <snapshots-worktree>
    python scripts/archive_snapshot.py output/results_2026-08-11.json --dest ...
    python scripts/archive_snapshot.py --dest <worktree> --audit

Exit codes: 0 ok, 1 usage/IO error, 2 the archive breached the hard size guard.
"""

import argparse
import gzip
import hashlib
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.snapshot_store import (list_snapshot_files,  # noqa: E402
                                 snapshot_date_from_path)

# GitHub hard-rejects a blob over 100 MiB.  Fail well before that: the guard
# exists to surface the growth curve while there is still room to act on it.
SOFT_LIMIT_BYTES = 50 * 1024 ** 2
HARD_LIMIT_BYTES = 80 * 1024 ** 2

_CHUNK = 1 << 20

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_TOO_BIG = 2


def check_archive_size(n_bytes, soft=None, hard=None):
    """``(level, message)`` for an archive of *n_bytes*.

    *level* is ``'ok'``, ``'warn'`` (past the soft limit — the archive is still
    written and pushed) or ``'fail'`` (past the hard limit — do not push).

    The limits resolve from the module constants at call time rather than as
    default arguments, so raising or lowering them (a test, a future universe
    expansion) takes effect everywhere.
    """
    soft = SOFT_LIMIT_BYTES if soft is None else soft
    hard = HARD_LIMIT_BYTES if hard is None else hard
    mb = n_bytes / 1024 ** 2
    if n_bytes > hard:
        return 'fail', (
            f'archive is {mb:.1f} MiB, over the {hard / 1024 ** 2:.0f} MiB hard '
            f'guard — do not push; GitHub rejects a blob over 100 MiB')
    if n_bytes > soft:
        return 'warn', (
            f'archive is {mb:.1f} MiB, over the {soft / 1024 ** 2:.0f} MiB soft '
            f'guard — still pushable, but the trend needs attention')
    return 'ok', f'archive is {mb:.1f} MiB'


def _sha256_and_size(fh):
    """Stream *fh* to EOF, returning ``(hexdigest, bytes_read)``."""
    h = hashlib.sha256()
    n = 0
    while True:
        chunk = fh.read(_CHUNK)
        if not chunk:
            break
        h.update(chunk)
        n += len(chunk)
    return h.hexdigest(), n


def archive_snapshot(src, dest_dir, hard=None, soft=None):
    """Gzip *src* into *dest_dir*; return ``(dest_path, raw, gz, level)``.

    Raises ``ValueError`` when *src* is not a canonical snapshot and
    ``OSError`` when the round-trip does not reproduce the source bytes.
    """
    run_date = snapshot_date_from_path(src)
    if run_date is None:
        raise ValueError(f'not a canonical snapshot filename: {src}')
    dest = os.path.join(dest_dir, f'results_{run_date}.json.gz')
    tmp = f'{dest}.tmp.{os.getpid()}'
    try:
        # mtime=0 AND filename='' so identical content yields an identical
        # blob: re-archiving a day must not churn the archive branch with a
        # no-op commit.  Without filename='' GzipFile stores the temp file's
        # name (which carries the pid) in the gzip header.
        with open(src, 'rb') as fin, open(tmp, 'wb') as raw_out:
            with gzip.GzipFile(filename='', fileobj=raw_out, mode='wb',
                               compresslevel=9, mtime=0) as gz_out:
                shutil.copyfileobj(fin, gz_out, _CHUNK)
        with open(src, 'rb') as fin:
            src_digest, raw_bytes = _sha256_and_size(fin)
        with gzip.open(tmp, 'rb') as fin:
            back_digest, back_bytes = _sha256_and_size(fin)
        if (back_digest, back_bytes) != (src_digest, raw_bytes):
            raise OSError(
                f'archive round-trip mismatch for {src}: '
                f'{back_bytes} bytes read back vs {raw_bytes} written')
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    gz_bytes = os.path.getsize(dest)
    level, _ = check_archive_size(gz_bytes, soft=soft, hard=hard)
    return dest, raw_bytes, gz_bytes, level


def audit(results_dir, dest_dir):
    """Dates present in *results_dir* but missing from the archive."""
    local = {d for d, _ in list_snapshot_files(results_dir)}
    archived = {d for d, _ in list_snapshot_files(dest_dir)}
    return sorted(local - archived)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('source', nargs='?',
                    help='snapshot to archive (default: newest in --results-dir)')
    ap.add_argument('--dest', required=True,
                    help='the data/snapshots worktree to write into')
    ap.add_argument('--results-dir', default='output',
                    help='where to look for the newest snapshot (default: output)')
    ap.add_argument('--audit', action='store_true',
                    help='list local snapshot dates missing from the archive '
                         'and exit without writing anything')
    args = ap.parse_args(argv)

    if not os.path.isdir(args.dest):
        print(f'[archive] destination is not a directory: {args.dest}')
        return EXIT_ERROR

    if args.audit:
        missing = audit(args.results_dir, args.dest)
        if missing:
            print(f'[archive] {len(missing)} snapshot(s) missing from the archive:')
            for d in missing:
                print(f'  {d}')
        else:
            print('[archive] archive holds every local snapshot')
        return EXIT_OK

    src = args.source
    if not src:
        # Never $(date): a 3-6 h run that crosses midnight would name a file
        # that does not exist, and the archive would silently skip the night.
        files = list_snapshot_files(args.results_dir)
        if not files:
            print(f'[archive] no snapshots found in {args.results_dir}')
            return EXIT_ERROR
        src = files[-1][1]
    if not os.path.exists(src):
        print(f'[archive] source does not exist: {src}')
        return EXIT_ERROR

    try:
        dest, raw, gz, level = archive_snapshot(src, args.dest)
    except (OSError, ValueError) as e:
        print(f'[archive] failed: {e}')
        return EXIT_ERROR

    ratio = (raw / gz) if gz else 0.0
    print(f'[archive] {os.path.basename(src)} -> {os.path.basename(dest)}  '
          f'raw={raw / 1024 ** 2:.1f} MiB  gz={gz / 1024 ** 2:.1f} MiB  '
          f'ratio={ratio:.1f}x')
    _, message = check_archive_size(gz)
    if level == 'fail':
        print(f'[archive] FAIL: {message}')
        return EXIT_TOO_BIG
    if level == 'warn':
        print(f'[archive] WARNING: {message}')
    return EXIT_OK


if __name__ == '__main__':
    sys.exit(main())
