#!/usr/bin/env python3
"""Gzip aged run artifacts in ``output/`` in place, verified, to cap disk use.

A daily snapshot is ~87 MiB of plain JSON and ``output/`` had reached 7.3 GB,
4.7 GB of it ``results_<date>.json`` files nobody rewrites once the next run
lands.  Every snapshot reader goes through ``data/snapshot_store`` and accepts
``results_<date>.json.gz`` transparently, so an old snapshot loses nothing by
being gzipped (~3x smaller).

What is compacted:

* ``results_<date>.json`` — all but the newest ``--keep-plain`` snapshots
  (default 3).  The newest stay plain: the pipeline, the enrich_* scripts and
  rescore_and_render rewrite the current file in place, and run.sh names it.
* ``retired/results_<date>.json`` — always.
* Dated sidecars older than ``--keep-days`` (default 30) counted back from the
  newest snapshot date, not the wall clock, so a dormant directory is not
  swept on its first run in months: ``events_<date>.json``,
  ``stock_analysis_results_<date>*.html``, ``portfolio_report_<date>.txt`` and
  ``run_<date>.log``.  Nothing reads these after their own day.

Each file is streamed into a deterministic gzip (``mtime=0``, ``filename=''``,
the same bytes ``archive_snapshot.py`` produces), read back and compared by
SHA-256 against the source, and only then is the source removed.  The source's
mtime is carried onto the ``.gz``.  If a ``.gz`` already exists it must hold
the identical bytes or the file is left alone and reported.

Dry run by default; pass ``--apply`` to change anything.

Usage::

    python scripts/compact_output.py                  # report what would change
    python scripts/compact_output.py --apply
    python scripts/compact_output.py --results-dir output --keep-plain 5 --apply
"""

import argparse
import gzip
import hashlib
import os
import re
import shutil
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.snapshot_store import list_snapshot_files  # noqa: E402

RETIRED_DIR = 'retired'
_CHUNK = 1 << 20

_SIDECAR_RE = re.compile(
    r'^(?:events|stock_analysis_results|portfolio_report|run)_'
    r'(\d{4}-\d{2}-\d{2})[^/]*\.(?:json|html|txt|log)$')


def _sha256_and_size(fh):
    h = hashlib.sha256()
    n = 0
    while True:
        chunk = fh.read(_CHUNK)
        if not chunk:
            break
        h.update(chunk)
        n += len(chunk)
    return h.hexdigest(), n


def gzip_verified(src):
    """Replace *src* with ``src + '.gz'``; return ``(raw_bytes, gz_bytes)``.

    Raises ``OSError`` when the round trip does not reproduce *src* or when a
    ``.gz`` with different content is already present; *src* is untouched then.
    """
    dest = src + '.gz'
    with open(src, 'rb') as fin:
        src_digest, raw_bytes = _sha256_and_size(fin)
    if os.path.exists(dest):
        with gzip.open(dest, 'rb') as fin:
            if _sha256_and_size(fin) != (src_digest, raw_bytes):
                raise OSError(f'{dest} already exists with different content')
    else:
        tmp = f'{dest}.tmp.{os.getpid()}'
        try:
            with open(src, 'rb') as fin, open(tmp, 'wb') as raw_out:
                with gzip.GzipFile(filename='', fileobj=raw_out, mode='wb',
                                   compresslevel=9, mtime=0) as gz_out:
                    shutil.copyfileobj(fin, gz_out, _CHUNK)
            with gzip.open(tmp, 'rb') as fin:
                if _sha256_and_size(fin) != (src_digest, raw_bytes):
                    raise OSError(f'gzip round-trip mismatch for {src}')
            st = os.stat(src)
            os.utime(tmp, (st.st_atime, st.st_mtime))
            os.replace(tmp, dest)
        except BaseException:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise
    os.remove(src)
    return raw_bytes, os.path.getsize(dest)


def plan(results_dir, keep_plain=3, keep_days=30):
    """Paths to gzip, oldest first."""
    snapshots = list_snapshot_files(results_dir)
    plain = [p for _, p in snapshots if not p.endswith('.gz')]
    todo = plain[:-keep_plain] if keep_plain > 0 else plain

    retired = os.path.join(results_dir, RETIRED_DIR)
    if os.path.isdir(retired):
        todo += [p for _, p in list_snapshot_files(retired)
                 if not p.endswith('.gz')]

    if snapshots:
        cutoff = (date.fromisoformat(snapshots[-1][0])
                  - timedelta(days=keep_days)).isoformat()
        for name in sorted(os.listdir(results_dir)):
            m = _SIDECAR_RE.match(name)
            if m and m.group(1) < cutoff:
                todo.append(os.path.join(results_dir, name))
    return todo


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--results-dir', default='output')
    ap.add_argument('--keep-plain', type=int, default=3,
                    help='newest snapshots left as plain JSON (default: 3)')
    ap.add_argument('--keep-days', type=int, default=30,
                    help='dated sidecars younger than this stay plain (default: 30)')
    ap.add_argument('--apply', action='store_true',
                    help='gzip the files (default: dry run)')
    args = ap.parse_args(argv)

    if not os.path.isdir(args.results_dir):
        print(f'[compact] not a directory: {args.results_dir}')
        return 1
    todo = plan(args.results_dir, args.keep_plain, args.keep_days)
    if not todo:
        print('[compact] nothing to compact')
        return 0

    raw_total = gz_total = 0
    failed = 0
    for path in todo:
        rel = os.path.relpath(path, args.results_dir)
        if not args.apply:
            size = os.path.getsize(path)
            raw_total += size
            print(f'[compact] would gzip {rel} ({size / 1024 ** 2:.1f} MiB)')
            continue
        try:
            raw, gz = gzip_verified(path)
        except OSError as e:
            print(f'[compact] SKIPPED {rel}: {e}')
            failed += 1
            continue
        raw_total += raw
        gz_total += gz
        print(f'[compact] {rel}: {raw / 1024 ** 2:.1f} -> {gz / 1024 ** 2:.1f} MiB')

    if args.apply:
        print(f'[compact] {len(todo) - failed} file(s): '
              f'{raw_total / 1024 ** 3:.2f} GiB -> {gz_total / 1024 ** 3:.2f} GiB'
              f'{f", {failed} skipped" if failed else ""}')
    else:
        print(f'[compact] dry run: {len(todo)} file(s), {raw_total / 1024 ** 3:.2f} GiB '
              'plain; pass --apply to gzip them')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
