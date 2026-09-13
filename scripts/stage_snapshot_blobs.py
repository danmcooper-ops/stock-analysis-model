#!/usr/bin/env python3
"""Materialize the history blobs that staged archive snapshots reference.

The cloud routine stages snapshots out of a blob-less partial clone of
``data/snapshots`` with ``git show HEAD:<name>``.  An archived
``results_<date>.json.gz`` refers to its ``edgar_history`` values by hash
(``blobs/edgar_history/<sha[:2]>/<sha>.json.gz``, see
``data/snapshot_store.write_snapshot_file``), so those blob files must land in
``<dest>/blobs`` too or reading the staged snapshot fails.

``git show`` per blob would lazily fetch ~2,500 objects one round trip at a
time.  Instead: the tree listing (trees are present in the partial clone) maps
paths to object ids, the missing ids are fetched in a few batched requests —
the same ``fetch`` invocation git's own promisor fetch uses — and
``git cat-file --batch`` streams them out.  Blobs already present and intact
under *dest* are skipped, so staging consecutive days costs only the handful
of histories that changed.

Usage::

    python scripts/stage_snapshot_blobs.py --repo <snap-clone> --dest output \\
        output/results_2026-09-08.json.gz output/results_2026-09-09.json.gz
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.snapshot_store import (BLOB_DIRNAME, _read_blob_file,  # noqa: E402
                                 blob_relpath, snapshot_blob_refs)

FETCH_BATCH = 1000


def _git(repo, *args, **kw):
    return subprocess.run(['git', '-C', repo, *args], check=True, **kw)


def _tree_oids(repo, rev='HEAD'):
    """``{path: oid}`` for every file under ``blobs/`` at *rev* (trees only)."""
    out = _git(repo, 'ls-tree', '-r', rev, '--', BLOB_DIRNAME + '/',
               capture_output=True, text=True).stdout
    oids = {}
    for line in out.splitlines():
        meta, _, path = line.partition('\t')
        parts = meta.split()
        if len(parts) == 3 and parts[1] == 'blob':
            oids[path] = parts[2]
    return oids


def _cat_blobs(repo, oids):
    """``{oid: bytes}`` via one ``git cat-file --batch`` process."""
    proc = subprocess.run(['git', '-C', repo, 'cat-file', '--batch'],
                          input=''.join(o + '\n' for o in oids).encode(),
                          capture_output=True, check=True)
    buf, pos, out = proc.stdout, 0, {}
    while pos < len(buf):
        nl = buf.index(b'\n', pos)
        header = buf[pos:nl].decode().split()
        pos = nl + 1
        if len(header) != 3:
            raise OSError(f'git cat-file could not read object: {" ".join(header)}')
        oid, _, size = header
        size = int(size)
        out[oid] = buf[pos:pos + size]
        pos += size + 1                           # content + trailing newline
    return out


def stage_blobs(repo, dest, snapshots, remote='origin', rev='HEAD', log=print):
    """Write every blob *snapshots* reference into ``<dest>/blobs``.

    Returns the number of blob files written.  Raises ``OSError`` when the
    archive tree lacks a referenced blob or a fetched blob fails its hash.
    """
    wanted = {}
    for snap in snapshots:
        for key, sha in snapshot_blob_refs(snap):
            rel = blob_relpath(key, sha)
            if not os.path.exists(os.path.join(dest, BLOB_DIRNAME, rel)) or \
                    _read_blob_file(os.path.join(dest, BLOB_DIRNAME, rel), sha) is None:
                wanted[os.path.join(BLOB_DIRNAME, rel)] = sha
    if not wanted:
        return 0
    tree = _tree_oids(repo, rev)
    missing = sorted(p for p in wanted if p not in tree)
    if missing:
        raise OSError(f'{len(missing)} referenced blob(s) are not in {rev} of the '
                      f'archive, e.g. {missing[0]}')
    oids = sorted({tree[p] for p in wanted})
    for i in range(0, len(oids), FETCH_BATCH):
        # Batched: explicit object ids, noop negotiation, no refs written —
        # what git's lazy promisor fetch runs, minus one round trip per blob.
        # A full clone already has them; the fetch is then a cheap no-op.
        _git(repo, '-c', 'fetch.negotiationAlgorithm=noop', 'fetch', '-q',
             '--no-tags', '--no-write-fetch-head', '--recurse-submodules=no',
             '--filter=blob:none', remote, *oids[i:i + FETCH_BATCH])
    contents = _cat_blobs(repo, oids)
    for path, sha in wanted.items():
        fp = os.path.join(dest, path)
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        tmp = f'{fp}.stage.{os.getpid()}'
        with open(tmp, 'wb') as fh:
            fh.write(contents[tree[path]])
        if _read_blob_file(tmp, sha) is None:
            os.remove(tmp)
            raise OSError(f'blob {path} failed its hash check after fetch')
        os.replace(tmp, fp)
    log(f'[stage-blobs] {len(wanted)} blob(s) staged into {os.path.join(dest, BLOB_DIRNAME)}')
    return len(wanted)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('snapshots', nargs='+', help='staged snapshot files')
    ap.add_argument('--repo', required=True, help='the data/snapshots clone')
    ap.add_argument('--dest', required=True, help='directory holding the staged snapshots')
    ap.add_argument('--remote', default='origin')
    args = ap.parse_args(argv)
    try:
        stage_blobs(args.repo, args.dest, args.snapshots, args.remote)
    except (OSError, subprocess.CalledProcessError) as e:
        print(f'[stage-blobs] failed: {e}')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
