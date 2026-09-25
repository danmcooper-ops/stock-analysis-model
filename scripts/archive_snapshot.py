#!/usr/bin/env python3
"""Archive a daily snapshot to the ``data/snapshots`` branch, gzipped.

The nightly runbook used to ``cp`` ``output/results_<date>.json`` straight into
the snapshots worktree.  That file crossed GitHub's 100 MiB per-blob cap:
``results_2026-09-01.json`` reached 97.2 MiB and ``results_2026-08-11.json``
was rejected outright at 102.4 MB and lost — quietly, because the runbook step
was non-blocking.  Gzipping takes a ~87 MiB snapshot to ~27 MiB, which is also
what keeps the archive branch's checkout from growing ~90 MB a night.

``edgar_history`` is stored once per distinct value: the archived snapshot
carries a ``{"$blob": "<sha256>"}`` reference per row and the value lives in
``blobs/edgar_history/<sha[:2]>/<sha>.json.gz`` next to it (see
``data/snapshot_store.write_snapshot_file``).  ~99% of tickers' histories are
unchanged day over day, so a night adds ~20 small blobs rather than re-archiving
~12 MB of gzipped history; git keeps an unchanged blob once however many
snapshots reference it.

The guarantee is unchanged: the archive, with its blobs resolved and
re-serialized in the source's encoding, must reproduce the source bytes
exactly (SHA-256 and length).  A snapshot whose bytes are not the canonical
encoding of their own parse (compact, or json.dump's default separators, which
pre-2026-09 files used) is archived verbatim instead, as before.  A truncated
or corrupt archive or blob therefore fails here rather than entering the
backtest corpus.  The check parses the snapshot, so peak RSS is roughly one
parsed snapshot (~0.9 GB for a 90 MB file); the verbatim fallback stays
memory-flat.

The size guard applies to the snapshot file: GitHub's cap is per blob, and each
history blob is a few KB.

Usage::

    python scripts/archive_snapshot.py --dest <snapshots-worktree>
    python scripts/archive_snapshot.py output/results_2026-08-11.json --dest ...
    python scripts/archive_snapshot.py --dest <worktree> --audit
    python scripts/archive_snapshot.py --dest <worktree> --list-blobs paths.txt
    python scripts/archive_snapshot.py --externalize <archive-dir> [--include-plain]

Exit codes: 0 ok, 1 usage/IO error, 2 the archive breached the hard size guard.
"""

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.snapshot_store import (BLOB_DIRNAME, blob_relpath,  # noqa: E402
                                 externalize_blobs, has_blob_refs,
                                 list_snapshot_files,
                                 read_snapshot, snapshot_blob_refs,
                                 snapshot_date_from_path, write_snapshot_file)

# GitHub hard-rejects a blob over 100 MiB.  Fail well before that: the guard
# exists to surface the growth curve while there is still room to act on it.
SOFT_LIMIT_BYTES = 50 * 1024 ** 2
HARD_LIMIT_BYTES = 80 * 1024 ** 2

_CHUNK = 1 << 20

# The encodings a snapshot has been written in: write_snapshot_file's compact
# form, and json.dump's defaults (every snapshot before the compact writer).
_SEPARATORS = ((',', ':'), (', ', ': '))

ALREADY = 'already-externalized'

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


def _encoded_digest(data, seps):
    enc = json.dumps(data, separators=seps, default=str).encode('utf-8')
    return hashlib.sha256(enc).hexdigest(), len(enc)


def _source_encoding(data, digest, n_bytes):
    """The separators that re-encode *data* to the source bytes, or None."""
    for seps in _SEPARATORS:
        if _encoded_digest(data, seps) == (digest, n_bytes):
            return seps
    return None


def _tmp_path(dest):
    # Ends in .gz so write_snapshot_file gzips it; the '.tmp.' keeps it out of
    # list_snapshot_files, whose pattern is anchored on the canonical name.
    return f'{dest[:-len(".gz")]}.tmp.{os.getpid()}.gz'


def _copy_verbatim(src, tmp, src_digest, raw_bytes):
    """Stream *src* into *tmp* gzipped, byte-exact, memory-flat."""
    # mtime=0 AND filename='' so identical content yields an identical
    # blob: re-archiving a day must not churn the archive branch with a
    # no-op commit.  Without filename='' GzipFile stores the temp file's
    # name (which carries the pid) in the gzip header.
    with open(src, 'rb') as fin, open(tmp, 'wb') as raw_out:
        with gzip.GzipFile(filename='', fileobj=raw_out, mode='wb',
                           compresslevel=9, mtime=0) as gz_out:
            shutil.copyfileobj(fin, gz_out, _CHUNK)
    with gzip.open(tmp, 'rb') as fin:
        back = _sha256_and_size(fin)
    if back != (src_digest, raw_bytes):
        raise OSError(
            f'archive round-trip mismatch for {src}: '
            f'{back[1]} bytes read back vs {raw_bytes} written')


def _write_externalized(src, dest, blob_root):
    """Write *src* (plain or .gz) to *dest* in the blob-reference form.

    Returns ``(raw_bytes, stats)``; *stats* is None when the source is not a
    canonical encoding (the caller decides whether to copy it verbatim) and
    :data:`ALREADY` when it already holds references (nothing is written).
    Raises ``OSError`` when the written archive does not reproduce *src*.
    """
    opener = gzip.open if src.endswith('.gz') else open
    with opener(src, 'rb') as fh:
        raw = fh.read()
    digest, raw_bytes = hashlib.sha256(raw).hexdigest(), len(raw)
    data = json.loads(raw)
    del raw
    if has_blob_refs(data):
        return raw_bytes, ALREADY
    seps = _source_encoding(data, digest, raw_bytes)
    if seps is None:
        return raw_bytes, None
    out, stats = externalize_blobs(data, blob_root)
    del data
    tmp = _tmp_path(dest)
    try:
        write_snapshot_file(tmp, out, blob_root=blob_root)
        del out
        # blob_cache=False: every blob is re-read from disk and its hash
        # re-checked, so a corrupt or missing blob fails the archive here.
        back = read_snapshot(tmp, blob_cache=False)
        if _encoded_digest(back, seps) != (digest, raw_bytes):
            raise OSError(f'archive round-trip mismatch for {src}: the archive '
                          f'with its blobs resolved does not reproduce the source')
        del back
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return raw_bytes, stats


def archive_snapshot(src, dest_dir, hard=None, soft=None, stats_out=None):
    """Gzip *src* into *dest_dir*; return ``(dest_path, raw, gz, level)``.

    History blobs go to ``<dest_dir>/blobs``.  Pass a dict as *stats_out* to
    receive ``{'refs', 'new_blobs', 'new_bytes', 'verbatim'}``.

    Raises ``ValueError`` when *src* is not a canonical snapshot and
    ``OSError`` when the round-trip does not reproduce the source bytes.
    """
    run_date = snapshot_date_from_path(src)
    if run_date is None or src.endswith('.gz'):
        raise ValueError(f'not a canonical plain snapshot filename: {src}')
    dest = os.path.join(dest_dir, f'results_{run_date}.json.gz')
    blob_root = os.path.join(dest_dir, BLOB_DIRNAME)
    raw_bytes, stats = _write_externalized(src, dest, blob_root)
    if stats is None or stats == ALREADY:
        with open(src, 'rb') as fin:
            src_digest, raw_bytes = _sha256_and_size(fin)
        tmp = _tmp_path(dest)
        try:
            _copy_verbatim(src, tmp, src_digest, raw_bytes)
            os.replace(tmp, dest)
        except BaseException:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise
        stats = {'refs': 0, 'new_blobs': 0, 'new_bytes': 0, 'verbatim': True}
    else:
        stats['verbatim'] = False
    if stats_out is not None:
        stats_out.update(stats)
    gz_bytes = os.path.getsize(dest)
    level, _ = check_archive_size(gz_bytes, soft=soft, hard=hard)
    return dest, raw_bytes, gz_bytes, level


def blob_paths(snapshot_path, dest_dir):
    """Blob files the snapshot references, as paths relative to *dest_dir*
    (what the archive step must ``git add`` alongside the snapshot)."""
    root = os.path.join(dest_dir, BLOB_DIRNAME)
    return [os.path.relpath(os.path.join(root, blob_relpath(k, sha)), dest_dir)
            for k, sha in snapshot_blob_refs(snapshot_path)]


def externalize_dir(archive_dir, include_plain=False, remove_plain=False, log=print):
    """One-off migration: rewrite every snapshot in *archive_dir* (and its
    ``retired/``) into the blob-reference form, each verified against the
    file it replaces.  Returns ``(converted, skipped, failed)`` counts.

    ``.json.gz`` files are rewritten in place.  With *include_plain*, a plain
    ``results_<date>.json`` with no ``.gz`` sibling is archived to one; it is
    deleted afterwards only with *remove_plain*.  A file whose bytes are not a
    canonical encoding is left untouched and counted as skipped.
    """
    blob_root = os.path.join(archive_dir, BLOB_DIRNAME)
    dirs = [archive_dir]
    if os.path.isdir(os.path.join(archive_dir, 'retired')):
        dirs.append(os.path.join(archive_dir, 'retired'))
    converted = skipped = failed = 0
    for d in dirs:
        for name in sorted(os.listdir(d)):
            path = os.path.join(d, name)
            day = snapshot_date_from_path(name)
            if day is None:
                continue
            is_gz = name.endswith('.gz')
            dest = path if is_gz else f'{path}.gz'
            if not is_gz and (not include_plain or os.path.exists(dest)):
                continue
            before = os.path.getsize(path)
            try:
                _, stats = _write_externalized(path, dest, blob_root)
            except (OSError, ValueError) as e:
                failed += 1
                log(f'[externalize] FAILED {path}: {e}')
                continue
            if stats is None or stats == ALREADY:
                skipped += 1
                why = 'already externalized' if stats else 'not a canonical encoding'
                log(f'[externalize] skipped {path}: {why}')
                continue
            converted += 1
            log(f'[externalize] {os.path.relpath(dest, archive_dir)}: '
                f'{before / 1024 ** 2:.1f} -> {os.path.getsize(dest) / 1024 ** 2:.1f} MiB, '
                f'{stats["refs"]} refs, {stats["new_blobs"]} new blobs '
                f'({stats["new_bytes"] / 1024 ** 2:.1f} MiB)')
            if not is_gz and remove_plain:
                os.remove(path)
    return converted, skipped, failed


def audit(results_dir, dest_dir):
    """Dates present in *results_dir* but missing from the archive."""
    local = {d for d, _ in list_snapshot_files(results_dir)}
    archived = {d for d, _ in list_snapshot_files(dest_dir)}
    return sorted(local - archived)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('source', nargs='?',
                    help='snapshot to archive (default: newest in --results-dir)')
    ap.add_argument('--dest',
                    help='the data/snapshots worktree to write into')
    ap.add_argument('--results-dir', default='output',
                    help='where to look for the newest snapshot (default: output)')
    ap.add_argument('--audit', action='store_true',
                    help='list local snapshot dates missing from the archive '
                         'and exit without writing anything')
    ap.add_argument('--list-blobs', metavar='FILE',
                    help='write the blob paths (relative to --dest) the archived '
                         'snapshot references to FILE, one per line')
    ap.add_argument('--externalize', metavar='ARCHIVE_DIR',
                    help='one-off migration: convert every snapshot in '
                         'ARCHIVE_DIR to the blob-reference form, verified')
    ap.add_argument('--include-plain', action='store_true',
                    help='with --externalize: also archive plain .json files '
                         'that have no .gz sibling')
    ap.add_argument('--remove-plain', action='store_true',
                    help='with --include-plain: delete each plain file once its '
                         '.gz is verified')
    args = ap.parse_args(argv)

    if args.externalize:
        if not os.path.isdir(args.externalize):
            print(f'[externalize] not a directory: {args.externalize}')
            return EXIT_ERROR
        converted, skipped, failed = externalize_dir(
            args.externalize, args.include_plain, args.remove_plain)
        print(f'[externalize] converted={converted} skipped={skipped} failed={failed}')
        return EXIT_ERROR if failed else EXIT_OK

    if not args.dest or not os.path.isdir(args.dest):
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
        # Never $(date): a run that crosses midnight would name a file
        # that does not exist, and the archive would silently skip the night.
        files = [(d, p) for d, p in list_snapshot_files(args.results_dir)
                 if not p.endswith('.gz')]
        if not files:
            print(f'[archive] no snapshots found in {args.results_dir}')
            return EXIT_ERROR
        src = files[-1][1]
    if not os.path.exists(src):
        print(f'[archive] source does not exist: {src}')
        return EXIT_ERROR

    stats = {}
    try:
        dest, raw, gz, level = archive_snapshot(src, args.dest, stats_out=stats)
        if args.list_blobs:
            with open(args.list_blobs, 'w', encoding='utf-8') as fh:
                for p in blob_paths(dest, args.dest):
                    fh.write(p + '\n')
    except (OSError, ValueError) as e:
        print(f'[archive] failed: {e}')
        return EXIT_ERROR

    ratio = (raw / gz) if gz else 0.0
    form = ('verbatim (not a canonical encoding)' if stats['verbatim'] else
            f'{stats["refs"]} history refs, {stats["new_blobs"]} new blobs '
            f'({stats["new_bytes"] / 1024 ** 2:.1f} MiB)')
    print(f'[archive] {os.path.basename(src)} -> {os.path.basename(dest)}  '
          f'raw={raw / 1024 ** 2:.1f} MiB  gz={gz / 1024 ** 2:.1f} MiB  '
          f'ratio={ratio:.1f}x  {form}')
    _, message = check_archive_size(gz)
    if level == 'fail':
        print(f'[archive] FAIL: {message}')
        return EXIT_TOO_BIG
    if level == 'warn':
        print(f'[archive] WARNING: {message}')
    return EXIT_OK


if __name__ == '__main__':
    sys.exit(main())
