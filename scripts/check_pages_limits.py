#!/usr/bin/env python3
"""Check a site directory against Cloudflare Pages' deploy limits.

Cloudflare Pages (free plan) refuses a deploy with any file of 25 MiB or more,
or with more than 20,000 files. ``index.html`` came within 0.5 MiB of the first
limit before ``report_html.pack_rows`` started shipping the inline ``DATA`` rows
column-wise (2026-09-28), which took it to ~9.5 MiB, ~38% of it. Its per-ticker
shard folders (``vol/``, ``px/``, ``hist/``) still grow with the universe, so
this runs before every deploy (run.sh step 08b) and warns at 80% of either
limit, when there is still time to split a file, instead of letting the deploy
fail.

Usage:
    python scripts/check_pages_limits.py docs/
    python scripts/check_pages_limits.py docs/ --max-file-mib 25 --max-files 20000

Exit code: 0 within the limits (warnings included), 1 over a limit, 2 bad
arguments.
"""
import argparse
import os
import sys

MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_FILES = 20_000
WARN_RATIO = 0.8


def scan(root):
    """Return ``[(relative path, size)]`` for every file under *root*, the way
    wrangler uploads them (dot-directories such as .git are skipped)."""
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith('.')]
        for name in filenames:
            path = os.path.join(dirpath, name)
            if os.path.isfile(path) and not os.path.islink(path):
                out.append((os.path.relpath(path, root), os.path.getsize(path)))
    return out


def check(files, max_file_bytes=MAX_FILE_BYTES, max_files=MAX_FILES, warn_ratio=WARN_RATIO):
    """Return ``(errors, warnings)`` as lists of messages."""
    errors, warnings = [], []
    for rel, size in sorted(files, key=lambda f: -f[1]):
        if size >= max_file_bytes:
            errors.append(f'{rel} is {size / 2**20:.1f} MiB (limit {max_file_bytes / 2**20:.0f} MiB)')
        elif size >= warn_ratio * max_file_bytes:
            warnings.append(f'{rel} is {size / 2**20:.1f} MiB, {size / max_file_bytes:.0%} '
                            f'of the {max_file_bytes / 2**20:.0f} MiB per-file limit')
    n = len(files)
    if n > max_files:
        errors.append(f'{n:,} files (limit {max_files:,})')
    elif n >= warn_ratio * max_files:
        warnings.append(f'{n:,} files, {n / max_files:.0%} of the {max_files:,}-file limit')
    return errors, warnings


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('root', help='the directory that will be deployed')
    ap.add_argument('--max-file-mib', type=float, default=MAX_FILE_BYTES / 2**20)
    ap.add_argument('--max-files', type=int, default=MAX_FILES)
    a = ap.parse_args(argv)
    if not os.path.isdir(a.root):
        print(f'{a.root} is not a directory', file=sys.stderr)
        return 2
    files = scan(a.root)
    errors, warnings = check(files, int(a.max_file_mib * 2**20), a.max_files)
    largest = max(files, key=lambda f: f[1], default=('(none)', 0))
    print(f'{a.root}: {len(files):,} files, {sum(s for _, s in files) / 2**20:.1f} MiB; '
          f'largest {largest[0]} ({largest[1] / 2**20:.1f} MiB)')
    for w in warnings:
        print(f'WARNING: {w}')
    for e in errors:
        print(f'ERROR: {e}', file=sys.stderr)
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
