#!/usr/bin/env python3
"""Relabel mis-dated snapshots and retire duplicates in the snapshot archive.

Fourteen archived snapshots carry a weekend or holiday date: their runs
started after a session's close and were named after the calendar day they
started on, so they hold the PREVIOUS session's prices under the wrong date.
That distorts the backtest -- a Sunday label measures forward returns from
Monday's close (``data/price_store.window_closes`` picks the nearest bar, ties
to the later one) -- and where the real session is also archived, the same
market data is counted twice.

* ``--map OLD=NEW`` rewrites a snapshot to the session it reflects: the
  top-level ``date`` becomes NEW and ``provenance.relabeled_from`` records OLD.
* ``--retire DATE`` moves a duplicate into ``retired/``, which no snapshot
  reader lists (``list_snapshot_files`` only matches top-level
  ``results_<date>.json[.gz]``).

The ARCHIVE copy (``--dest``, the ``data/snapshots`` worktree) is the source
of truth for both destinations. A local ``output/`` copy may have been
re-rendered later with newer prices (``results_2026-09-05.json`` was, on
09-07), which would put look-ahead data into the corpus.

Usage::

    python scripts/relabel_snapshots.py --dest <snapshots-worktree> \\
        --map 2026-09-05=2026-09-04 --retire 2026-07-03 --dry-run

Every write is verified by reading it back and comparing it, as JSON, with
the source (bar the relabel fields). The archive side leaves ``git add``,
commit and push to the caller.
Exit codes: 0 ok, 1 refused or failed, 2 an archive breached the size guard.
"""

import argparse
import copy
import json
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.snapshot_store import read_snapshot, write_snapshot_file  # noqa: E402
from scripts.archive_snapshot import check_archive_size  # noqa: E402
from scripts.market_open import market_status  # noqa: E402

RETIRED_DIR = 'retired'


def _existing(directory, day):
    """Paths of ``results_<day>.json`` / ``.json.gz`` present in *directory*."""
    return [p for p in (os.path.join(directory, f'results_{day}.json'),
                        os.path.join(directory, f'results_{day}.json.gz'))
            if os.path.exists(p)]


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), default=str)


def relabeled(data, old, new, reason):
    """A copy of snapshot *data* dated *new*, recording that it was *old*."""
    if not isinstance(data, dict):
        raise ValueError(f'{old}: bare-list snapshot has no date field to relabel')
    out = copy.copy(data)
    out['date'] = new
    prov = dict(out.get('provenance') or {})
    prov['relabeled_from'] = old
    prov['relabel_reason'] = reason
    out['provenance'] = prov
    return out


def _write_verified(path, data):
    write_snapshot_file(path, data)
    if _canonical(read_snapshot(path)) != _canonical(data):
        os.remove(path)
        raise OSError(f'read-back of {path} does not match what was written')


def plan(maps, retires, dest):
    """Validate the requested operations; return a list of refusal messages."""
    errors = []
    olds = [o for o, _ in maps] + list(retires)
    if len(set(olds)) != len(olds):
        errors.append('a date appears more than once across --map/--retire')
    for old, new in maps:
        if market_status(date.fromisoformat(old))[0]:
            errors.append(f'{old} is a trading day; only weekend/holiday labels are relabeled')
        if not market_status(date.fromisoformat(new))[0]:
            errors.append(f'{new} is not a trading day')
        if not _existing(dest, old):
            errors.append(f'{old}: no results_{old}.json[.gz] in {dest}')
        if _existing(dest, new):
            errors.append(f'{new}: already archived in {dest}; refusing to overwrite')
    for day in retires:
        if not _existing(dest, day):
            errors.append(f'{day}: no results_{day}.json[.gz] in {dest}')
    return errors


def apply(maps, retires, dest, results_dir=None, dry_run=False, log=print):
    """Relabel and retire; return the list of archive paths removed/added."""
    changed = {'added': [], 'removed': []}
    worst = 'ok'
    for old, new in maps:
        src = _existing(dest, old)[0]
        log(f'relabel {old} -> {new}  ({os.path.basename(src)})')
        if dry_run:
            continue
        data = relabeled(read_snapshot(src), old, new,
                         'run started after the session closed and was named '
                         'after its start day')
        target = os.path.join(dest, f'results_{new}.json.gz')
        _write_verified(target, data)
        level, msg = check_archive_size(os.path.getsize(target))
        log(f'  {os.path.basename(target)}: {msg}')
        if level == 'fail':
            worst = 'fail'
        for p in _existing(dest, old):
            os.remove(p)
            changed['removed'].append(p)
        changed['added'].append(target)
        if results_dir:
            for p in _existing(results_dir, old):
                os.remove(p)
            _write_verified(os.path.join(results_dir, f'results_{new}.json'), data)
    for day in retires:
        src = _existing(dest, day)[0]
        log(f'retire  {day}  ({os.path.basename(src)}) -> {RETIRED_DIR}/')
        if dry_run:
            continue
        data = read_snapshot(src)
        os.makedirs(os.path.join(dest, RETIRED_DIR), exist_ok=True)
        target = os.path.join(dest, RETIRED_DIR, f'results_{day}.json.gz')
        _write_verified(target, data)
        for p in _existing(dest, day):
            os.remove(p)
            changed['removed'].append(p)
        changed['added'].append(target)
        if results_dir:
            local_retired = os.path.join(results_dir, RETIRED_DIR)
            os.makedirs(local_retired, exist_ok=True)
            for p in _existing(results_dir, day):
                os.replace(p, os.path.join(local_retired, os.path.basename(p)))
    changed['worst'] = worst
    return changed


def _pair(value):
    old, sep, new = value.partition('=')
    if not sep:
        raise argparse.ArgumentTypeError(f'expected OLD=NEW, got {value!r}')
    return date.fromisoformat(old).isoformat(), date.fromisoformat(new).isoformat()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--dest', required=True, help='data/snapshots worktree (source of truth)')
    ap.add_argument('--results-dir', default=None,
                    help='also rename the local copies here (e.g. output); written from the archive copy')
    ap.add_argument('--map', dest='maps', action='append', type=_pair, default=[],
                    metavar='OLD=NEW', help='relabel a snapshot to the session it reflects')
    ap.add_argument('--retire', action='append', default=[], metavar='DATE',
                    type=lambda s: date.fromisoformat(s).isoformat(),
                    help='move a duplicate snapshot into retired/')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args(argv)
    if not args.maps and not args.retire:
        ap.error('nothing to do: give --map and/or --retire')
    errors = plan(args.maps, args.retire, args.dest)
    if errors:
        for e in errors:
            print(f'[relabel] REFUSED: {e}')
        return 1
    try:
        changed = apply(args.maps, args.retire, args.dest, args.results_dir,
                        dry_run=args.dry_run, log=lambda m: print(f'[relabel] {m}'))
    except (OSError, ValueError) as e:
        print(f'[relabel] FAILED: {e}')
        return 1
    if args.dry_run:
        print('[relabel] dry run: nothing written')
        return 0
    print(f"[relabel] {len(changed['added'])} written, {len(changed['removed'])} removed in {args.dest}")
    return 2 if changed['worst'] == 'fail' else 0


if __name__ == '__main__':
    sys.exit(main())
