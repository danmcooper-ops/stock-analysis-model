# scripts/db_gen_registry.py
"""Generate the Postgres column registry from archived snapshots.

Scans every snapshot, infers one type per result-row key and writes
``data/db/columns.py`` (``COLUMNS``/``EXTRA_KEYS``) plus the typed column block
of the core migration. See ``design/supabase-migration.md`` (amendment A2):
every key whose values are always scalar and of one compatible type gets a
typed column; dicts, lists and keys with mixed types stay in ``extra jsonb``.

Type rules (``None`` is no evidence; ``"Infinity"``/``"NaN"`` strings count as
floats, as the DuckDB store already treats them):

    all bool                     -> boolean
    all int (fits int64)         -> bigint
    ints/floats (ints <= 2**53)  -> double precision
    all str                      -> text
    ever dict/list, or a mix     -> extra

A key also stays in ``extra`` when it is not a plain identifier or was not
seen in any of the newest ``--recent`` snapshots (default 10): retired keys
such as ``_gate_spread_>_5%`` keep round-tripping for old rows without
becoming dead columns.

Usage:
    # a blob-less clone of the archive branch (run.sh step 02 makes the same)
    git clone --filter=blob:none --depth 1 --no-checkout --single-branch \\
        -b data/snapshots https://github.com/danmcooper-ops/stock-analysis-model.git /tmp/snaparch
    python scripts/db_gen_registry.py --archive-git /tmp/snaparch
    python scripts/db_gen_registry.py --results-dir output        # local snapshots
    python scripts/db_gen_registry.py --check output/results_2026-09-25.json.gz

``--check`` compares one snapshot with the committed registry and exits 1 when
a key is new or no longer fits its column (a candidate for a promotion
migration). It never rewrites anything.
"""
import argparse
import gzip
import json
import os
import pprint
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.db.codec import NONFINITE_STRINGS  # noqa: E402
from data.db.schema import is_safe_ident, replace_generated_block  # noqa: E402
from data.snapshot_store import (BLOB_KEYS, DEFAULT_EXCLUDE_KEYS, list_snapshot_files,  # noqa: E402
                                 open_snapshot, split_snapshot)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLUMNS_PY = os.path.join(REPO, 'data', 'db', 'columns.py')
MIGRATIONS = os.path.join(REPO, 'supabase', 'migrations')
_SNAP_RE = re.compile(r'^results_(\d{4}-\d{2}-\d{2})\.json(\.gz)?$')
SKIP_KEYS = frozenset(DEFAULT_EXCLUDE_KEYS) | frozenset(BLOB_KEYS) | {'ticker'}


def _kind(v):
    if isinstance(v, bool):
        return 'bool'
    if isinstance(v, int):
        return 'int'
    if isinstance(v, float):
        return 'float'
    if isinstance(v, str):
        return 'float' if v in NONFINITE_STRINGS else 'str'
    return 'container'


class Observations:
    """Per-key evidence gathered across snapshots."""

    def __init__(self):
        self.kinds = {}      # key -> set of kinds
        self.max_int = {}    # key -> largest |int| seen
        self.seen = set()    # every key seen, even if always None
        self.last_seen = {}  # key -> index into dates of the newest snapshot holding it
        self.dates = []

    def add_rows(self, run_date, rows):
        self.dates.append(run_date)
        i = len(self.dates) - 1
        for r in rows:
            for k, v in r.items():
                if k in SKIP_KEYS:
                    continue
                self.seen.add(k)
                self.last_seen[k] = i
                if v is None:
                    continue
                kind = _kind(v)
                self.kinds.setdefault(k, set()).add(kind)
                if kind == 'int':
                    self.max_int[k] = max(self.max_int.get(k, 0), abs(v))


def decide(kinds, max_int=0):
    """Column type for a key's observed *kinds*, or None to keep it in extra."""
    if not kinds or 'container' in kinds:
        return None
    if kinds == {'bool'}:
        return 'boolean'
    if kinds == {'str'}:
        return 'text'
    if kinds == {'int'}:
        return 'bigint' if max_int <= 2 ** 63 - 1 else None
    if kinds <= {'int', 'float'}:
        return 'double precision' if max_int <= 2 ** 53 else None
    return None


def build_registry(obs, recent=10):
    """``(columns, extra_keys, mixed)``; *mixed* maps keys kept in extra for a
    type mix to their kinds, and retired/unsafe keys map to their reason."""
    columns, extra, mixed = {}, set(), {}
    oldest_recent = len(obs.dates) - recent
    for k in sorted(obs.seen):
        kinds = obs.kinds.get(k, set())
        t = decide(kinds, obs.max_int.get(k, 0))
        if t and not is_safe_ident(k):
            t, mixed[k] = None, 'not an identifier'
        elif t and obs.last_seen[k] < oldest_recent:
            t, mixed[k] = None, f'retired (last seen {obs.dates[obs.last_seen[k]]})'
        if t:
            columns[k] = t
        else:
            extra.add(k)
            if k not in mixed and kinds and 'container' not in kinds:
                mixed[k] = sorted(kinds)
    return columns, extra, mixed


def _iter_archive_git(repo):
    names = subprocess.check_output(['git', '-C', repo, 'ls-tree', '--name-only', 'HEAD'], text=True).split()
    by_date = {}
    for nm in sorted(names):
        m = _SNAP_RE.match(nm)
        if m and (m.group(1) not in by_date or not nm.endswith('.gz')):
            by_date[m.group(1)] = nm
    for d in sorted(by_date):
        raw = subprocess.check_output(['git', '-C', repo, 'show', f'HEAD:{by_date[d]}'])
        if by_date[d].endswith('.gz'):
            raw = gzip.decompress(raw)
        yield d, split_snapshot(json.loads(raw))[1]


def _iter_results_dir(results_dir):
    for d, path in list_snapshot_files(results_dir):
        with open_snapshot(path) as f:
            yield d, split_snapshot(json.load(f))[1]


def write_columns_py(columns, extra, obs, path=COLUMNS_PY):
    body = f'''"""Postgres column registry for ``core.results`` — generated, do not edit.

Regenerate with ``python scripts/db_gen_registry.py`` (see its docstring for
the type rules). ``COLUMNS`` maps each promoted row key to its column type, in
column order; ``EXTRA_KEYS`` are observed keys kept in ``extra jsonb`` (dicts,
lists, mixed types). A key in neither is new and also lands in ``extra``.
"""

GENERATED_FROM = {{'dates': {len(obs.dates)}, 'first': {min(obs.dates)!r}, 'last': {max(obs.dates)!r}}}

COLUMNS = {pprint.pformat(columns, width=100, sort_dicts=False)}

EXTRA_KEYS = frozenset({pprint.pformat(sorted(extra), width=100, compact=True)})
'''
    with open(path, 'w', encoding='utf-8') as f:
        f.write(body)


def core_migration_path():
    for nm in sorted(os.listdir(MIGRATIONS)):
        if nm.endswith('_core_schema.sql'):
            return os.path.join(MIGRATIONS, nm)
    raise SystemExit('no *_core_schema.sql migration found')


def check(snapshot_path):
    from data.db.columns import COLUMNS, EXTRA_KEYS
    from data.db.codec import CastError, cast
    with open_snapshot(snapshot_path) as f:
        rows = split_snapshot(json.load(f))[1]
    new, misfit = set(), {}
    for r in rows:
        for k, v in r.items():
            if k in SKIP_KEYS:
                continue
            if k not in COLUMNS and k not in EXTRA_KEYS:
                new.add(k)
            elif k in COLUMNS:
                try:
                    cast(v, COLUMNS[k])
                except CastError:
                    misfit[k] = misfit.get(k, 0) + 1
    for k in sorted(new):
        print(f'new key (lands in extra until promoted): {k}')
    for k, n in sorted(misfit.items()):
        print(f'{n} value(s) of {k} do not fit {COLUMNS[k]} (stored in extra)')
    return 1 if new or misfit else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--archive-git', help='blob-less clone of the data/snapshots branch')
    src.add_argument('--results-dir', help='directory of results_<date>.json[.gz] files')
    src.add_argument('--check', metavar='SNAPSHOT', help='compare one snapshot with the committed registry')
    ap.add_argument('--recent', type=int, default=10,
                    help='promote only keys seen in the newest N snapshots (default 10)')
    a = ap.parse_args(argv)
    if a.check:
        return check(a.check)

    obs = Observations()
    it = _iter_archive_git(a.archive_git) if a.archive_git else _iter_results_dir(a.results_dir)
    for d, rows in it:
        obs.add_rows(d, rows)
        print(f'  {d}: {len(rows)} rows', flush=True)
    if not obs.dates:
        raise SystemExit('no snapshots found')
    columns, extra, mixed = build_registry(obs, a.recent)
    mig = core_migration_path()
    with open(mig, encoding='utf-8') as f:
        sql = replace_generated_block(f.read(), columns)   # raises before anything is written
    write_columns_py(columns, extra, obs)
    with open(mig, 'w', encoding='utf-8') as f:
        f.write(sql)
    by_type = {}
    for t in columns.values():
        by_type[t] = by_type.get(t, 0) + 1
    print(f'{len(obs.dates)} snapshots ({min(obs.dates)} .. {max(obs.dates)}); '
          f'{len(columns)} typed columns {by_type}; {len(extra)} keys in extra')
    for k, why in sorted(mixed.items()):
        print(f'  -> extra: {k}: {why}')
    print(f'wrote {os.path.relpath(COLUMNS_PY, REPO)} and {os.path.relpath(mig, REPO)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
