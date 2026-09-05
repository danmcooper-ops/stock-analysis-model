# tests/test_snapshot_io.py
"""Tests for snapshot file I/O: the compact atomic writer, the gzip-aware
reader/discovery helpers (data/snapshot_store.py) and the archive script that
puts a run on the data/snapshots branch (scripts/archive_snapshot.py).

The archive branch hit GitHub's 100 MiB per-blob cap — one snapshot was
rejected and lost — so the encoding and the size guard are both load-bearing.
"""

import gzip
import json
import os
from datetime import date

import numpy as np
import pytest

from data.snapshot_store import (
    list_snapshot_files, load_snapshot_file, read_snapshot,
    snapshot_date_from_path, write_snapshot_file,
)
from scripts.archive_snapshot import (
    EXIT_ERROR, EXIT_OK, EXIT_TOO_BIG, archive_snapshot, audit,
    check_archive_size, main,
)

SNAP = {
    'date': '2026-09-03',
    'risk_free_rate': 0.04,
    'count': 2,
    'results': [
        {'ticker': 'AAA', 'rating': 'BUY', 'price': 1.5, 'beta': None},
        {'ticker': 'BBB', 'rating': 'PASS', 'price': 2.0, 'beta': 1.1},
    ],
}


# --- the compact atomic writer ---------------------------------------------

def test_write_snapshot_file_is_compact(tmp_path):
    """indent=2 in a snapshot writer is what pushed one file past the cap."""
    p = tmp_path / 'results_2026-09-03.json'
    write_snapshot_file(str(p), SNAP)
    raw = p.read_bytes()
    assert b'\n' not in raw
    assert b', ' not in raw
    assert b'": ' not in raw
    assert read_snapshot(str(p)) == SNAP


def test_write_snapshot_file_gz_roundtrips(tmp_path):
    p = tmp_path / 'results_2026-09-03.json.gz'
    write_snapshot_file(str(p), SNAP)
    assert p.read_bytes()[:2] == b'\x1f\x8b'
    assert read_snapshot(str(p)) == SNAP
    with gzip.open(str(p), 'rt', encoding='utf-8') as fh:
        assert json.load(fh) == SNAP


def test_write_snapshot_file_gz_is_deterministic(tmp_path):
    """Same content must yield the same blob, or every re-archive churns the
    archive branch with a no-op commit."""
    a, b = tmp_path / 'a.json.gz', tmp_path / 'b.json.gz'
    write_snapshot_file(str(a), SNAP)
    write_snapshot_file(str(b), SNAP)
    assert a.read_bytes() == b.read_bytes()


def test_write_snapshot_file_is_atomic(tmp_path, monkeypatch):
    """The enrich_* chain rewrites the canonical file in place; a crash
    partway through must not truncate the run's only copy."""
    p = tmp_path / 'results_2026-09-03.json'
    write_snapshot_file(str(p), SNAP)
    before = p.read_bytes()

    def boom(*a, **kw):
        raise RuntimeError('disk full')

    monkeypatch.setattr(json, 'dump', boom)
    with pytest.raises(RuntimeError):
        write_snapshot_file(str(p), {'date': 'other'})
    assert p.read_bytes() == before
    assert [f for f in os.listdir(tmp_path) if '.tmp.' in f] == []


def test_write_snapshot_file_encodes_numpy_and_dates(tmp_path):
    """The enrich scripts previously used a bare json.dump with no default=,
    which raises on any numpy scalar."""
    p = tmp_path / 'results_2026-09-03.json'
    write_snapshot_file(str(p), {'a': np.float64(1.5), 'b': np.int64(3),
                                 'c': date(2026, 9, 3)})
    # default=str, matching what the canonical writer has always used: np.float64
    # is a float subclass so it stays a JSON number, while np.int64 is not and
    # becomes a string.  Pinned deliberately — changing this encoding would
    # shift values under a backtest corpus built for cross-run comparability.
    assert read_snapshot(str(p)) == {'a': 1.5, 'b': '3', 'c': '2026-09-03'}


# --- discovery and reading --------------------------------------------------

@pytest.mark.parametrize('name,expected', [
    ('results_2026-09-03.json', '2026-09-03'),
    ('results_2026-09-03.json.gz', '2026-09-03'),
    ('results_2026-09-03_replay.json', None),
    ('results_2026-09-03_replay.json.gz', None),
    ('results_2026-09-03.json.tmp.999', None),
    ('backtest_summary_2026-09-03.json', None),
    ('results_not-a-date.json', None),
])
def test_snapshot_date_from_path(name, expected):
    assert snapshot_date_from_path(name) == expected


def test_discovery_finds_gz_and_loads_it(tmp_path):
    write_snapshot_file(str(tmp_path / 'results_2026-09-03.json.gz'), SNAP)
    found = list_snapshot_files(str(tmp_path))
    assert [d for d, _ in found] == ['2026-09-03']
    meta, rows = load_snapshot_file(found[0][1])
    assert meta['date'] == '2026-09-03'
    assert [r['ticker'] for r in rows] == ['AAA', 'BBB']


def test_discovery_prefers_plain_json_when_both_present(tmp_path):
    """A date carried by both forms must be emitted once, as the live file:
    backtest.load_corpus does dict(list_snapshot_files(...))."""
    write_snapshot_file(str(tmp_path / 'results_2026-09-03.json'), SNAP)
    write_snapshot_file(str(tmp_path / 'results_2026-09-03.json.gz'), SNAP)
    found = list_snapshot_files(str(tmp_path))
    assert len(found) == 1
    assert found[0][1].endswith('results_2026-09-03.json')


def test_discovery_ignores_temp_and_replay_files(tmp_path):
    write_snapshot_file(str(tmp_path / 'results_2026-09-03.json'), SNAP)
    (tmp_path / 'results_2026-09-04.json.tmp.999').write_text('{}', encoding='utf-8')
    (tmp_path / 'results_2026-09-05_replay.json').write_text('{}', encoding='utf-8')
    assert [d for d, _ in list_snapshot_files(str(tmp_path))] == ['2026-09-03']


def test_mixed_directory_is_ordered_by_date(tmp_path):
    write_snapshot_file(str(tmp_path / 'results_2026-09-01.json.gz'), SNAP)
    write_snapshot_file(str(tmp_path / 'results_2026-09-02.json'), SNAP)
    write_snapshot_file(str(tmp_path / 'results_2026-09-03.json.gz'), SNAP)
    assert [d for d, _ in list_snapshot_files(str(tmp_path))] == [
        '2026-09-01', '2026-09-02', '2026-09-03']


# --- the size guard ---------------------------------------------------------

@pytest.mark.parametrize('n,level', [
    (10, 'ok'),
    (50 * 1024 ** 2, 'ok'),
    (50 * 1024 ** 2 + 1, 'warn'),
    (80 * 1024 ** 2, 'warn'),
    (80 * 1024 ** 2 + 1, 'fail'),
])
def test_check_archive_size_thresholds(n, level):
    assert check_archive_size(n)[0] == level


# --- the archive script -----------------------------------------------------

def test_archive_snapshot_writes_verified_gz(tmp_path):
    src_dir, dest = tmp_path / 'output', tmp_path / 'archive'
    src_dir.mkdir()
    dest.mkdir()
    src = src_dir / 'results_2026-09-03.json'
    write_snapshot_file(str(src), SNAP)

    path, raw, gz, level = archive_snapshot(str(src), str(dest))

    assert os.path.basename(path) == 'results_2026-09-03.json.gz'
    assert raw == src.stat().st_size
    assert gz == os.path.getsize(path)
    assert level == 'ok'
    # byte-exact: the archive must decompress to the source, not merely parse
    with gzip.open(path, 'rb') as fh:
        assert fh.read() == src.read_bytes()
    assert [f for f in os.listdir(dest) if '.tmp.' in f] == []


def test_archive_snapshot_rejects_non_canonical_name(tmp_path):
    src = tmp_path / 'results_2026-09-03_replay.json'
    write_snapshot_file(str(src), SNAP)
    dest = tmp_path / 'archive'
    dest.mkdir()
    with pytest.raises(ValueError):
        archive_snapshot(str(src), str(dest))


def test_archive_snapshot_is_deterministic(tmp_path):
    src_dir, dest = tmp_path / 'output', tmp_path / 'archive'
    src_dir.mkdir()
    dest.mkdir()
    src = src_dir / 'results_2026-09-03.json'
    write_snapshot_file(str(src), SNAP)
    first = archive_snapshot(str(src), str(dest))[0]
    blob = open(first, 'rb').read()
    archive_snapshot(str(src), str(dest))
    assert open(first, 'rb').read() == blob


def test_main_defaults_to_newest_snapshot(tmp_path, capsys):
    src_dir, dest = tmp_path / 'output', tmp_path / 'archive'
    src_dir.mkdir()
    dest.mkdir()
    write_snapshot_file(str(src_dir / 'results_2026-09-01.json'), SNAP)
    write_snapshot_file(str(src_dir / 'results_2026-09-03.json'), SNAP)

    rc = main(['--results-dir', str(src_dir), '--dest', str(dest)])

    assert rc == EXIT_OK
    assert os.path.exists(dest / 'results_2026-09-03.json.gz')
    assert not os.path.exists(dest / 'results_2026-09-01.json.gz')
    assert 'ratio=' in capsys.readouterr().out


def test_main_fails_loudly_over_the_hard_guard(tmp_path, monkeypatch, capsys):
    """A rejected push is otherwise silent — that is how 2026-08-11 was lost."""
    import scripts.archive_snapshot as mod
    src_dir, dest = tmp_path / 'output', tmp_path / 'archive'
    src_dir.mkdir()
    dest.mkdir()
    write_snapshot_file(str(src_dir / 'results_2026-09-03.json'), SNAP)
    monkeypatch.setattr(mod, 'HARD_LIMIT_BYTES', 1)

    rc = main(['--results-dir', str(src_dir), '--dest', str(dest)])

    assert rc == EXIT_TOO_BIG
    assert 'FAIL' in capsys.readouterr().out


def test_main_errors_on_missing_source_or_dest(tmp_path):
    dest = tmp_path / 'archive'
    dest.mkdir()
    assert main(['--results-dir', str(tmp_path / 'nope'),
                 '--dest', str(dest)]) == EXIT_ERROR
    assert main(['--dest', str(tmp_path / 'not-a-dir')]) == EXIT_ERROR


def test_audit_lists_dates_missing_from_the_archive(tmp_path):
    src_dir, dest = tmp_path / 'output', tmp_path / 'archive'
    src_dir.mkdir()
    dest.mkdir()
    write_snapshot_file(str(src_dir / 'results_2026-09-02.json'), SNAP)
    write_snapshot_file(str(src_dir / 'results_2026-09-03.json'), SNAP)
    archive_snapshot(str(src_dir / 'results_2026-09-03.json'), str(dest))
    assert audit(str(src_dir), str(dest)) == ['2026-09-02']
