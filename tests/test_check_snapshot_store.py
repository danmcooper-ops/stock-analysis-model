# tests/test_check_snapshot_store.py
"""Tests for scripts/check_snapshot_store.py: a store that stops keeping up
with the snapshots must fail the check, not pass silently."""

import json
import os
import time

import pytest

from data.snapshot_store import SnapshotStore, db_path_for, sync_snapshot_file
from scripts.check_snapshot_store import check_store, main


def _write(d, day, tickers):
    p = d / f'results_{day}.json'
    p.write_text(json.dumps({'date': day, 'results': [
        {'ticker': t, 'rating': 'BUY', 'price': 10.0} for t in tickers]}), encoding='utf-8')
    return str(p)


@pytest.fixture
def results_dir(tmp_path):
    d = tmp_path / 'output'
    d.mkdir()
    return d


def test_in_sync_store_passes(results_dir, capsys):
    p = _write(results_dir, '2026-01-02', ['AAA', 'BBB', 'AAA'])   # duplicate ticker
    assert sync_snapshot_file(p)
    problems, info = check_store(str(results_dir), '2026-01-02')
    assert problems == [] and 'store holds 2 rows' in info[0]
    assert main(['--results-dir', str(results_dir), '--date', '2026-01-02']) == 0
    assert capsys.readouterr().out.startswith('OK: store holds 2 rows')


def test_missing_store_or_date_fails(results_dir):
    _write(results_dir, '2026-01-01', ['AAA'])
    p = _write(results_dir, '2026-01-02', ['AAA'])
    assert 'no snapshot store' in check_store(str(results_dir), '2026-01-02')[0][0]
    assert sync_snapshot_file(p)
    problems, info = check_store(str(results_dir), '2026-01-01')
    assert 'no rows for 2026-01-01' in problems[0]
    # Other dates missing are informational, not failures.
    problems, info = check_store(str(results_dir), '2026-01-02')
    assert problems == [] and any('2026-01-01' in line for line in info)


def test_failed_resync_after_a_rewrite_fails(results_dir):
    p = _write(results_dir, '2026-01-02', ['AAA'])
    assert sync_snapshot_file(p)
    # An enrichment step rewrites the snapshot, and its re-sync fails.
    later = time.time() + 60
    _write(results_dir, '2026-01-02', ['AAA', 'BBB'])
    os.utime(p, (later, later))
    problems, _ = check_store(str(results_dir), '2026-01-02')
    assert any('1 rows' in line and '2 tickers' in line for line in problems)
    assert any('stale rows' in line for line in problems)
    assert main(['--results-dir', str(results_dir), '--date', '2026-01-02']) == 1


def test_stale_schema_fails(results_dir):
    p = _write(results_dir, '2026-01-02', ['AAA'])
    assert sync_snapshot_file(p)
    with SnapshotStore(db_path_for(str(results_dir))) as store:
        store._con.execute("UPDATE schema_version SET version = 1")
    problems, _ = check_store(str(results_dir), '2026-01-02')
    assert 'schema v1' in problems[0]


def test_no_snapshot_is_exit_2(results_dir, capsys):
    assert main(['--results-dir', str(results_dir), '--date', '2026-01-09']) == 2
    assert 'could not read' in capsys.readouterr().err
