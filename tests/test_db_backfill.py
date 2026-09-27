"""Backfill resume logic and the parity check's history rule (offline)."""
import pytest

from data.db.publish import canonical_sha256
from data.snapshot_store import write_snapshot_file
from scripts import db_backfill
from scripts.db_parity_check import HistoryFromFiles


def _snap(d, ratings):
    return {'date': d, 'risk_free_rate': 0.04,
            'results': [{'ticker': t, 'rating': r} for t, r in ratings.items()]}


class FakeTransport:
    def __init__(self, runs=(), refuse=()):
        self.runs = list(runs)
        self.refuse = set(refuse)
        self.published = []

    def call(self, fn, args, idempotent=False):
        if fn == 'list_runs':
            return self.runs
        if fn == 'publish_run':
            if args['p_run_date'] in self.refuse:
                from data.db.publish import PublishError
                raise PublishError('publish_run: refused for the test')
            self.published.append(args['p_run_date'])
            return {'rows': args['p_expect']['n_rows'], 'rating_changes': 0, 'new_blobs': 0, 'warnings': []}
        return {'staged': True}


@pytest.fixture
def archive(tmp_path):
    snaps = {'2026-04-20': _snap('2026-04-20', {'A': 'BUY'}),
             '2026-04-21': _snap('2026-04-21', {'A': 'HOLD'}),
             '2026-04-22': _snap('2026-04-22', {'A': 'HOLD', 'B': 'BUY'})}
    for d, data in snaps.items():
        write_snapshot_file(str(tmp_path / f'results_{d}.json'), data)
    return tmp_path, snaps


def _run(monkeypatch, transport, *args):
    monkeypatch.setattr(db_backfill, 'make_transport', lambda a: (transport, None))
    return db_backfill.main(list(args))


def test_backfill_publishes_in_date_order(monkeypatch, archive):
    path, _ = archive
    t = FakeTransport()
    assert _run(monkeypatch, t, '--results-dir', str(path)) == 0
    assert t.published == ['2026-04-20', '2026-04-21', '2026-04-22']


def test_backfill_skips_unchanged_and_republishes_changed(monkeypatch, archive):
    path, snaps = archive
    runs = [{'run_date': '2026-04-20', 'status': 'complete', 'source_sha256': canonical_sha256(snaps['2026-04-20'])},
            {'run_date': '2026-04-21', 'status': 'complete', 'source_sha256': '0' * 64}]     # file changed since
    t = FakeTransport(runs)
    assert _run(monkeypatch, t, '--results-dir', str(path)) == 0
    assert t.published == ['2026-04-21', '2026-04-22']
    t = FakeTransport(runs)
    assert _run(monkeypatch, t, '--results-dir', str(path), '--replace', '--since', '2026-04-20',
                '--until', '2026-04-21') == 0
    assert t.published == ['2026-04-20', '2026-04-21']


def test_backfill_stops_at_a_refusal_unless_keep_going(monkeypatch, archive):
    path, _ = archive
    t = FakeTransport(refuse={'2026-04-21'})
    assert _run(monkeypatch, t, '--results-dir', str(path)) == 1
    assert t.published == ['2026-04-20']
    t = FakeTransport(refuse={'2026-04-21'})
    assert _run(monkeypatch, t, '--results-dir', str(path), '--keep-going') == 1
    assert t.published == ['2026-04-20', '2026-04-22']


def test_dry_run_sends_nothing(monkeypatch, archive):
    path, _ = archive
    t = FakeTransport()
    assert _run(monkeypatch, t, '--results-dir', str(path), '--dry-run') == 0
    assert t.published == []


def test_history_rule_matches_the_duckdb_store():
    h = HistoryFromFiles()
    h.add('d1', [{'ticker': 'A', 'rating': 'BUY'}, {'ticker': 'B', 'rating': None}])
    h.add('d2', [{'ticker': 'A', 'rating': 'BUY'}, {'ticker': 'B', 'rating': 'HOLD'}])
    h.add('d3', [{'ticker': 'A', 'rating': ''}, {'ticker': 'B', 'rating': 'HOLD'}])      # gap
    h.add('d4', [{'ticker': 'A', 'rating': 'BUY'}, {'ticker': 'B', 'rating': 'PASS'}])
    assert h.points == {('A', 'd1', 'BUY'), ('B', 'd2', 'HOLD'), ('B', 'd4', 'PASS')}
