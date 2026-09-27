# tests/test_backtest_parquet_staleness.py
"""A Parquet export must never shadow a snapshot that has since been rewritten.

load_corpus prefers <results_dir>/parquet/results_<date>.parquet over both the
DuckDB store and the JSON, but only db_publish / storage.upload_run ever wrote
those files -- rescore_and_render and the enrich_* steps refresh the store via
sync_snapshot_file and left the export alone. A stale export therefore served
the ratings the snapshot held BEFORE the re-score, together with its old
provenance.scoring.params_hash: the pooled model was then invisible to
model_regimes and to backtest_cloud.compare's NOTICE, the two checks that exist
to catch exactly that. So the export is invalidated when the snapshot is
rewritten, rebuilt by export_dir, and ignored by the reader if it is older.
"""

import json
import os
import time

import pytest

from data import snapshot_store as ss
from scripts.backtest import load_corpus

pytest.importorskip('pyarrow')

from data.db.parquet import export_dir, export_snapshot, parquet_path  # noqa: E402

DATE = '2026-07-06'


def _snapshot(rating, params_hash):
    return {'date': DATE, 'count': 2,
            'provenance': {'scoring': {'params_hash': params_hash,
                                       'git_sha': 'abc1234'}},
            'results': [{'ticker': 'AAA', 'price': 10.0, 'rating': rating,
                         '_composite_score': 0.7},
                        {'ticker': 'BBB', 'price': 20.0, 'rating': 'PASS',
                         '_composite_score': 0.1}]}


def _write(results_dir, data):
    path = os.path.join(results_dir, f'results_{DATE}.json')
    ss.write_snapshot_file(path, data)
    return path


def _age(path, seconds=5):
    """Backdate a file, so 'older than the snapshot' does not depend on clock
    resolution."""
    old = time.time() - seconds
    os.utime(path, (old, old))


@pytest.fixture
def corpus(tmp_path):
    d = tmp_path / 'output'
    d.mkdir()
    (d / 'parquet').mkdir()
    return str(d)


def test_stale_export_is_ignored_and_the_rewritten_snapshot_wins(corpus, caplog):
    path = _write(corpus, _snapshot('BUY', 'OLDHASH'))
    export_snapshot(ss.read_snapshot(path), DATE, parquet_path(corpus + '/parquet', DATE))
    _age(parquet_path(corpus + '/parquet', DATE))
    # the re-score: same file, new rating and a new scoring fingerprint
    _write(corpus, _snapshot('PASS', 'NEWHASH'))

    with caplog.at_level('WARNING'):
        snaps = load_corpus(corpus)
    assert len(snaps) == 1
    rows = {r['ticker']: r for r in snaps[0]['results']}
    assert rows['AAA']['rating'] == 'PASS'                    # not the stale BUY
    assert (snaps[0]['provenance']['scoring']['params_hash'] == 'NEWHASH')
    assert 'older than' in caplog.text


def test_a_current_export_is_still_used(corpus):
    path = _write(corpus, _snapshot('BUY', 'HASH1'))
    _age(path)                                                # snapshot older
    export_snapshot(ss.read_snapshot(path), DATE, parquet_path(corpus + '/parquet', DATE))
    snaps = load_corpus(corpus)
    rows = {r['ticker']: r for r in snaps[0]['results']}
    assert rows['AAA']['rating'] == 'BUY'
    assert not ss.parquet_export_is_stale(
        parquet_path(corpus + '/parquet', DATE), path)


def test_sync_snapshot_file_drops_the_export(corpus):
    path = _write(corpus, _snapshot('BUY', 'HASH1'))
    pq = parquet_path(corpus + '/parquet', DATE)
    export_snapshot(ss.read_snapshot(path), DATE, pq)
    assert os.path.exists(pq)
    rewritten = _snapshot('PASS', 'HASH2')
    ss.write_snapshot_file(path, rewritten)
    assert ss.sync_snapshot_file(path, data=rewritten) is True
    assert not os.path.exists(pq), 'the derived export outlived its snapshot'


def test_export_dir_rebuilds_a_stale_file_without_replace(corpus):
    path = _write(corpus, _snapshot('BUY', 'HASH1'))
    pq_dir = os.path.join(corpus, 'parquet')
    export_snapshot(ss.read_snapshot(path), DATE, parquet_path(pq_dir, DATE))
    _age(parquet_path(pq_dir, DATE))
    _write(corpus, _snapshot('PASS', 'HASH2'))

    assert export_dir(corpus, pq_dir) == [(DATE, 2)]          # rebuilt, no replace=
    snaps = load_corpus(corpus)
    rows = {r['ticker']: r for r in snaps[0]['results']}
    assert rows['AAA']['rating'] == 'PASS'
    # and a current file is left alone
    assert export_dir(corpus, pq_dir) == []


def test_missing_export_is_not_an_error(corpus):
    _write(corpus, _snapshot('BUY', 'HASH1'))
    assert ss.parquet_export_is_stale(
        parquet_path(os.path.join(corpus, 'parquet'), DATE), 'nope.json') is True
    assert ss.invalidate_parquet_export(
        os.path.join(corpus, f'results_{DATE}.json')) is None
    snaps = load_corpus(corpus)
    assert json.loads(json.dumps(snaps[0]['date'])) == DATE
