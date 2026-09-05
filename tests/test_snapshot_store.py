# tests/test_snapshot_store.py
"""Tests for the DuckDB snapshot store (data/snapshot_store.py) and the
readers that use it (gate N/A report, portfolio tracker prior lookup,
report_html prior-rating / rating-history fast paths, ingest CLI)."""

import json
import math
import os
from datetime import date

import numpy as np
import pytest

from data.snapshot_store import (
    SnapshotStore, db_path_for, list_snapshot_files, prior_snapshot_file,
    snapshot_date_from_path, split_snapshot, sync_snapshot_file,
)
from scripts.ingest_snapshots import ingest_dir


def _row(ticker, rating='BUY', **kw):
    r = {
        'ticker': ticker, 'rating': rating, 'rating_raw': rating,
        'sector': 'Technology', 'price': 100.0, 'mcap': 5e10,
        'shares_out': 500_000_000, 'founder_led': False,
        '_composite_score': 60.0, '_gates_passed': '11/17',
        '_rating_cap_reasons': [], 'dcf_sens_range': (90.0, 130.0),
        'edgar_history': {'revenue_history': {'2024-12-31': 1.0e9},
                          'operating_income_history': {'2024-12-31': 2.0e8},
                          'years_available': 11},
        'beta': None, '_gate_roic': 0.2, '_gate_mos': None,
    }
    r.update(kw)
    return r


def _write(d, day, rows, meta=None, bare=False):
    p = d / f'results_{day}.json'
    if bare:
        p.write_text(json.dumps(rows), encoding='utf-8')
    else:
        payload = {'date': day, 'risk_free_rate': 0.04,
                   'risk_free_rate_source': 'fred', 'count': len(rows),
                   'provenance': {'run': day}}
        payload.update(meta or {})
        payload['results'] = rows
        p.write_text(json.dumps(payload), encoding='utf-8')
    return str(p)


@pytest.fixture
def results_dir(tmp_path):
    d = tmp_path / 'output'
    d.mkdir()
    _write(d, '2026-01-01', [_row('AAA', 'HOLD'), _row('BBB', 'PASS')], bare=True)
    _write(d, '2026-01-02', [_row('AAA', 'HOLD', founder_led=True),
                             _row('BBB', 'BUY', mcap=None),
                             _row('CCC', 'LEAN BUY', new_field=3)])
    _write(d, '2026-01-03', [_row('AAA', 'BUY'), _row('BBB', 'BUY'),
                             _row('CCC', 'LEAN BUY', new_field=3.5)])
    # Re-scored copy: must never be treated as a live snapshot.
    (d / 'results_2026-01-02_replay.json').write_text(
        json.dumps({'date': '2026-01-02', 'results': [_row('ZZZ')]}), encoding='utf-8')
    return d


# --- file helpers ----------------------------------------------------------

def test_snapshot_discovery_rejects_replay_and_bad_dates(results_dir):
    files = list_snapshot_files(str(results_dir))
    assert [d for d, _ in files] == ['2026-01-01', '2026-01-02', '2026-01-03']
    assert snapshot_date_from_path('output/results_2026-01-02_replay.json') is None
    assert snapshot_date_from_path('output/results_2026-13-40.json') is None
    assert prior_snapshot_file(str(results_dir), '2026-01-03')[0] == '2026-01-02'
    assert prior_snapshot_file(str(results_dir), date(2026, 1, 1)) is None
    assert split_snapshot([{'ticker': 'A'}]) == ({}, [{'ticker': 'A'}])
    meta, rows = split_snapshot({'date': 'x', 'results': [1]})
    assert meta == {'date': 'x'} and rows == [1]


# --- ingest ----------------------------------------------------------------

def test_ingest_is_idempotent_and_replace_reloads(results_dir):
    with SnapshotStore(db_path_for(str(results_dir))) as store:
        for _, p in list_snapshot_files(str(results_dir)):
            assert store.ingest_json(p) is True
        assert store.ingest_json(str(results_dir / 'results_2026-01-02.json')) is False
        assert store.ingest_json(str(results_dir / 'results_2026-01-02_replay.json')) is False
        assert store.dates() == ['2026-01-01', '2026-01-02', '2026-01-03']
        assert store.counts() == {'2026-01-01': 2, '2026-01-02': 3, '2026-01-03': 3}
        assert store.latest_date() == '2026-01-03'
        assert store.latest_date(before='2026-01-03') == '2026-01-02'
        assert store.latest_date(before=date(2026, 1, 1)) is None
        # replace=True reloads the date (row count stays consistent)
        assert store.ingest_json(str(results_dir / 'results_2026-01-02.json'),
                                 replace=True) is True
        assert store.counts()['2026-01-02'] == 3
        # bare-list snapshots record no meta; dict snapshots keep theirs
        assert store.run_meta('2026-01-01')['risk_free_rate'] is None
        meta = store.run_meta('2026-01-02')
        assert meta['risk_free_rate'] == 0.04 and meta['provenance'] == {'run': '2026-01-02'}
        assert store.run_meta('2030-01-01') is None


def test_schema_drift_adds_and_widens_columns(results_dir):
    with SnapshotStore(db_path_for(str(results_dir))) as store:
        store.ingest_json(str(results_dir / 'results_2026-01-01.json'))
        types = store.column_types()
        assert 'new_field' not in types
        assert types['founder_led'] == 'BOOLEAN'
        assert types['_rating_cap_reasons'] == 'JSON'
        assert types['edgar_history'] == 'JSON'  # slim projection, see below
        assert types['shares_out'] == 'BIGINT'
        # all-null keys create no column until a value shows up
        assert 'beta' not in types and '_gate_mos' not in types
        store.ingest_json(str(results_dir / 'results_2026-01-02.json'))
        types = store.column_types()
        assert types['new_field'] == 'BIGINT'
        store.ingest_json(str(results_dir / 'results_2026-01-03.json'))
        assert store.column_types()['new_field'] == 'DOUBLE'  # widened by 3.5
        rows = {r['ticker']: r for r in store.rows('2026-01-02', ['new_field', 'mcap'])}
        assert rows['CCC']['new_field'] == 3 and rows['AAA']['new_field'] is None
        assert rows['BBB']['mcap'] is None
        # string in a numeric column widens it to VARCHAR without failing
        store.ingest_rows({'date': '2026-01-04'},
                          [_row('AAA', mcap='n/a')], replace=True)
        assert store.column_types()['mcap'] == 'VARCHAR'
        assert store.rows('2026-01-04', ['mcap'])[0]['mcap'] == 'n/a'


def test_ingest_rows_normalises_numpy_and_dedupes_tickers(tmp_path):
    with SnapshotStore(str(tmp_path / 's.duckdb')) as store:
        rows = [_row('AAA', pe=np.float64(12.5), beta=np.nan),
                _row('AAA', pe=13.0, n=np.int64(3)),  # duplicate: last wins
                {'ticker': None, 'rating': 'BUY'},  # no ticker: dropped
                {'no_ticker': 1}]
        assert store.ingest_rows({}, rows, run_date=date(2026, 2, 1)) is True
        got = store.rows('2026-02-01')
        assert len(got) == 1 and got[0]['pe'] == 13.0 and got[0]['n'] == 3
        assert got[0]['dcf_sens_range'] == [90.0, 130.0]  # JSON round-trips
        assert got[0]['_rating_cap_reasons'] == []
        with pytest.raises(ValueError):
            store.ingest_rows({}, rows)  # no date anywhere
        # edgar_history is stored as the slim projection the re-scoring path
        # reads, not the whole 54-series block.
        assert got[0]['edgar_history'] == {
            'operating_income_history': {'2024-12-31': 2.0e8},
            'years_available': 11}
        # projections={} keeps the block whole; exclude drops it entirely.
        store.ingest_rows({}, rows, run_date='2026-02-02', projections={})
        whole = store.rows('2026-02-02', ['edgar_history'])[0]['edgar_history']
        assert whole['revenue_history'] == {'2024-12-31': 1.0e9}
        store.ingest_rows({}, rows, run_date='2026-02-03',
                          exclude=('edgar_history',))
        assert store.rows('2026-02-03', ['edgar_history'])[0]['edgar_history'] is None


def test_rows_prior_rows_and_unknown_columns(results_dir):
    with SnapshotStore(db_path_for(str(results_dir))) as store:
        for _, p in list_snapshot_files(str(results_dir)):
            store.ingest_json(p)
        d, rows = store.prior_rows('2026-01-03', ['shares_out', 'mcap', 'nope'])
        assert d == '2026-01-02'
        assert [r['ticker'] for r in rows] == ['AAA', 'BBB', 'CCC']
        assert rows[0] == {'ticker': 'AAA', 'shares_out': 500_000_000,
                           'mcap': 5e10, 'nope': None}
        assert store.prior_rows('2026-01-01') == (None, [])
        assert store.rows('2030-01-01') == []
        df = store.query("SELECT ticker FROM results WHERE date = ? AND rating = 'BUY' "
                         "ORDER BY ticker", ['2026-01-03'])
        assert list(df['ticker']) == ['AAA', 'BBB']


def test_last_known_rows_and_rating_history(results_dir):
    d = results_dir
    # Degraded run on 01-04 drops BBB; 01-05 is the render date.
    _write(d, '2026-01-04', [_row('AAA', 'BUY'), _row('CCC', 'HOLD')])
    with SnapshotStore(db_path_for(str(d))) as store:
        for _, p in list_snapshot_files(str(d)):
            store.ingest_json(p)
        primary, out, n_fb = store.last_known_rows(
            '2026-01-05', ['rating', '_composite_score', 'missing'], max_lookback=7)
        assert primary == '2026-01-04'
        assert out['AAA']['rating'] == 'BUY' and out['CCC']['rating'] == 'HOLD'
        assert out['BBB']['rating'] == 'BUY'      # from 01-03 via look-back
        assert out['BBB']['missing'] is None and n_fb == 1
        # look-back of 1 sees only the primary snapshot
        _, out1, n_fb1 = store.last_known_rows('2026-01-05', ['rating'], max_lookback=1)
        assert set(out1) == {'AAA', 'CCC'} and n_fb1 == 0
        assert store.last_known_rows('2026-01-01', ['rating']) == (None, {}, 0)

        hist = store.rating_history()
        assert hist['AAA'] == [['2026-01-01', 'HOLD'], ['2026-01-03', 'BUY']]
        assert hist['BBB'] == [['2026-01-01', 'PASS'], ['2026-01-02', 'BUY']]
        assert hist['CCC'] == [['2026-01-02', 'LEAN BUY'], ['2026-01-04', 'HOLD']]
        assert store.rating_history(before='2026-01-03')['AAA'] == [['2026-01-01', 'HOLD']]
        assert store.rating_history(column='not_a_column') == {}


def test_open_existing_and_read_only(tmp_path):
    p = str(tmp_path / 'missing.duckdb')
    assert SnapshotStore.open_existing(p) is None
    assert not os.path.exists(p)
    with SnapshotStore(p) as store:
        store.ingest_rows({}, [_row('AAA')], run_date='2026-03-01')
    ro = SnapshotStore.open_existing(p)
    assert ro is not None and ro.read_only
    with ro:
        assert ro.dates() == ['2026-03-01']
        with pytest.raises(RuntimeError):
            ro.ingest_rows({}, [_row('BBB')], run_date='2026-03-02')


def test_sync_snapshot_file_from_memory_and_disk(results_dir):
    p = str(results_dir / 'results_2026-01-02.json')
    data = {'date': '2026-01-02', 'results': [_row('AAA', 'PASS')]}
    assert sync_snapshot_file(p, data=data) is True
    with SnapshotStore.open_existing(db_path_for(str(results_dir))) as store:
        assert [r['rating'] for r in store.rows('2026-01-02', ['rating'])] == ['PASS']
    # From disk (re-parses the file, replacing the in-memory version)
    assert sync_snapshot_file(p) is True
    with SnapshotStore.open_existing(db_path_for(str(results_dir))) as store:
        assert store.counts()['2026-01-02'] == 3
    assert sync_snapshot_file(str(results_dir / 'results_2026-01-02_replay.json')) is False
    # Unreadable target never raises
    assert sync_snapshot_file(p, db_path=str(results_dir / 'nodir' / 'x' / 'y.duckdb'),
                              data=data) in (True, False)


# --- ingest CLI ------------------------------------------------------------

def test_ingest_cli_backfills_incrementally(results_dir, capsys):
    ingested, skipped, failed = ingest_dir(str(results_dir))
    assert ingested == ['2026-01-01', '2026-01-02', '2026-01-03'] and not skipped and not failed
    ingested, skipped, failed = ingest_dir(str(results_dir))
    assert not ingested and len(skipped) == 3
    ingested, _, _ = ingest_dir(str(results_dir), since='2026-01-03', replace=True)
    assert ingested == ['2026-01-03']
    ingested, skipped, _ = ingest_dir(
        str(results_dir), paths=[str(results_dir / 'results_2026-01-02.json'),
                                 str(results_dir / 'results_2026-01-02_replay.json')])
    assert skipped == ['2026-01-02'] and not ingested
    (results_dir / 'results_2026-01-06.json').write_text('{not json', encoding='utf-8')
    _, _, failed = ingest_dir(str(results_dir))
    assert failed == ['2026-01-06']
    from scripts.ingest_snapshots import main
    assert main(['--results-dir', str(results_dir)]) == 1  # the broken file
    (results_dir / 'results_2026-01-06.json').unlink()
    assert main(['--results-dir', str(results_dir)]) == 0


# --- readers ---------------------------------------------------------------

def test_gate_na_report_reads_prior_from_store(results_dir, capsys):
    from scripts.gate_na_report import _load_prior_records, _prior_snapshot
    cur = str(results_dir / 'results_2026-01-03.json')
    prior = _prior_snapshot(cur)
    assert prior[0] == '2026-01-02'
    gates = [{'key': '_gate_roic'}, {'key': '_gate_mos'}]
    # No store yet: JSON path (full rows)
    recs = _load_prior_records(prior, gates)
    assert len(recs) == 3 and 'edgar_history' in recs[0]
    ingest_dir(str(results_dir))
    recs = _load_prior_records(prior, gates)
    assert len(recs) == 3 and set(recs[0]) == {'ticker', '_gate_roic', '_gate_mos'}
    assert all(r['_gate_mos'] is None for r in recs)
    assert _prior_snapshot('output/results_2026-01-02_replay.json') is None


def test_track_portfolio_prior_lookup(results_dir):
    from scripts.track_portfolio import _find_latest_results, _load_prior_by_ticker
    latest = _find_latest_results(str(results_dir))
    assert latest.endswith('results_2026-01-03.json')
    prev = _find_latest_results(str(results_dir), exclude=latest)
    assert prev.endswith('results_2026-01-02.json')
    by_tk = _load_prior_by_ticker(prev)
    assert by_tk['BBB']['rating'] == 'BUY' and 'edgar_history' in by_tk['BBB']
    ingest_dir(str(results_dir))
    by_tk = _load_prior_by_ticker(prev)
    assert by_tk == {'AAA': {'ticker': 'AAA', 'rating': 'HOLD'},
                     'BBB': {'ticker': 'BBB', 'rating': 'BUY'},
                     'CCC': {'ticker': 'CCC', 'rating': 'LEAN BUY'}}


def test_report_html_readers_match_json_path(results_dir, capsys):
    from scripts.report_html import _load_prev_ratings, _load_rating_history
    out_dir = str(results_dir)
    run_date = date(2026, 1, 3)
    extra = ['_gate_roic', '_gate_mos']
    json_prev = _load_prev_ratings(out_dir, run_date, extra_keys=extra)
    json_hist = _load_rating_history(out_dir, run_date, cache_name='rh_json.json')
    # Store lagging the files (only the oldest date) must NOT be used.
    with SnapshotStore(db_path_for(out_dir)) as store:
        store.ingest_json(str(results_dir / 'results_2026-01-01.json'))
    capsys.readouterr()
    assert _load_prev_ratings(out_dir, run_date, extra_keys=extra) == json_prev
    assert 'snapshot store' not in capsys.readouterr().out
    ingest_dir(out_dir)
    capsys.readouterr()
    store_prev = _load_prev_ratings(out_dir, run_date, extra_keys=extra)
    store_hist = _load_rating_history(out_dir, run_date, cache_name='rh_store.json')
    out = capsys.readouterr().out
    assert 'rate-change baseline read from snapshot store' in out
    assert 'rating history read from snapshot store' in out
    assert store_prev == json_prev
    assert store_hist == json_hist
    assert store_prev[0] == '2026-01-02'
    assert store_prev[1]['BBB']['rating'] == 'BUY'
    assert store_prev[1]['BBB']['_rating_cap_reasons'] == []
    assert store_hist['BBB'] == [['2026-01-01', 'PASS'], ['2026-01-02', 'BUY']]
    # The store path writes no rating_history cache file.
    assert not (results_dir / 'rh_store.json').exists()
    # No run_date: the JSON path is used (store needs a cut-off).
    assert _load_prev_ratings(out_dir, None, extra_keys=extra)[0] == '2026-01-03'


def test_analyze_stock_carry_forward_reads_store(results_dir, capsys):
    from scripts.analyze_stock import _load_carry_forward_rows
    d, p = list_snapshot_files(str(results_dir))[-1]
    rows = _load_carry_forward_rows(d, p)
    assert len(rows) == 3 and 'edgar_history' in rows[0]  # JSON path
    ingest_dir(str(results_dir))
    rows = _load_carry_forward_rows(d, p)
    assert {r['ticker'] for r in rows} == {'AAA', 'BBB', 'CCC'}
    assert set(rows[0]) == {'ticker', 'shares_out', 'mcap'}
    assert 'snapshot store' in capsys.readouterr().out


# --- schema versioning ------------------------------------------------------

def test_stale_schema_is_ignored_by_readers_and_rebuilt_on_write(results_dir, caplog):
    """A store built under an older layout holds columns projected by different
    rules, so readers must ignore it rather than trust it."""
    import data.snapshot_store as ss
    db = db_path_for(str(results_dir))
    ingest_dir(str(results_dir))
    with SnapshotStore.open_existing(db) as store:
        assert store.schema_version() == ss.SCHEMA_VERSION
        assert store.dates()

    # Simulate a store written by an older release.
    with SnapshotStore(db) as store:
        store._con.execute("UPDATE schema_version SET version = ?",
                           [ss.SCHEMA_VERSION - 1])

    # Readers refuse it, so every migrated caller falls back to the JSON files.
    assert SnapshotStore.open_existing(db) is None

    # A writable open rebuilds it empty at the current version; the JSONs stay
    # the source of truth, so a re-ingest refills it.
    with SnapshotStore(db) as store:
        assert store.schema_version() == ss.SCHEMA_VERSION
        assert store.dates() == []
    ingested, _, _ = ingest_dir(str(results_dir))
    assert ingested == ['2026-01-01', '2026-01-02', '2026-01-03']


# --- what the store may drop, and what it must not -------------------------

class TestExclusionsAndTypes:

    def test_excluded_keys_are_never_read_by_the_scoring_path(self):
        """Guard on DEFAULT_EXCLUDE_KEYS: it drops report payload only.

        Adding a field scoring depends on would silently change re-scored
        ratings for every store-backed backtest, so the list is asserted
        disjoint from the gate fields, the rating drivers and the columns
        query_results exposes.
        """
        import data.snapshot_store as ss
        from scripts.query_results import (DEFAULT_COLUMNS, FIELD_ALIASES,
                                           HISTORY_COLUMNS, INDEX_COLUMNS)
        from scripts.report_html import _PREV_DRIVER_KEYS
        from scripts.scoring import GATES
        excluded = set(ss.DEFAULT_EXCLUDE_KEYS)
        assert excluded, 'the narrative payload should still be excluded'
        assert not excluded & {g.field for g in GATES}
        assert not excluded & set(_PREV_DRIVER_KEYS)
        assert not excluded & (set(INDEX_COLUMNS) | set(DEFAULT_COLUMNS)
                               | set(HISTORY_COLUMNS) | set(FIELD_ALIASES.values()))
        # Fields that look like report payload but that scoring does read.
        assert not excluded & {'roic_by_year', '_nopat_by_year', '_ic_by_year',
                               '_trap_components', 'edgar_history'}

    def test_excluded_columns_are_absent_but_the_row_is_otherwise_whole(self, tmp_path):
        import data.snapshot_store as ss
        row = _row('AAA')
        row.update({k: ['payload'] for k in ss.DEFAULT_EXCLUDE_KEYS})
        with SnapshotStore(str(tmp_path / 's.duckdb')) as store:
            store.ingest_rows({}, [row], run_date='2026-01-01')
            got = store.rows('2026-01-01')[0]
            assert not set(got) & set(ss.DEFAULT_EXCLUDE_KEYS)
            assert got['rating'] == 'BUY' and got['_gate_roic'] == 0.2

    def test_nan_survives_the_round_trip(self, tmp_path):
        """Scoring reads a missing value (None) as N/A but a NaN as a failed
        comparison, so collapsing NaN to NULL moves rows between those."""
        with SnapshotStore(str(tmp_path / 's.duckdb')) as store:
            store.ingest_rows({}, [_row('AAA', mos=float('nan')),
                                   _row('BBB', mos=0.2)], run_date='2026-01-01')
            got = {r['ticker']: r for r in store.rows('2026-01-01', ['mos'])}
            assert math.isnan(got['AAA']['mos'])
            assert got['BBB']['mos'] == 0.2

    def test_stringified_infinity_does_not_make_a_numeric_column_textual(self, tmp_path):
        """One "Infinity" among 2,413 floats used to turn every `pe` in the
        real archive into a string. It must cast, not widen."""
        with SnapshotStore(str(tmp_path / 's.duckdb')) as store:
            store.ingest_rows({}, [_row('AAA', pe=18.5), _row('BBB', pe='Infinity'),
                                   _row('CCC', pe=-21.0)], run_date='2026-01-01')
            assert store.column_types()['pe'] == 'DOUBLE'
            got = {r['ticker']: r for r in store.rows('2026-01-01', ['pe'])}
            assert got['AAA']['pe'] == 18.5 and got['CCC']['pe'] == -21.0
            assert math.isinf(got['BBB']['pe'])

    def test_a_genuinely_textual_column_keeps_those_words(self, tmp_path):
        """The coercion is driven by the column's other values, so a ticker
        called NAN or a company called 'Infinity Corp' is left alone."""
        with SnapshotStore(str(tmp_path / 's.duckdb')) as store:
            store.ingest_rows({}, [_row('NAN', company_name='NaN Holdings'),
                                   _row('BBB', company_name='Infinity')],
                              run_date='2026-01-01')
            got = {r['ticker']: r for r in store.rows('2026-01-01', ['company_name'])}
            assert got['NAN']['company_name'] == 'NaN Holdings'
            assert got['BBB']['company_name'] == 'Infinity'

    def test_nonfinite_nested_in_a_json_block_is_stored_as_null(self, tmp_path):
        """Bare NaN/Infinity tokens are not valid JSON and DuckDB rejects them."""
        with SnapshotStore(str(tmp_path / 's.duckdb')) as store:
            store.ingest_rows({}, [_row('AAA', roic_by_year={
                '2024': float('inf'), '2023': float('nan'), '2022': 0.2})],
                run_date='2026-01-01')
            got = store.rows('2026-01-01', ['roic_by_year'])[0]['roic_by_year']
            assert got == {'2024': None, '2023': None, '2022': 0.2}
