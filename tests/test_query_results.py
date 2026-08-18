# tests/test_query_results.py
"""Tests for the historical snapshot query CLI (scripts/query_results.py)."""

import sys
import os
import json
import time
import pytest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.query_results import (
    list_snapshot_files, load_snapshot, pick_snapshot,
    parse_where, apply_wheres, resolve_field,
    build_index, history_full_scan, _index_paths,
    INDEX_COLUMNS, main,
)


def _row(ticker, rating='BUY', sector='Technology', composite=60.0, **kw):
    r = {
        'ticker': ticker, 'company_name': ticker + ' Corp', 'rating': rating,
        'rating_raw': rating, 'sector': sector, 'industry': 'Software',
        'country': 'United States', '_composite_score': composite,
        '_composite_score_raw': composite, 'price': 100.0, 'mcap': 5e10,
        'mos': 0.15, 'roic': 0.18, 'pe': 22.0, '_gates_passed': '11/17',
    }
    r.update(kw)
    return r


@pytest.fixture
def results_dir(tmp_path):
    """3 canonical snapshots (oldest is old-schema: bare list, thin rows)
    plus one _replay file that must always be ignored."""
    d = tmp_path / 'output'
    d.mkdir()
    # Old schema: bare list, missing several INDEX_COLUMNS fields.
    old_rows = [{'ticker': 'AAA', 'rating': 'HOLD', 'sector': 'Technology',
                 '_composite_score': 40.0, 'price': 90.0}]
    (d / 'results_2026-01-01.json').write_text(json.dumps(old_rows))
    (d / 'results_2026-01-02.json').write_text(json.dumps({
        'date': '2026-01-02', 'count': 2,
        'results': [_row('AAA', composite=55.0, price=95.0),
                    _row('BBB', rating='PASS', sector='Healthcare',
                         composite=20.0, mcap=2e9)],
    }))
    (d / 'results_2026-01-03.json').write_text(json.dumps({
        'date': '2026-01-03', 'count': 3,
        'results': [_row('AAA', composite=62.0, price=101.0),
                    _row('BBB', rating='PASS', sector='Healthcare',
                         composite=22.0, mcap=2e9),
                    _row('CCC', rating='LEAN BUY', composite=45.0,
                         _gates_passed='9/17')],
    }))
    (d / 'results_2026-01-02_replay.json').write_text(json.dumps({
        'date': '2026-01-02', 'results': [_row('ZZZ', composite=99.0)],
    }))
    return str(d)


# --- discovery -------------------------------------------------------------

def test_replay_files_excluded(results_dir):
    files = list_snapshot_files(results_dir)
    assert [d for d, _ in files] == ['2026-01-01', '2026-01-02', '2026-01-03']
    assert not any('replay' in os.path.basename(p) for _, p in files)


def test_load_snapshot_handles_bare_list_and_dict(results_dir):
    files = dict(list_snapshot_files(results_dir))
    meta, rows = load_snapshot(files['2026-01-01'])
    assert meta == {} and len(rows) == 1
    meta, rows = load_snapshot(files['2026-01-02'])
    assert meta['count'] == 2 and len(rows) == 2


def test_pick_snapshot_latest_and_miss(results_dir):
    files = list_snapshot_files(results_dir)
    assert pick_snapshot(files, None)[0] == '2026-01-03'
    assert pick_snapshot(files, '2026-01-02')[0] == '2026-01-02'
    with pytest.raises(SystemExit) as e:
        pick_snapshot(files, '2026-02-15')
    assert 'Nearest available' in str(e.value)


# --- where parsing / application -------------------------------------------

@pytest.mark.parametrize('expr,expected', [
    ('composite>=57', ('composite', '>=', 57.0)),
    ('mcap<=1e10', ('mcap', '<=', 1e10)),
    ('pe>10', ('pe', '>', 10.0)),
    ('pe<10', ('pe', '<', 10.0)),
    ('country==Canada', ('country', '==', 'Canada')),
    ('rating!=PASS', ('rating', '!=', 'PASS')),
])
def test_parse_where_ops(expr, expected):
    assert parse_where(expr) == expected


def test_parse_where_bad_expr_exits():
    with pytest.raises(SystemExit):
        parse_where('no-operator-here')


def test_resolve_field_alias_and_unknown():
    cols = ['_composite_score', 'mcap', 'ticker']
    assert resolve_field('composite', cols) == '_composite_score'
    assert resolve_field('market_cap', cols) == 'mcap'
    with pytest.raises(SystemExit) as e:
        resolve_field('compositee_score', cols)
    assert 'Closest matches' in str(e.value)


def test_apply_wheres_numeric_excludes_missing():
    df = pd.DataFrame([
        {'ticker': 'A', 'roic': 0.20},
        {'ticker': 'B', 'roic': None},   # missing → excluded by numeric filter
        {'ticker': 'C', 'roic': 0.05},
    ])
    out = apply_wheres(df, [('roic', '>=', 0.10)])
    assert list(out['ticker']) == ['A']


def test_apply_wheres_string_and_gates():
    df = pd.DataFrame([
        {'ticker': 'A', 'country': 'Canada', '_gates_passed': '11/17'},
        {'ticker': 'B', 'country': 'United States', '_gates_passed': '5/17'},
    ])
    assert list(apply_wheres(df, [('country', '==', 'Canada')])['ticker']) == ['A']
    # "11/17" strings compare by their leading number
    assert list(apply_wheres(df, [('gates', '>=', 10.0)])['ticker']) == ['A']


# --- index cache ------------------------------------------------------------

def test_build_index_and_incremental_rebuild(results_dir):
    idx = build_index(results_dir)
    assert sorted(idx['date'].unique()) == ['2026-01-01', '2026-01-02', '2026-01-03']
    assert len(idx) == 6
    assert list(idx.columns) == INDEX_COLUMNS
    # Old-schema snapshot: absent fields survive as NaN, not KeyErrors.
    old = idx[idx['date'] == '2026-01-01'].iloc[0]
    assert pd.isna(old['roic']) and pd.isna(old['industry'])

    # Second build touches nothing (all index files stay fresh).
    pq, csv = _index_paths(results_dir, '2026-01-02')
    dest = pq if os.path.exists(pq) else csv
    mtime = os.path.getmtime(dest)
    build_index(results_dir)
    assert os.path.getmtime(dest) == mtime

    # Touching one source JSON rebuilds only that snapshot's index file.
    src = os.path.join(results_dir, 'results_2026-01-02.json')
    future = time.time() + 5
    os.utime(src, (future, future))
    pq3, csv3 = _index_paths(results_dir, '2026-01-03')
    dest3 = pq3 if os.path.exists(pq3) else csv3
    mtime3 = os.path.getmtime(dest3)
    build_index(results_dir)
    assert os.path.getmtime(dest) > mtime
    assert os.path.getmtime(dest3) == mtime3


def test_history_via_index_ordered(results_dir):
    idx = build_index(results_dir)
    hist = idx[idx['ticker'] == 'AAA'].sort_values('date')
    assert list(hist['date']) == ['2026-01-01', '2026-01-02', '2026-01-03']
    assert list(hist['rating']) == ['HOLD', 'BUY', 'BUY']
    assert list(hist['_composite_score']) == [40.0, 55.0, 62.0]


def test_history_full_scan_matches_index(results_dir):
    df = history_full_scan(results_dir, 'AAA',
                           ['date', 'rating', '_composite_score', 'company_name'])
    assert list(df['date']) == ['2026-01-01', '2026-01-02', '2026-01-03']
    # company_name is off-index; old snapshot lacks it → None/NaN.
    assert pd.isna(df['company_name'].iloc[0])
    assert df['company_name'].iloc[1] == 'AAA Corp'


# --- end-to-end CLI ----------------------------------------------------------

def _run_main(argv, capsys):
    old = sys.argv
    sys.argv = ['query_results.py'] + argv
    try:
        main()
    finally:
        sys.argv = old
    return capsys.readouterr()


def test_cli_snapshot_filters_and_csv(results_dir, tmp_path, capsys):
    out_csv = str(tmp_path / 'out.csv')
    res = _run_main(['--results-dir', results_dir,
                     '--where', 'composite>=40', '--rating', 'BUY',
                     '--csv', out_csv], capsys)
    assert 'AAA' in res.out and 'BBB' not in res.out
    written = pd.read_csv(out_csv)
    assert list(written['ticker']) == ['AAA']


def test_cli_history_json(results_dir, tmp_path, capsys):
    out_json = str(tmp_path / 'out.json')
    res = _run_main(['--results-dir', results_dir,
                     '--ticker', 'aaa', '--history', '--json', out_json], capsys)
    assert 'HOLD' in res.out
    recs = json.loads(open(out_json).read())
    assert [r['date'] for r in recs] == ['2026-01-01', '2026-01-02', '2026-01-03']


def test_cli_unknown_ticker_history_exits(results_dir, capsys):
    with pytest.raises(SystemExit) as e:
        _run_main(['--results-dir', results_dir, '--ticker', 'NOPE', '--history'],
                  capsys)
    assert 'not found' in str(e.value)
