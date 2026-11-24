"""scripts/publish_bond_history.py: the bond repo's history tables as the
app's bucketed app/bond_px/ files."""
import json

import pandas as pd
import pytest

from scripts import publish_bond_history as pbh


@pytest.fixture
def history(tmp_path):
    src = tmp_path / 'history'
    src.mkdir()
    pd.DataFrame({
        'cusip': ['46647PDX1', '46647PDX1', '46647PDX1', 'ZZZ000000'],
        'report_date': pd.to_datetime(['2026-05-31', '2026-06-30', '2026-06-30', '2026-06-30']),
        'month': pd.to_datetime(['2026-05-31', '2026-06-30', '2026-06-30', '2026-06-30']),
        'clean_price_marked': [101.23456, 100.5, 100.6, 99.0],
    }).to_parquet(src / 'marks_monthly.parquet')
    pd.DataFrame({
        'price_date': pd.to_datetime(['2026-11-02', '2026-11-03']),
        'cusip': ['912828XX1', '912828XX1'],
        'end_of_day': [99.5, 99.75],
    }).to_parquet(src / 'treasury_daily.parquet')
    pd.DataFrame({
        'run_date': pd.to_datetime(['2026-11-21', '2026-11-20', '2026-11-21']),
        'cusip': ['46647PDX1', '46647PDX1', '46647PDX1'],
        'clean_price_est': [102.0, 101.9, float('nan')],
    }).to_parquet(src / 'estimates_daily.parquet')
    bonds = tmp_path / 'bonds.json'
    bonds.write_text(json.dumps({
        'run_date': '2026-11-21',
        'columns': [{'k': 'issuer_name'}, {'k': 'cusip'}],
        'rows': [['JPM', '46647PDX1'], ['UST', '912828XX1'], ['NONE', 'AB1234567']],
    }), encoding='utf-8')
    return src, bonds


def test_buckets_series_and_pruning(tmp_path, history):
    src, bonds = history
    dest = tmp_path / 'bond_px'
    dest.mkdir()
    (dest / 'ZZ.json').write_text('{}', encoding='utf-8')       # stale bucket
    (dest / '46 2.json').write_text('{}', encoding='utf-8')     # iCloud conflict copy
    assert pbh.main([str(bonds), str(dest), '--history-dir', str(src)]) == 0
    assert sorted(p.name for p in dest.iterdir()) == ['46.json', '91.json', 'meta.json']
    jpm = json.loads((dest / '46.json').read_text(encoding='utf-8'))
    assert jpm['v'] == pbh.VERSION and jpm['generated'] == '2026-11-21'
    b = jpm['bonds']['46647PDX1']
    # One point a day (the later duplicate wins), oldest first, 3 decimals.
    assert b['mark'] == [['2026-05-31', 101.235], ['2026-06-30', 100.6]]
    # NaN dropped, sorted by date.
    assert b['est'] == [['2026-11-20', 101.9], ['2026-11-21', 102.0]]
    assert 'tsy' not in b
    ust = json.loads((dest / '91.json').read_text(encoding='utf-8'))['bonds']['912828XX1']
    assert ust == {'tsy': [['2026-11-02', 99.5], ['2026-11-03', 99.75]]}
    # A bond outside bonds.json (ZZZ000000) is not published.
    assert 'ZZZ000000' not in json.dumps(jpm)


def test_missing_history_removes_folder(tmp_path, history):
    _, bonds = history
    dest = tmp_path / 'bond_px'
    dest.mkdir()
    (dest / '46.json').write_text('{}', encoding='utf-8')
    assert pbh.main([str(bonds), str(dest), '--history-dir', str(tmp_path / 'nope')]) == 0
    assert not dest.exists()


def test_unreadable_table_is_best_effort(tmp_path, history):
    src, bonds = history
    (src / 'marks_monthly.parquet').write_text('not parquet', encoding='utf-8')
    dest = tmp_path / 'bond_px'
    assert pbh.main([str(bonds), str(dest), '--history-dir', str(src)]) == 0
    assert not dest.exists()


def test_bucket_is_first_two_characters():
    assert pbh.bucket('46647pdx1') == '46'


def test_thin_weekly_keeps_last_year_daily():
    from datetime import date, timedelta
    start = date(2019, 1, 1)
    pts = [[(start + timedelta(days=i)).isoformat(), 100.0 + i / 1000] for i in range(2000)]
    out = pbh.thin_weekly(pts)
    end = date.fromisoformat(pts[-1][0])
    cut = (end - timedelta(days=pbh.DAILY_DAYS)).isoformat()
    recent = [p for p in out if p[0] >= cut]
    assert recent == [p for p in pts if p[0] >= cut]          # last year untouched
    old = [p for p in out if p[0] < cut]
    weeks = {date.fromisoformat(p[0]).isocalendar()[:2] for p in old}
    assert len(old) == len(weeks)                              # one a week
    assert out == sorted(out)                                  # oldest first
    assert pbh.thin_weekly([]) == []


def test_oversize_bucket_is_skipped(tmp_path, capsys):
    dest = tmp_path / 'bond_px'
    bonds = {'46647PDX1': {'mark': [['2026-05-31', 101.0]]}, '912828XX1': {'tsy': [['2026-11-02', 99.5]] * 50}}
    n = pbh.write(bonds, str(dest), '2026-11-21', max_file_bytes=200)
    assert n == 1 and sorted(p.name for p in dest.iterdir()) == ['46.json', 'meta.json']
    assert 'not published' in capsys.readouterr().out


def test_findings_filters_flags_and_ages(tmp_path):
    """The bond-history session's caveats: marks outside 5..250 dropped,
    single-fund marks tagged, latest-mark flags, estimates carry their mark
    age, TIPS flagged real."""
    src = tmp_path / 'h'
    src.mkdir()
    pd.DataFrame({
        'cusip': ['AA0000001'] * 4 + ['912810TIP'],
        'report_date': pd.to_datetime(['2026-03-31', '2026-04-30', '2026-05-31', '2026-06-30', '2026-06-30']),
        'month': pd.to_datetime(['2026-03-31', '2026-04-30', '2026-05-31', '2026-06-30', '2026-06-30']),
        'clean_price_marked': [3.0, 99.0, 260.0, 98.0, 101.0],
        'n_funds': [1, 1, 5, 4, 6],
        'is_default': [False, False, False, True, False],
        'in_arrears': [False] * 5,
        'is_convertible': [False] * 5,
    }).to_parquet(src / 'marks_monthly.parquet')
    pd.DataFrame({'price_date': pd.to_datetime(['2026-06-30']), 'cusip': ['912810TIP'],
                  'security_type': ['TIPS'], 'end_of_day': [101.01]}).to_parquet(src / 'treasury_daily.parquet')
    pd.DataFrame({'run_date': pd.to_datetime(['2026-11-20']), 'cusip': ['AA0000001'],
                  'clean_price_est': [97.5], 'mark_age_days': [143.0]}).to_parquet(src / 'estimates_daily.parquet')
    bonds = pbh.build(str(src), {'AA0000001', '912810TIP'})
    a = bonds['AA0000001']
    assert a['mark'] == [['2026-04-30', 99.0, 1], ['2026-06-30', 98.0]]   # 3 and 260 dropped
    assert a['flags'] == ['default']
    assert a['est'] == [['2026-11-20', 97.5, 143]]
    assert bonds['912810TIP']['flags'] == ['real']


def test_validation_compares_same_day_treasury_marks():
    n = 40
    days = pd.date_range('2024-01-31', periods=n, freq='ME')
    marks = pd.DataFrame({'cusip': ['912828AAA'] * n, 'report_date': days,
                          'clean_price_marked': [100.0 + (0.02 if i % 2 else -0.01) for i in range(n)],
                          'n_funds': [5] * n})
    tsy = pd.DataFrame({'cusip': ['912828AAA'] * n, 'price_date': days, 'end_of_day': [100.0] * n})
    v = pbh.validation(marks, tsy)
    assert v['n'] == n and v['median_abs'] == pytest.approx(0.015, abs=0.006) and v['min_funds'] == 3
    # Too few funds, or too few pairs: nothing to say.
    assert pbh.validation(marks.assign(n_funds=1), tsy) is None
    assert pbh.validation(marks.head(5), tsy) is None
