# tests/test_backtest_cloud.py
"""Helpers of the weekly backtest's cloud Routine (scripts/backtest_cloud.py)."""

import json
import os
import subprocess
from datetime import date

import pandas as pd
import pytest

from scripts import backtest_cloud as bc


def test_pick_snapshots_prefers_plain_and_filters():
    names = ['results_2026-07-01.json.gz', 'results_2026-07-06.json.gz',
             'results_2026-07-06.json', 'results_2026-08-01.json.gz',
             'results_2026-09-20.json.gz', 'rating_history.json',
             'results_2026-07-07_replay.json']
    assert bc.pick_snapshots(names, since='2026-07-06') == [
        'results_2026-07-06.json', 'results_2026-08-01.json.gz',
        'results_2026-09-20.json.gz']
    assert bc.pick_snapshots(names, since='2026-07-06', matured_days=30,
                             today=date(2026, 9, 26), newest=1) == [
        'results_2026-08-01.json.gz']


def _parquet(path, end):
    idx = pd.date_range('2026-06-01', end, freq='B')
    pd.DataFrame({'Close': 1.0}, index=idx.rename('Date')).to_parquet(path)


def test_check_prices_requires_current_benchmark_and_coverage(tmp_path):
    _parquet(tmp_path / 'SPY.parquet', '2026-09-25')
    for t in ['A', 'B', 'C']:
        _parquet(tmp_path / f'{t}.parquet', '2026-09-25')
    ok, lines = bc.check_prices(['SPY', 'A', 'B', 'C'], str(tmp_path),
                                today=date(2026, 9, 27))
    assert ok, lines
    ok, lines = bc.check_prices(['SPY', 'A', 'B', 'C', 'D'], str(tmp_path),
                                today=date(2026, 9, 27))
    assert not ok and any('4 of 5' in ln and ln.startswith('PROBLEM') for ln in lines)
    ok, lines = bc.check_prices(['SPY', 'A'], str(tmp_path), today=date(2026, 10, 9))
    assert not ok and lines[0].startswith('PROBLEM: SPY')


def test_check_prices_with_a_kept_cache_ignores_stale_unrefreshed_files(tmp_path):
    # A Mac keeps the price directory between Sundays. A file left from last
    # week that tonight's download failed to refresh must not count towards
    # the coverage floor, or a throttled night would pass the gate.
    _parquet(tmp_path / 'SPY.parquet', '2026-09-25')
    _parquet(tmp_path / 'CUR.parquet', '2026-09-25')     # current: skipped by the download
    _parquet(tmp_path / 'NEW.parquet', '2026-08-14')     # rewritten tonight (a delisted name)
    _parquet(tmp_path / 'OLD.parquet', '2026-09-18')     # last week's, not refreshed
    start = 1_000_000.0
    for t in ['SPY', 'CUR', 'OLD']:
        os.utime(tmp_path / f'{t}.parquet', (start - 3600, start - 3600))
    os.utime(tmp_path / 'NEW.parquet', (start + 60, start + 60))
    tickers = ['SPY', 'CUR', 'NEW', 'OLD']
    ok, lines = bc.check_prices(tickers, str(tmp_path), today=date(2026, 9, 27),
                                min_share=0.9, fresh_since=start)
    assert not ok
    assert any(ln.startswith('PROBLEM: 3 of 4') for ln in lines), lines
    assert any('1 stale file(s)' in ln for ln in lines), lines
    # Without fresh_since (the cloud's cold directory) every file counts.
    ok, _ = bc.check_prices(tickers, str(tmp_path), today=date(2026, 9, 27), min_share=0.9)
    assert ok


def test_matured_tickers_reads_only_matured_snapshots(tmp_path):
    for d, tickers in [('2026-08-01', ['OLD']), ('2026-09-20', ['NEW'])]:
        (tmp_path / f'results_{d}.json').write_text(json.dumps(
            {'date': d, 'results': [{'ticker': t} for t in tickers]}), encoding='utf-8')
    got = bc.matured_tickers(str(tmp_path), 30, today=date(2026, 9, 26))
    assert 'OLD' in got and 'NEW' not in got and 'SPY' in got


def _summary(**over):
    s = {'snapshots': ['2026-07-06', '2026-07-07'],
         'skipped_snapshots': [['2026-07-01', 'dated before 2026-07-06']],
         'unmeasured': [], 'coverage': [
             {'run_date': '2026-07-06', 'horizon': 30, 'coverage': 0.97}],
         'provenance': {'min_return_coverage': 0.9}}
    s.update(over)
    return s


def test_compare_clean_week():
    assert bc.compare_summaries(_summary(), _summary()) == []


def test_compare_flags_every_regression():
    cur = _summary(
        snapshots=['2026-07-06'],
        skipped_snapshots=[['2026-07-01', 'dated before 2026-07-06'],
                           ['2026-09-22', 'missing gate fields: x']],
        unmeasured=[{'run_date': '2026-08-01', 'horizon': 30,
                     'reason': 'no benchmark return'}],
        coverage=[{'run_date': '2026-07-06', 'horizon': 30, 'coverage': 0.5}])
    out = bc.compare_summaries(cur, _summary())
    assert len(out) == 4
    assert any('2026-07-07' in ln for ln in out)
    assert any('newly skipped snapshot 2026-09-22' in ln for ln in out)
    assert any('unmeasured' in ln for ln in out)
    assert any('50.0%' in ln for ln in out)


def test_compare_without_prior_still_checks_the_current_week():
    assert bc.compare_summaries(_summary(), None) == []
    low = _summary(coverage=[{'run_date': '2026-07-06', 'horizon': 30, 'coverage': 0.2}])
    assert len(bc.compare_summaries(low, None)) == 1


def test_compare_against_pre_provenance_summary_sets_a_baseline():
    prior = {'snapshots': ['2026-07-06']}                # an old-format summary
    assert bc.compare_summaries(_summary(), prior) == []


def test_prior_summary_is_the_newest_earlier_one(tmp_path):
    for d in ['2026-09-13', '2026-09-20', '2026-09-27']:
        (tmp_path / f'backtest_summary_{d}.json').write_text('{}', encoding='utf-8')
    cur = str(tmp_path / 'backtest_summary_2026-09-27.json')
    assert bc._prior_summary_path(cur).endswith('backtest_summary_2026-09-20.json')
    assert bc._prior_summary_path(str(tmp_path / 'backtest_summary_2026-09-13.json')) is None


def _git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args], check=True,
                          capture_output=True, text=True).stdout


@pytest.fixture
def archive(tmp_path):
    """A tiny data/snapshots-shaped repo, cloned blob-less like the routine."""
    src = tmp_path / 'src'
    src.mkdir()
    _git(src, 'init', '-q', '-b', 'data/snapshots')
    for d in ['2026-07-01', '2026-07-06', '2026-08-03']:
        (src / f'results_{d}.json').write_text(json.dumps(
            {'date': d, 'results': [{'ticker': 'AAA'}]}), encoding='utf-8')
    (src / 'returns').mkdir()
    (src / 'returns' / '2026-07-06_h30.json').write_text('{"coverage": 1.0}', encoding='utf-8')
    (src / 'backtest_summary_2026-09-20.json').write_text('{"snapshots": []}', encoding='utf-8')
    (src / 'rating_history.json').write_text('{}', encoding='utf-8')
    _git(src, 'add', '-A')
    _git(src, '-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-q', '-m', 'seed')
    _git(src, 'config', 'uploadpack.allowFilter', 'true')
    _git(src, 'config', 'uploadpack.allowAnySHA1InWant', 'true')
    clone = tmp_path / 'clone'
    subprocess.run(['git', 'clone', '-q', '--filter=blob:none', '--no-checkout',
                    '--single-branch', '-b', 'data/snapshots', f'file://{src}', str(clone)],
                   check=True)
    return clone


def test_stage_materializes_corpus_returns_and_prior_summary(archive, tmp_path):
    dest = tmp_path / 'out'
    picked = bc.stage(str(archive), str(dest), since='2026-07-06', log=lambda *_: None)
    assert picked == ['results_2026-07-06.json', 'results_2026-08-03.json']
    assert sorted(os.listdir(dest)) == ['backtest_summary_2026-09-20.json', 'results_2026-07-06.json',
                                        'results_2026-08-03.json', 'returns']
    assert (dest / 'returns' / '2026-07-06_h30.json').read_text(encoding='utf-8') == '{"coverage": 1.0}'
    assert json.loads((dest / 'results_2026-08-03.json').read_text(encoding='utf-8'))['date'] == '2026-08-03'


def test_stage_refuses_an_empty_selection(archive, tmp_path):
    with pytest.raises(OSError):
        bc.stage(str(archive), str(tmp_path / 'out'), since='2027-01-01', log=lambda *_: None)
