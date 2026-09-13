import gzip
import os

import pytest

from data.snapshot_store import list_snapshot_files, read_snapshot
from scripts.compact_output import gzip_verified, main, plan


def _snap(directory, day, body=None):
    os.makedirs(directory, exist_ok=True)
    p = os.path.join(directory, f'results_{day}.json')
    with open(p, 'w', encoding='utf-8') as f:
        f.write(body or '{"date":"%s","results":[{"ticker":"A"}]}' % day)
    return p


def _touch(directory, name):
    p = os.path.join(directory, name)
    with open(p, 'w', encoding='utf-8') as f:
        f.write('x' * 100)
    return p


def test_plan_keeps_newest_snapshots_plain(tmp_path):
    out = str(tmp_path)
    for d in ('2026-09-01', '2026-09-02', '2026-09-03', '2026-09-04'):
        _snap(out, d)
    todo = [os.path.basename(p) for p in plan(out, keep_plain=2)]
    assert todo == ['results_2026-09-01.json', 'results_2026-09-02.json']


def test_plan_ignores_already_gzipped_and_non_canonical(tmp_path):
    out = str(tmp_path)
    gzip_verified(_snap(out, '2026-09-01'))
    _snap(out, '2026-09-02')
    _touch(out, 'results_2026-09-02_replay.json')
    assert plan(out, keep_plain=0) == [os.path.join(out, 'results_2026-09-02.json')]


def test_plan_includes_retired_and_aged_sidecars(tmp_path):
    out = str(tmp_path)
    _snap(out, '2026-09-10')
    retired = _snap(os.path.join(out, 'retired'), '2026-07-03')
    old = [_touch(out, n) for n in ('events_2026-08-01.json',
                                    'stock_analysis_results_2026-08-01.html',
                                    'stock_analysis_results_2026-08-03_cssfix.html',
                                    'portfolio_report_2026-08-01.txt',
                                    'run_2026-08-01.log')]
    for n in ('events_2026-09-01.json', 'run_summary_2026-08-01.json',
              'backtest_2026-08-01.xlsx', 'stock_analysis_results_2026-08-01.xlsx'):
        _touch(out, n)
    assert sorted(plan(out, keep_plain=1, keep_days=30)) == sorted([retired] + old)


def test_gzip_verified_round_trips_and_preserves_mtime(tmp_path):
    p = _snap(str(tmp_path), '2026-09-01')
    with open(p, 'rb') as f:
        original = f.read()
    os.utime(p, (1_700_000_000, 1_700_000_000))
    gzip_verified(p)
    assert not os.path.exists(p)
    with gzip.open(p + '.gz', 'rb') as f:
        assert f.read() == original
    assert os.path.getmtime(p + '.gz') == 1_700_000_000
    assert read_snapshot(p + '.gz')['date'] == '2026-09-01'
    assert list_snapshot_files(str(tmp_path)) == [('2026-09-01', p + '.gz')]


def test_gzip_verified_is_deterministic(tmp_path):
    a = _snap(str(tmp_path / 'a'), '2026-09-01')
    b = _snap(str(tmp_path / 'b'), '2026-09-01')
    gzip_verified(a)
    gzip_verified(b)
    with open(a + '.gz', 'rb') as fa, open(b + '.gz', 'rb') as fb:
        assert fa.read() == fb.read()


def test_existing_identical_gz_just_removes_plain(tmp_path):
    p = _snap(str(tmp_path), '2026-09-01')
    with open(p, 'rb') as f, gzip.open(p + '.gz', 'wb') as g:
        g.write(f.read())
    gzip_verified(p)
    assert not os.path.exists(p)


def test_existing_different_gz_leaves_source(tmp_path):
    p = _snap(str(tmp_path), '2026-09-01')
    with gzip.open(p + '.gz', 'wb') as g:
        g.write(b'{"date":"other"}')
    with pytest.raises(OSError):
        gzip_verified(p)
    assert os.path.exists(p)


def test_main_dry_run_changes_nothing_then_apply(tmp_path, capsys):
    out = str(tmp_path)
    for d in ('2026-09-01', '2026-09-02'):
        _snap(out, d)
    assert main(['--results-dir', out, '--keep-plain', '1']) == 0
    assert os.path.exists(os.path.join(out, 'results_2026-09-01.json'))
    assert 'dry run' in capsys.readouterr().out
    assert main(['--results-dir', out, '--keep-plain', '1', '--apply']) == 0
    assert sorted(os.listdir(out)) == ['results_2026-09-01.json.gz',
                                       'results_2026-09-02.json']
