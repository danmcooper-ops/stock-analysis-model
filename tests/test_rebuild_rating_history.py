"""The rating-history cache rebuild (scripts/rebuild_rating_history.py) and
the loader's stop-at-unreadable-day rule (report_html._advance_rating_history_cache).

Both guard the same failure: a day the incremental cache never folds in,
because ``last_scanned`` moved past it. On 2026-09-14 the inline rebuild in
run.sh left 2026-04-21..08-25 out that way, which surfaced as 1,107 parity
mismatches against the database on 09-28.
"""
import json
import subprocess

import pytest

from scripts import rebuild_rating_history as rb
from scripts import report_html

# One ticker whose rating moves on days the staged set does not cover.
RATINGS = {
    '2026-04-20': 'HOLD',
    '2026-05-01': 'PASS',
    '2026-06-01': 'PASS',
    '2026-07-01': 'BUY',
    '2026-08-26': 'BUY',
    '2026-09-01': 'LEAN BUY',
    '2026-09-25': 'LEAN BUY',
}
TRUTH = [['2026-04-20', 'HOLD'], ['2026-05-01', 'PASS'],
         ['2026-07-01', 'BUY'], ['2026-09-01', 'LEAN BUY']]


@pytest.fixture(autouse=True)
def _no_database(monkeypatch):
    monkeypatch.delenv('SNAPSHOT_STORE_BACKEND', raising=False)


def _git(*args, cwd):
    subprocess.run(['git', *args], cwd=cwd, check=True, capture_output=True, text=True)


def _snapshot(date, rating=None):
    return json.dumps([{'ticker': 'X', 'rating': rating or RATINGS[date]}])


def _archive(tmp_path, files):
    """A git clone of an archive branch holding *files* ({name: text})."""
    repo = tmp_path / 'snapshots-data'
    _git('init', '-q', '-b', 'data/snapshots', str(repo), cwd=tmp_path)
    for k, v in (('user.name', 't'), ('user.email', 't@example.com')):
        _git('config', k, v, cwd=repo)
    for name, text in files.items():
        (repo / name).write_text(text, encoding='utf-8')
    _git('add', '-A', cwd=repo)
    _git('commit', '-q', '-m', 'archive', cwd=repo)
    return repo


def _cache(out):
    return json.loads((out / 'rating_history.json').read_text(encoding='utf-8'))


def test_archive_snapshots_prefers_plain_and_ignores_other_files():
    names = ['results_2026-09-24.json.gz', 'results_2026-09-24.json',
             'results_2026-09-25.json.gz', 'rating_history.json',
             'backtest_summary_2026-09-26.json', 'results_2026-09-2.json']
    assert rb.archive_snapshots(names) == {
        '2026-09-24': 'results_2026-09-24.json',
        '2026-09-25': 'results_2026-09-25.json.gz',
    }


def test_rebuild_covers_days_older_than_the_staged_ones(tmp_path):
    repo = _archive(tmp_path, {f'results_{d}.json': _snapshot(d) for d in RATINGS})
    out = tmp_path / 'output'
    out.mkdir()
    # run.sh step 02 stages the newest snapshots into output/ before the rebuild.
    for d in list(RATINGS)[-3:]:
        (out / f'results_{d}.json').write_text(_snapshot(d), encoding='utf-8')

    n, tickers = rb.rebuild(str(repo), str(out), log=lambda *_: None)

    assert (n, tickers) == (len(RATINGS), 1)
    c = _cache(out)
    assert c['last_scanned'] == '2026-09-25'
    assert c['hist']['X'] == TRUTH
    assert not list(out.glob('.rating_history.json.tmp'))


def test_rebuild_refuses_a_day_it_cannot_fold_in(tmp_path):
    files = {f'results_{d}.json': _snapshot(d) for d in RATINGS}
    files['results_2026-06-01.json'] = json.dumps({'meta': 'no results list'})
    repo = _archive(tmp_path, files)
    out = tmp_path / 'output'
    out.mkdir()
    (out / 'rating_history.json').write_text('{"kept": true}', encoding='utf-8')

    with pytest.raises(rb.RebuildError, match='results_2026-06-01.json was not folded in'):
        rb.rebuild(str(repo), str(out), log=lambda *_: None)
    assert json.loads((out / 'rating_history.json').read_text(encoding='utf-8')) == {'kept': True}


def test_cli_fails_on_a_corrupt_snapshot(tmp_path, capsys):
    files = {f'results_{d}.json': _snapshot(d) for d in RATINGS}
    files['results_2026-06-01.json'] = '{not json'
    repo = _archive(tmp_path, files)
    out = tmp_path / 'output'

    assert rb.main([str(repo), str(out)]) == 1
    assert 'rating-history rebuild failed' in capsys.readouterr().err
    assert not (out / 'rating_history.json').exists()


def test_loader_stops_at_an_unreadable_day_and_resumes_there(tmp_path, capsys):
    for d in ('2026-04-20', '2026-05-01', '2026-07-01'):
        (tmp_path / f'results_{d}.json').write_text(_snapshot(d), encoding='utf-8')
    (tmp_path / 'results_2026-05-01.json').write_text('{not json', encoding='utf-8')

    report_html._load_rating_history(str(tmp_path), None)
    c = _cache(tmp_path)
    # Scanning 07-01 past the unreadable 05-01 would lose 05-01 for good.
    assert c['last_scanned'] == '2026-04-20'
    assert c['hist']['X'] == [['2026-04-20', 'HOLD']]
    assert 'cache stops at 2026-04-20' in capsys.readouterr().out

    (tmp_path / 'results_2026-05-01.json').write_text(_snapshot('2026-05-01'), encoding='utf-8')
    report_html._load_rating_history(str(tmp_path), None)
    c = _cache(tmp_path)
    assert c['last_scanned'] == '2026-07-01'
    assert c['hist']['X'] == [['2026-04-20', 'HOLD'], ['2026-05-01', 'PASS'], ['2026-07-01', 'BUY']]
