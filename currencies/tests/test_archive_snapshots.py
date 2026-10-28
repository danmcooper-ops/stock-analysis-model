"""The data/snapshots archive (scripts/archive_snapshots.py), offline.

Runs against a local bare repository standing in for origin, so it needs git
but no network. The properties that matter: the checkout's working tree and
branch are never touched, an unchanged day makes no commit, days pruned from
the local history stay on the branch, and a branch that moved during the push
is rebuilt on top of, not overwritten.
"""

import json
import os
import subprocess

import pytest

from scripts import archive_snapshots as ar

BRANCH = 'refs/heads/' + ar.BRANCH


def _git(cwd, *args):
    return subprocess.run(['git', '-C', str(cwd), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def _day(history, day, rating='LONG'):
    d = history / day
    d.mkdir(parents=True, exist_ok=True)
    (d / 'results.json').write_text(json.dumps(
        {'date': day, 'results': [{'code': 'JPY', 'rating': rating}]}))
    (d / 'manifest.json').write_text(json.dumps({'run_date': day}))


@pytest.fixture
def repos(tmp_path, monkeypatch):
    for var in ('NAME', 'EMAIL'):  # use the fallback identity, not the host's
        monkeypatch.delenv('GIT_AUTHOR_' + var, raising=False)
        monkeypatch.delenv('GIT_COMMITTER_' + var, raising=False)
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', os.devnull)
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')
    origin = tmp_path / 'origin.git'
    _git(tmp_path, 'init', '--quiet', '--bare', str(origin))
    repo = tmp_path / 'checkout'
    _git(tmp_path, 'init', '--quiet', '-b', 'main', str(repo))
    (repo / 'README.md').write_text('hello\n')
    _git(repo, 'add', 'README.md')
    _git(repo, '-c', 'user.name=t', '-c', 'user.email=t@t', 'commit',
         '--quiet', '-m', 'init')
    _git(repo, 'remote', 'add', 'origin', str(origin))
    _git(repo, 'push', '--quiet', 'origin', 'main')
    return repo, origin, tmp_path / 'history'


def _files(origin):
    return sorted(_git(origin, 'ls-tree', '-r', '--name-only',
                       BRANCH).splitlines())


def test_creates_the_branch_and_leaves_the_checkout_alone(repos):
    repo, origin, history = repos
    _day(history, '2026-10-15')
    _day(history, '2026-10-16')
    (repo / 'README.md').write_text('local edit\n')  # uncommitted work
    head = _git(repo, 'rev-parse', 'HEAD')

    assert ar.archive(str(repo), str(history), log=lambda m: None) == 0

    assert _files(origin) == ['2026-10-15/manifest.json',
                              '2026-10-15/results.json',
                              '2026-10-16/manifest.json',
                              '2026-10-16/results.json']
    assert _git(origin, 'log', '-1', '--format=%s|%an', BRANCH) == \
        'Snapshot: 2026-10-16 (2 days, 2026-10-15 .. 2026-10-16)|' + \
        ar.BOT_NAME
    assert _git(repo, 'rev-parse', 'HEAD') == head
    assert _git(repo, 'branch', '--show-current') == 'main'
    assert (repo / 'README.md').read_text() == 'local edit\n'
    assert _git(repo, 'status', '--porcelain') == 'M README.md'


def test_unchanged_history_makes_no_commit(repos):
    repo, origin, history = repos
    _day(history, '2026-10-15')
    ar.archive(str(repo), str(history), log=lambda m: None)
    tip = _git(origin, 'rev-parse', BRANCH)
    msgs = []
    assert ar.archive(str(repo), str(history), log=msgs.append) == 0
    assert _git(origin, 'rev-parse', BRANCH) == tip
    assert 'already holds' in msgs[-1]


def test_new_day_is_added_and_pruned_days_are_kept(repos):
    repo, origin, history = repos
    _day(history, '2026-10-15')
    ar.archive(str(repo), str(history), log=lambda m: None)
    (history / '2026-10-15' / 'results.json').unlink()  # pruned locally
    (history / '2026-10-15' / 'manifest.json').unlink()
    os.rmdir(history / '2026-10-15')
    _day(history, '2026-10-16', rating='SHORT')

    ar.archive(str(repo), str(history), log=lambda m: None)

    assert '2026-10-15/results.json' in _files(origin)
    assert '2026-10-16/results.json' in _files(origin)
    assert _git(origin, 'log', '-1', '--format=%s', BRANCH) == \
        'Snapshot: 2026-10-16'
    assert _git(origin, 'rev-list', '--count', BRANCH) == '2'


def test_branch_moved_elsewhere_is_built_on_not_overwritten(repos, tmp_path):
    repo, origin, history = repos
    _day(history, '2026-10-15')
    ar.archive(str(repo), str(history), log=lambda m: None)
    # Another machine adds a day the local history never had.
    other = tmp_path / 'other'
    _git(tmp_path, 'clone', '--quiet', '-b', ar.BRANCH, str(origin), str(other))
    (other / '2026-10-14').mkdir()
    (other / '2026-10-14' / 'results.json').write_text('{}')
    _git(other, 'add', '.')
    _git(other, '-c', 'user.name=t', '-c', 'user.email=t@t', 'commit',
         '--quiet', '-m', 'other')
    _git(other, 'push', '--quiet', 'origin', ar.BRANCH)
    _day(history, '2026-10-16')

    assert ar.archive(str(repo), str(history), log=lambda m: None) == 0
    assert '2026-10-14/results.json' in _files(origin)
    assert '2026-10-16/results.json' in _files(origin)


def test_no_push_builds_but_leaves_origin_alone(repos):
    repo, origin, history = repos
    _day(history, '2026-10-15')
    msgs = []
    assert ar.archive(str(repo), str(history), push=False,
                      log=msgs.append) == 0
    assert 'not pushed' in msgs[-1]
    with pytest.raises(subprocess.CalledProcessError):
        _git(origin, 'rev-parse', '--verify', BRANCH)


def test_empty_history_is_not_an_error(repos):
    repo, _, history = repos
    assert ar.archive(str(repo), str(history), log=lambda m: None) == 0


def test_commit_message():
    assert ar.commit_message(['2026-10-16']) == 'Snapshot: 2026-10-16'
    assert ar.commit_message(['2026-10-14', '2026-10-16']) == \
        'Snapshot: 2026-10-16 (2 days, 2026-10-14 .. 2026-10-16)'
