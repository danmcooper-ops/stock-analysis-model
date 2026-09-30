# tests/test_run_daily.py
"""scripts/run_daily.sh, the Mac nightly pipeline (scheduled-tasks/MAC-MINI-SETUP.md).

The steps it gained from the cloud run.sh: the plan they appear in, the
.env loader, and the data/snapshots archive, which now shares its branch with
the weekly backtest's own clone and so has to survive a moved tip."""

import os
import re
import shutil
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_DAILY = os.path.join(REPO, 'scripts', 'run_daily.sh')

pytestmark = pytest.mark.skipif(shutil.which('bash') is None or shutil.which('git') is None,
                                reason='needs bash and git')


def _src():
    return open(RUN_DAILY, encoding='utf-8').read()


def _fn(name):
    m = re.search(rf'^{name}\(\) {{\n.*?^}}\n', _src(), re.S | re.M)
    assert m, f'{name}() not found in run_daily.sh'
    return m.group(0)


def _dry_run(tmp_path, env_file=None, extra_env=None):
    """The --dry-run plan, as the list of step names it would run."""
    # macOS `mktemp -t prefix`; GNU mktemp wants a template.
    shim = tmp_path / 'bin'
    shim.mkdir()
    real_mktemp = shutil.which('mktemp')
    (shim / 'mktemp').write_text(
        f'#!/bin/bash\n[ "$1" = -t ] && exec {real_mktemp} "${{TMPDIR:-/tmp}}/$2.XXXXXX"\n'
        f'exec {real_mktemp} "$@"\n', encoding='utf-8')
    (shim / 'mktemp').chmod(0o755)
    repo = tmp_path / 'repo'
    (repo / 'scripts').mkdir(parents=True)
    shutil.copy(RUN_DAILY, repo / 'scripts' / 'run_daily.sh')
    subprocess.run(['git', 'init', '-q', '-b', 'main', str(repo)], check=True)
    if env_file is not None:
        (repo / '.env').write_text(env_file, encoding='utf-8')
    base = {k: v for k, v in os.environ.items()
            if not k.startswith(('SUPABASE_', 'CLOUDFLARE_', 'CF_', 'SEC_EMAIL', 'DB_'))}
    env = {**base, 'PATH': f'{shim}:{os.environ["PATH"]}', 'HOME': str(tmp_path),
           'PYTHON': shutil.which('python3') or 'python3', **(extra_env or {})}
    r = subprocess.run(['bash', str(repo / 'scripts' / 'run_daily.sh'), '--dry-run',
                        '--date', '2026-09-25'],
                       env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    return re.findall(r'^--- \[(\w+)\]', r.stdout, re.M)


def test_plan_without_secrets_matches_the_old_pipeline_plus_topup(tmp_path):
    steps = _dry_run(tmp_path)
    for absent in ('sec_cache_save', 'price_cache_save', 'db_check', 'publish_cloudflare'):
        assert absent not in steps
    # db_publish always runs (and skips itself), as step 06a does in the cloud.
    order = ['analyze', 'enrich_pipeline', 'prices_topup', 'render', 'portfolio_alerts',
             'db_publish', 'archive_sync', 'archive_snapshot', 'archive_add',
             'archive_commit', 'archive_push', 'store_check', 'publish', 'compact_output']
    assert [s for s in steps if s in order] == order


def test_plan_with_secrets_in_env_file_adds_the_cloud_steps(tmp_path):
    steps = _dry_run(tmp_path, env_file=(
        'SUPABASE_URL=https://x.supabase.co\nSUPABASE_SERVICE_ROLE_KEY=k\n'
        'CLOUDFLARE_API_TOKEN=t\nCLOUDFLARE_ACCOUNT_ID=a\nCF_PAGES_PROJECT=p\n'))
    for present in ('sec_cache_save', 'price_cache_save', 'db_check', 'publish_cloudflare'):
        assert present in steps
    assert steps.index('price_cache_save') < steps.index('render')
    assert steps.index('db_publish') < steps.index('archive_snapshot')


def test_env_loader_keeps_the_environment_and_quotes_values(tmp_path):
    src = _src()
    m = re.search(r'^if \[ -f \.env \]; then\n.*?^fi\n', src, re.S | re.M)
    assert m
    (tmp_path / '.env').write_text(
        '# comment\nSEC_EMAIL = a@b.c \nTIINGO_API_KEY=from-file\n'
        'ODD=has spaces; $(echo injected)\nnot a line\n', encoding='utf-8')
    script = (f'VPY={shutil.which("python3") or "python3"}\n' + m.group(0)
              + 'printf "%s|%s|%s" "$SEC_EMAIL" "$TIINGO_API_KEY" "$ODD"\n')
    env = {k: v for k, v in os.environ.items() if k not in ('SEC_EMAIL', 'ODD')}
    env['TIINGO_API_KEY'] = 'from-env'
    r = subprocess.run(['bash', '-c', script], cwd=tmp_path, env=env,
                       capture_output=True, text=True, timeout=30)
    assert r.stdout == 'a@b.c|from-env|has spaces; $(echo injected)', r.stderr


def _git(*args, cwd):
    subprocess.run(['git', *args], cwd=cwd, check=True, capture_output=True, text=True)


def _snapshots_remote(tmp_path):
    """A bare remote holding data/snapshots, and the Mac's worktree of it."""
    remote = tmp_path / 'remote.git'
    _git('init', '-q', '--bare', str(remote), cwd=tmp_path)
    seed = tmp_path / 'seed'
    _git('init', '-q', '-b', 'data/snapshots', str(seed), cwd=tmp_path)
    for k, v in (('user.name', 't'), ('user.email', 't@example.com')):
        _git('config', k, v, cwd=seed)
    (seed / 'results_2026-09-24.json.gz').write_text('old', encoding='utf-8')
    _git('add', '-A', cwd=seed)
    _git('commit', '-q', '-m', 'Snapshot: 2026-09-24', cwd=seed)
    _git('push', '-q', str(remote), 'data/snapshots', cwd=seed)
    wt = tmp_path / 'snapshots-data'
    _git('clone', '-q', '-b', 'data/snapshots', str(remote), str(wt), cwd=tmp_path)
    for k, v in (('user.name', 't'), ('user.email', 't@example.com')):
        _git('config', k, v, cwd=wt)
    return remote, seed, wt


def _archive(tmp_path, wt, rundate='2026-09-25', sync=True):
    """Run the archive's add/commit/push with run_daily.sh's own functions."""
    work = tmp_path / 'work'
    (work / 'output').mkdir(parents=True)
    (work / 'data' / 'cache').mkdir(parents=True)
    (work / 'output' / 'rating_history.json').write_text('{"h": 1}', encoding='utf-8')
    (work / 'output' / 'portfolio_alerts.json').write_text('{"a": 1}', encoding='utf-8')
    (work / 'data' / 'cache' / 'screen_skip.json').write_text('{}', encoding='utf-8')  # too small
    (wt / f'results_{rundate}.json.gz').write_text('new', encoding='utf-8')
    (wt / 'blobs').mkdir(exist_ok=True)
    (wt / 'blobs' / 'x.json.gz').write_text('b', encoding='utf-8')
    script = ('set -uo pipefail\n'
              f'SNAP_WT="{wt}"; RUNDATE={rundate}\n'
              + _fn('archive_sync') + _fn('archive_state_files') + _fn('archive_push')
              + 'archive_sync || echo "sync failed"\n'
              + 'archive_state_files || exit 11\n'
              + f'git -C "$SNAP_WT" commit -q -m "Snapshot: {rundate}" || exit 12\n'
              + 'archive_push || exit 13\n')
    # archive_push sleeps between attempts; keep the test fast.
    script = script.replace('sleep $(( i * 10 ))', 'sleep 0')
    if not sync:
        # A commit that lands between the sync and the push.
        script = script.replace('git -C "$SNAP_WT" pull -q --ff-only origin data/snapshots', 'true')
    return subprocess.run(['bash', '-c', script], cwd=work, capture_output=True,
                          text=True, timeout=60)


def _remote_tree(seed, remote):
    _git('fetch', '-q', str(remote), 'data/snapshots', cwd=seed)
    out = subprocess.run(['git', 'ls-tree', '-r', '--name-only', 'FETCH_HEAD'], cwd=seed,
                         check=True, capture_output=True, text=True).stdout.split()
    log = subprocess.run(['git', 'log', '--format=%s', 'FETCH_HEAD'], cwd=seed,
                         check=True, capture_output=True, text=True).stdout.splitlines()
    return out, log


def test_archive_commits_the_state_files_with_the_snapshot(tmp_path):
    remote, seed, wt = _snapshots_remote(tmp_path)
    r = _archive(tmp_path, wt)
    assert r.returncode == 0, r.stdout + r.stderr
    files, log = _remote_tree(seed, remote)
    assert {'results_2026-09-25.json.gz', 'blobs/x.json.gz', 'rating_history.json',
            'portfolio_alerts.json'} <= set(files)
    assert 'screen_skip.json' not in files          # under the 10 KB guard
    assert log[0] == 'Snapshot: 2026-09-25'


def test_archive_survives_a_weekly_commit_landing_between_sync_and_push(tmp_path):
    remote, seed, wt = _snapshots_remote(tmp_path)
    # The weekly backtest pushes from its own clone after the worktree synced:
    # the sync is skipped, so the push meets the moved tip.
    (seed / 'backtest_summary_2026-09-27.json').write_text('{}', encoding='utf-8')
    _git('add', '-A', cwd=seed)
    _git('commit', '-q', '-m', 'Weekly backtest: 2026-09-27', cwd=seed)
    _git('push', '-q', str(remote), 'data/snapshots', cwd=seed)
    r = _archive(tmp_path, wt, sync=False)
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'rebasing on the remote tip' in r.stdout
    files, log = _remote_tree(seed, remote)
    assert 'backtest_summary_2026-09-27.json' in files and 'results_2026-09-25.json.gz' in files
    assert log[:2] == ['Snapshot: 2026-09-25', 'Weekly backtest: 2026-09-27']


def test_archive_sync_fast_forwards_a_behind_worktree(tmp_path):
    remote, seed, wt = _snapshots_remote(tmp_path)
    (seed / 'backtest_summary_2026-09-27.json').write_text('{}', encoding='utf-8')
    _git('add', '-A', cwd=seed)
    _git('commit', '-q', '-m', 'Weekly backtest: 2026-09-27', cwd=seed)
    _git('push', '-q', str(remote), 'data/snapshots', cwd=seed)
    r = _archive(tmp_path, wt)
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'rebasing' not in r.stdout             # the sync already caught up
    _, log = _remote_tree(seed, remote)
    assert log[:2] == ['Snapshot: 2026-09-25', 'Weekly backtest: 2026-09-27']
