# tests/test_mac_mini_scripts.py
"""scheduled-tasks/mac-mini/: the old-Mac pack and the new-Mac bootstrap.

Run against a fake home folder, so nothing touches the real one. The pack
must carry .env and never print a key's value. The bootstrap's --check mode
must report and change nothing."""

import os
import shutil
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACK = os.path.join(REPO, 'scheduled-tasks', 'mac-mini', 'pack_old_mac.sh')
BOOT = os.path.join(REPO, 'scheduled-tasks', 'mac-mini', 'bootstrap_mini.sh')

pytestmark = pytest.mark.skipif(shutil.which('bash') is None or shutil.which('git') is None,
                                reason='needs bash and git')


def _git(*args, cwd):
    subprocess.run(['git', *args], cwd=cwd, check=True, capture_output=True, text=True)


def _run(script, home, *args, **env):
    e = {k: v for k, v in os.environ.items()
         if not k.startswith(('SUPABASE_', 'SEC_EMAIL', 'TIINGO_'))}
    e.update(HOME=str(home), **env)
    return subprocess.run(['bash', script, *args], env=e, capture_output=True,
                          text=True, timeout=120)


@pytest.fixture
def old_mac(tmp_path):
    """A home folder with a checkout holding .env, caches and local work."""
    home = tmp_path / 'air'
    wt = home / 'Projects' / 'Workspace Folder'
    (wt / 'scripts').mkdir(parents=True)
    (wt / 'scripts' / 'analyze_stock.py').write_text('x', encoding='utf-8')
    (wt / '.gitignore').write_text('.env\noutput/\ndata/cache/\n', encoding='utf-8')
    _git('init', '-q', '-b', 'main', str(wt), cwd=tmp_path)
    for k, v in (('user.name', 't'), ('user.email', 't@example.com')):
        _git('config', k, v, cwd=wt)
    _git('add', '-A', cwd=wt)
    _git('commit', '-q', '-m', 'init', cwd=wt)
    (wt / 'uncommitted.py').write_text('y', encoding='utf-8')
    (wt / '.env').write_text('SEC_EMAIL=a@b.c\nTIINGO_API_KEY=sekrit-value\n', encoding='utf-8')
    (wt / 'output' / 'prices').mkdir(parents=True)
    (wt / 'output' / 'prices' / 'AAPL.parquet').write_bytes(b'p')
    sec = wt / 'data' / 'cache' / 'sec_facts'
    sec.mkdir(parents=True)
    (sec / '_state.json').write_text('{"last_index_sweep": "2026-09-08"}', encoding='utf-8')
    (sec / '0000320193.json.gz').write_bytes(b'b')
    # ~/Library is never searched; the Trash is.
    for d in (home / 'Library' / 'X', home / '.Trash' / 'Old'):
        (d / 'scripts').mkdir(parents=True)
        (d / 'scripts' / 'analyze_stock.py').write_text('x', encoding='utf-8')
    return home, wt


def test_pack_carries_env_privately_and_never_prints_values(old_mac):
    home, wt = old_mac
    r = _run(PACK, home)
    assert r.returncode == 0, r.stdout + r.stderr
    out = home / 'StockModelTransfer'
    env = out / 'secrets' / 'env'
    assert env.read_text(encoding='utf-8') == (wt / '.env').read_text(encoding='utf-8')
    assert oct(env.stat().st_mode & 0o777) == '0o600'
    assert oct(out.stat().st_mode & 0o777) == '0o700'
    report = (out / 'INVENTORY.txt').read_text(encoding='utf-8')
    assert 'sekrit-value' not in r.stdout + r.stderr + report
    assert 'SEC_EMAIL TIINGO_API_KEY' in report          # key names only
    assert 'uncommitted.py' in report and 'not on GitHub' in report
    assert '.Trash' in report and 'Library/X' not in report
    assert 'sweep watermark 2026-09-08' in report
    # caches are opt-in
    assert not (out / 'prices').exists() and not (out / 'sec_facts').exists()


def test_pack_caches_on_request(old_mac):
    home, _ = old_mac
    r = _run(PACK, home, '--with-sec-cache', '--with-prices')
    assert r.returncode == 0, r.stdout + r.stderr
    out = home / 'StockModelTransfer'
    assert (out / 'prices' / 'AAPL.parquet').exists()
    assert (out / 'sec_facts' / '_state.json').exists()


def test_pack_inventory_only_writes_nothing(old_mac):
    home, _ = old_mac
    r = _run(PACK, home, '--inventory-only')
    assert r.returncode == 0 and 'nothing written' in r.stdout
    assert not (home / 'StockModelTransfer').exists()


def test_pack_refuses_icloud_folders(old_mac):
    home, _ = old_mac
    r = _run(PACK, home, '--out', str(home / 'Desktop' / 'pack'))
    assert r.returncode == 2 and 'iCloud' in r.stderr
    assert not (home / 'Desktop').exists()


def test_bootstrap_check_reports_and_changes_nothing(tmp_path):
    remote = tmp_path / 'remote.git'
    _git('init', '-q', '--bare', str(remote), cwd=tmp_path)
    seed = tmp_path / 'seed'
    _git('init', '-q', '-b', 'main', str(seed), cwd=tmp_path)
    for k, v in (('user.name', 't'), ('user.email', 't@example.com')):
        _git('config', k, v, cwd=seed)
    (seed / 'README.md').write_text('x', encoding='utf-8')
    _git('add', '-A', cwd=seed)
    _git('commit', '-q', '-m', 'init', cwd=seed)
    _git('push', '-q', str(remote), 'main', cwd=seed)
    home = tmp_path / 'mini'
    (home / 'StockModelTransfer' / 'secrets').mkdir(parents=True)
    (home / 'StockModelTransfer' / 'secrets' / 'env').write_text('SEC_EMAIL=a@b.c\n', encoding='utf-8')
    before = sorted(p.relative_to(home) for p in home.rglob('*'))
    r = _run(BOOT, home, '--check', REPO_URL=str(remote), PYTHON3=sys.executable)
    assert r.returncode == 1, r.stdout + r.stderr       # things left to do
    assert 'ok    GitHub access' in r.stdout
    assert 'not done: clone into' in r.stdout
    assert 'not done: install .env from' in r.stdout
    assert sorted(p.relative_to(home) for p in home.rglob('*')) == before


def test_bootstrap_stops_without_repo_access(tmp_path):
    r = _run(BOOT, tmp_path, '--check', REPO_URL=str(tmp_path / 'nope.git'),
             PYTHON3=sys.executable)
    assert r.returncode == 1 and 'cannot read' in r.stdout


def test_bootstrap_check_changes_nothing_in_an_existing_checkout(tmp_path):
    """With a checkout, venv and snapshots worktree in place, --check reaches the
    seeding step; it must still create no folder and leave .env's mode alone."""
    home = tmp_path / 'mini'
    repo = home / 'Projects' / 'Workspace Folder'
    _git('init', '-q', '-b', 'main', str(repo), cwd=tmp_path)
    (repo / '.env').write_text('SEC_EMAIL=a@b.c\n', encoding='utf-8')
    os.chmod(repo / '.env', 0o644)
    (repo / '.claude' / 'worktrees' / 'snapshots-data').mkdir(parents=True)
    (repo / '.claude' / 'worktrees' / 'snapshots-data' / 'results_2026-09-29.json.gz').write_bytes(b'x')
    vbin = home / '.venvs' / 'stock-model' / 'bin'
    vbin.mkdir(parents=True)
    os.symlink(sys.executable, vbin / 'python')
    before = sorted(p.relative_to(home) for p in home.rglob('*'))
    r = _run(BOOT, home, '--check', REPO_URL=str(repo), PYTHON3=sys.executable)
    assert r.returncode == 1, r.stdout + r.stderr
    assert 'not done: chmod 600 .env' in r.stdout
    assert 'not done: copy the newest' in r.stdout
    assert sorted(p.relative_to(home) for p in home.rglob('*')) == before
    assert oct((repo / '.env').stat().st_mode & 0o777) == '0o644'


def test_bootstrap_rejects_a_missing_option_value(tmp_path):
    r = _run(BOOT, tmp_path, '--transfer')
    assert r.returncode == 2 and '--transfer needs a path' in r.stderr
