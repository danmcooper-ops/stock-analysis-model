# tests/test_weekly_mac_run.py
"""scheduled-tasks/weekly-backtest/mac_run.sh: the lock, signals and .env.

Run against a fake repo whose run.sh is a stub, so nothing real is launched.
The 2026-10-16 review found that killing mac_run.sh released the lock while
run.sh kept going (a relaunch then ran a second copy over the first one's work
directory), that a stale lock whose PID was reused read as a live run, and
that the whole of .env was exported to the job.
"""

import importlib.util
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAC_RUN = ROOT / 'scheduled-tasks' / 'weekly-backtest' / 'mac_run.sh'
ENV_KEYS = ROOT / 'scheduled-tasks' / 'weekly-backtest' / 'env_keys.py'

pytestmark = pytest.mark.skipif(sys.platform == 'win32', reason='bash script')

STUB = """#!/bin/bash
echo "stub pid $$ RUNDATE=$RUNDATE TIINGO=${TIINGO_API_KEY:-} SUPA=${SUPABASE_SERVICE_ROLE_KEY:-}" > "$STOCK_MODEL_WORK/stub.out"
sleep 60 &
echo $! > "$STOCK_MODEL_WORK/grandchild.pid"
wait
"""


@pytest.fixture
def fake(tmp_path):
    repo = tmp_path / 'repo'
    (repo / 'scheduled-tasks' / 'weekly-backtest').mkdir(parents=True)
    (repo / 'scheduled-tasks' / 'cloud-weekly-backtest').mkdir(parents=True)
    shutil.copy(MAC_RUN, repo / 'scheduled-tasks' / 'weekly-backtest' / 'mac_run.sh')
    shutil.copy(ENV_KEYS, repo / 'scheduled-tasks' / 'weekly-backtest' / 'env_keys.py')
    stub = repo / 'scheduled-tasks' / 'cloud-weekly-backtest' / 'run.sh'
    stub.write_text(STUB, encoding='utf-8')
    (repo / '.env').write_text(
        'TIINGO_API_KEY=first\nSUPABASE_SERVICE_ROLE_KEY=secret\nTIINGO_API_KEY=second\n'
        'DRY_RUN=1\n', encoding='utf-8')
    work, logs = tmp_path / 'work', tmp_path / 'logs'
    env = {k: v for k, v in os.environ.items()
           if k not in ('TIINGO_API_KEY', 'SUPABASE_SERVICE_ROLE_KEY', 'DRY_RUN', 'RUNDATE')}
    env.update(STOCK_MODEL_REPO=str(repo), STOCK_MODEL_WORK=str(work),
               LOGDIR=str(logs), PYTHON3=sys.executable, SSL_CERT_FILE='/dev/null')
    return repo, work, env


def _launch(repo, env):
    return subprocess.Popen(['bash', str(repo / 'scheduled-tasks' / 'weekly-backtest' / 'mac_run.sh')],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)


def _wait_for(path, timeout=10):
    t0 = time.time()
    while not path.exists() or not path.read_text(encoding='utf-8').strip():
        if time.time() - t0 > timeout:
            raise AssertionError(f'{path} never appeared')
        time.sleep(0.05)


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # a zombie still answers kill -0; ask ps for its state
    st = subprocess.run(['ps', '-o', 'stat=', '-p', str(pid)], capture_output=True,
                        text=True).stdout.strip()
    return bool(st) and not st.startswith('Z')


def test_term_stops_the_whole_run_before_releasing_the_lock(fake):
    repo, work, env = fake
    p = _launch(repo, env)
    _wait_for(work / 'grandchild.pid')
    lock = work / '.weekly.lock'
    assert lock.read_text(encoding='utf-8').strip() == str(p.pid)
    grandchild = int((work / 'grandchild.pid').read_text(encoding='utf-8'))
    assert _alive(grandchild)
    p.send_signal(signal.SIGTERM)
    out, _ = p.communicate(timeout=15)
    assert p.returncode == 143, out
    assert 'caught TERM' in out
    time.sleep(0.2)
    assert not _alive(grandchild)          # nothing of the run survives
    assert not lock.exists()


def test_a_second_launch_refuses_while_the_first_runs(fake):
    repo, work, env = fake
    p = _launch(repo, env)
    try:
        _wait_for(work / 'grandchild.pid')
        second = subprocess.run(['bash', str(repo / 'scheduled-tasks' / 'weekly-backtest' / 'mac_run.sh')],
                                env=env, capture_output=True, text=True, timeout=15)
        assert second.returncode == 1
        assert 'ALREADY RUNNING' in second.stderr
        assert (work / '.weekly.lock').read_text(encoding='utf-8').strip() == str(p.pid)
    finally:
        p.send_signal(signal.SIGTERM)
        p.communicate(timeout=15)


def test_a_stale_lock_held_by_another_live_process_is_taken_over(fake):
    """A reused PID: alive, but not a mac_run.sh (here, this test's own PID)."""
    repo, work, env = fake
    work.mkdir(parents=True)
    (work / '.weekly.lock').write_text(str(os.getpid()), encoding='utf-8')
    p = _launch(repo, env)
    try:
        _wait_for(work / 'grandchild.pid')
        assert (work / '.weekly.lock').read_text(encoding='utf-8').strip() == str(p.pid)
    finally:
        p.send_signal(signal.SIGTERM)
        out, _ = p.communicate(timeout=15)
    assert 'removing a stale lock' in out


def test_only_the_named_env_keys_reach_the_job_and_rundate_is_fixed(fake):
    repo, work, env = fake
    env['RUNDATE'] = '2026-10-18'
    p = _launch(repo, env)
    try:
        _wait_for(work / 'stub.out')
        line = (work / 'stub.out').read_text(encoding='utf-8')
    finally:
        p.send_signal(signal.SIGTERM)
        out, _ = p.communicate(timeout=15)
    assert 'RUNDATE=2026-10-18' in line
    assert 'TIINGO=first ' in line          # first occurrence wins
    assert line.rstrip().endswith('SUPA=')   # not exported
    assert (work.parent / 'logs' / 'weekly_2026-10-18.log').exists()


def test_env_keys_quoting_and_precedence():
    spec = importlib.util.spec_from_file_location('env_keys', ENV_KEYS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    lines = ['# c', 'export TIINGO_API_KEY="a b"', 'TIINGO_API_KEY=later',
             'X=$(rm -rf ~)', 'BAD KEY=1']
    assert mod.export_lines(lines, ['TIINGO_API_KEY', 'X'], environ={}) == [
        "export TIINGO_API_KEY='a b'", "export X='$(rm -rf ~)'"]
    assert mod.export_lines(lines, ['TIINGO_API_KEY'],
                            environ={'TIINGO_API_KEY': 'env'}) == []
