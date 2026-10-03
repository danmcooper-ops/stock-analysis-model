"""P4c: hist/ shards, the shard publisher, the Pages limit check, and the
Cloudflare publish step's skip path (design/supabase-migration.md)."""
import json
import os
import re
import shutil
import subprocess

import pytest

import scripts.publish_vol_shards as pvs
from scripts import check_pages_limits as cpl
from scripts.report_html import _build_hist_payload, _write_details_parts, _write_hist_shards

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_SH = os.path.join(REPO, 'scheduled-tasks', 'cloud-daily-stock-analysis', 'run.sh')


def _rows():
    return [
        {'ticker': 'AAA', 'edgar_history': {'revenue_history': {2022: 9e9, '2023-12-31': 10e9}}},
        {'ticker': 'BRK.B', 'edgar_history': {'earnings_history': {2023: 2e9}}},
        {'ticker': 'NOHIST'},
    ]


# --- hist shards ------------------------------------------------------------

def test_hist_shards_match_the_payload(tmp_path):
    payload = _build_hist_payload(_rows())
    hist_dir, index = tmp_path / 'hist', tmp_path / 'hist_index.json'
    got = _write_hist_shards(str(hist_dir), str(index), payload)
    assert got == sorted(payload) == ['AAA', 'BRK.B']
    for tk in got:
        shard = json.loads((hist_dir / f'{tk}.json').read_text(encoding='utf-8'))
        assert shard == json.loads(json.dumps(payload[tk]))
    assert json.loads(index.read_text(encoding='utf-8')) == {'tickers': got}
    assert sorted(os.listdir(hist_dir)) == ['AAA.json', 'BRK.B.json']


def test_hist_shards_are_rebuilt_and_skip_unsafe_names(tmp_path):
    hist_dir, index, legacy = tmp_path / 'hist', tmp_path / 'hist_index.json', tmp_path / 'hist.json'
    hist_dir.mkdir()
    (hist_dir / 'GONE.json').write_text('{}', encoding='utf-8')     # left the universe
    legacy.write_text('{}', encoding='utf-8')
    payload = {'OK': {'revenue': {}}, '../EVIL': {'x': 1}, 'A B': {'x': 1}}
    assert _write_hist_shards(str(hist_dir), str(index), payload, legacy_path=str(legacy)) == ['OK']
    assert os.listdir(hist_dir) == ['OK.json']
    assert not legacy.exists()
    assert not (tmp_path / 'EVIL.json').exists()


def test_no_hist_leaves_no_shards_or_index(tmp_path):
    hist_dir, index = tmp_path / 'hist', tmp_path / 'hist_index.json'
    _write_hist_shards(str(hist_dir), str(index), {'AAA': {}})
    assert _write_hist_shards(str(hist_dir), str(index), None) == []
    assert not hist_dir.exists() and not index.exists()


# --- details parts -----------------------------------------------------------

def test_details_parts_split_by_size_and_merge_back(tmp_path):
    payload = {f'T{i:03d}': {'news_headlines': ['x' * 900], 'n': i} for i in range(50)}
    ddir, index, legacy = tmp_path / 'details', tmp_path / 'details_index.json', tmp_path / 'details.json'
    legacy.write_text('{}', encoding='utf-8')
    parts = _write_details_parts(str(ddir), str(index), payload, legacy_path=str(legacy), part_bytes=10_000)
    assert len(parts) > 3 and parts == [str(i) for i in range(len(parts))]
    assert json.loads(index.read_text(encoding='utf-8')) == {'parts': parts}
    merged, order = {}, []
    for p in parts:
        chunk = json.loads((ddir / f'{p}.json').read_text(encoding='utf-8'))
        assert os.path.getsize(ddir / f'{p}.json') <= 10_000 + 1_000       # at most one ticker over
        assert not set(chunk) & set(merged)
        merged.update(chunk)
        order += list(chunk)
    assert merged == payload and order == sorted(payload)
    assert not legacy.exists()


def test_details_parts_rebuild_and_empty(tmp_path):
    ddir, index = tmp_path / 'details', tmp_path / 'details_index.json'
    _write_details_parts(str(ddir), str(index), {f'T{i}': {'x': 'y' * 100} for i in range(20)}, part_bytes=500)
    assert _write_details_parts(str(ddir), str(index), {'A': {'x': 1}}) == ['0']
    assert os.listdir(ddir) == ['0.json']
    assert _write_details_parts(str(ddir), str(index), {}) == []
    assert not ddir.exists() and not index.exists()


# --- publish_vol_shards -------------------------------------------------------

def _repo(tmp_path, hist=True):
    out = tmp_path / 'repo' / 'output'
    for fam, tickers in (('vol', ['AAA', 'BBB']), ('px', ['AAA']), ('hist', ['AAA', 'CCC']), ('details', ['0', '1'])):
        (out / fam).mkdir(parents=True)
        for t in tickers:
            (out / fam / f'{t}.json').write_text(json.dumps({fam: t}), encoding='utf-8')
    (out / 'prices_meta.json').write_text(json.dumps({'vol': ['AAA', 'BBB'], 'manifest': ['AAA']}),
                                          encoding='utf-8')
    (out / 'details_index.json').write_text(json.dumps({'parts': ['0', '1']}), encoding='utf-8')
    if hist:
        (out / 'hist_index.json').write_text(json.dumps({'tickers': ['AAA', 'CCC']}), encoding='utf-8')
    return tmp_path / 'repo', tmp_path / 'docs'


def test_publish_syncs_all_four_families(tmp_path, monkeypatch):
    repo, docs = _repo(tmp_path)
    (docs / 'hist').mkdir(parents=True)
    (docs / 'hist' / 'STALE.json').write_text('{}', encoding='utf-8')
    (docs / 'hist' / 'AAA 2.json').write_text('{}', encoding='utf-8')      # iCloud conflict copy
    monkeypatch.setenv('STOCK_MODEL_REPO', str(repo))
    monkeypatch.setenv('PAGES_DOCS', str(docs))
    assert pvs.main() == 0
    assert sorted(os.listdir(docs / 'vol')) == ['AAA.json', 'BBB.json']
    assert sorted(os.listdir(docs / 'px')) == ['AAA.json']
    assert sorted(os.listdir(docs / 'hist')) == ['AAA.json', 'CCC.json']
    assert sorted(os.listdir(docs / 'details')) == ['0.json', '1.json']
    assert json.loads((docs / 'hist' / 'CCC.json').read_text(encoding='utf-8')) == {'hist': 'CCC'}


def test_publish_fails_without_the_hist_index(tmp_path, monkeypatch):
    repo, docs = _repo(tmp_path, hist=False)
    monkeypatch.setenv('STOCK_MODEL_REPO', str(repo))
    monkeypatch.setenv('PAGES_DOCS', str(docs))
    assert pvs.main() == 1
    assert sorted(os.listdir(docs / 'px')) == ['AAA.json']                # the others still synced


def test_publish_fails_on_a_missing_shard(tmp_path, monkeypatch):
    repo, docs = _repo(tmp_path)
    os.remove(repo / 'output' / 'hist' / 'CCC.json')
    monkeypatch.setenv('STOCK_MODEL_REPO', str(repo))
    monkeypatch.setenv('PAGES_DOCS', str(docs))
    assert pvs.main() == 1


# --- check_pages_limits -------------------------------------------------------

def test_limits_pass_warn_and_fail():
    mib = 2 ** 20
    assert cpl.check([('a', mib)]) == ([], [])
    errors, warnings = cpl.check([('index.html', int(24.5 * mib))])
    assert errors == [] and 'index.html' in warnings[0] and '98%' in warnings[0]
    errors, _ = cpl.check([('hist.json', 25 * mib)])
    assert 'hist.json' in errors[0]
    errors, warnings = cpl.check([(str(i), 1) for i in range(16_000)])
    assert errors == [] and '16,000 files' in warnings[0]
    errors, _ = cpl.check([(str(i), 1) for i in range(20_001)])
    assert '20,001 files' in errors[0]


def test_limits_cli(tmp_path, capsys):
    site = tmp_path / 'docs'
    (site / 'hist').mkdir(parents=True)
    (site / 'index.html').write_bytes(b'x' * 900)
    (site / 'hist' / 'AAA.json').write_bytes(b'{}')
    (site / '.git').mkdir()
    (site / '.git' / 'big').write_bytes(b'x' * 5000)                     # not deployed
    assert cpl.main([str(site), '--max-file-mib', str(1000 / 2**20)]) == 0
    assert 'WARNING: index.html' in capsys.readouterr().out
    assert cpl.main([str(site), '--max-files', '1']) == 1
    assert cpl.main([str(tmp_path / 'nope')]) == 2


# --- run.sh step 08b ---------------------------------------------------------

def _publish_cloudflare_fn():
    src = open(RUN_SH, encoding='utf-8').read()
    m = re.search(r'^publish_cloudflare\(\) \{\n.*?^\}\n', src, re.S | re.M)
    assert m, 'publish_cloudflare() not found in run.sh'
    return m.group(0)


@pytest.mark.skipif(shutil.which('bash') is None, reason='needs bash')
@pytest.mark.parametrize('env', [
    {},                                                                         # no secrets
    {'CLOUDFLARE_API_TOKEN': 't', 'CLOUDFLARE_ACCOUNT_ID': 'a'},                # no project
    {'CLOUDFLARE_API_TOKEN': 't', 'CLOUDFLARE_ACCOUNT_ID': 'a', 'CF_PAGES_PROJECT': 'p', 'SMOKE': '1'},
    {'CLOUDFLARE_API_TOKEN': 't', 'CLOUDFLARE_ACCOUNT_ID': 'a', 'CF_PAGES_PROJECT': 'p', 'DRY_RUN': '1'},
])
def test_cloudflare_step_skips_cleanly(tmp_path, env):
    script = ('set -uo pipefail\nSMOKE="${SMOKE:-0}"; DRY_RUN="${DRY_RUN:-0}"\n'
              f'PAGES={tmp_path}; PYTHON=false; REPO={REPO}; WORK={tmp_path}; RUNDATE=2031-11-03\n'
              + _publish_cloudflare_fn() + 'publish_cloudflare\n')
    base = {k: v for k, v in os.environ.items()
            if k not in ('CLOUDFLARE_API_TOKEN', 'CLOUDFLARE_ACCOUNT_ID', 'CF_PAGES_PROJECT', 'SMOKE', 'DRY_RUN')}
    r = subprocess.run(['bash', '-c', script], env={**base, **env}, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    assert 'Cloudflare publish skipped' in r.stdout


def _run_sh_fn(name):
    src = open(RUN_SH, encoding='utf-8').read()
    m = re.search(rf'^{name}\(\) \{{\n.*?^\}}\n', src, re.S | re.M)
    assert m, f'{name}() not found in run.sh'
    return m.group(0)


@pytest.mark.skipif(shutil.which('bash') is None, reason='needs bash')
@pytest.mark.parametrize('fn,skip_text', [('publish_pages', 'publish skipped:'),
                                          ('publish_cloudflare', 'Cloudflare publish skipped:')])
def test_a_degraded_run_is_not_published_unless_forced(tmp_path, fn, skip_text):
    """2026-09-30: 739 of 2,514 rows were force-pushed over a good report."""
    line = 'COVERAGE degraded rows=739 prior_rows=2514 (2026-09-29) ratio=0.294 min=0.70'
    head = ('set -uo pipefail\nSMOKE=0; DRY_RUN=0\n'
            f'PAGES={tmp_path}/pages; HTML={tmp_path}/none.html; PYTHON=false; REPO={REPO}; '
            f'WORK={tmp_path}; RUNDATE=2031-11-03\nCOVERAGE_LINE="{line}"\n')
    env = {k: v for k, v in os.environ.items() if not k.startswith(('CLOUDFLARE_', 'CF_'))}
    r = subprocess.run(['bash', '-c', head + 'DEGRADED=1\n' + _run_sh_fn(fn) + f'{fn}\n'],
                       env=env, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    assert skip_text in r.stdout and 'last good report' in r.stdout and line in r.stdout
    assert not (tmp_path / 'pages').exists()            # nothing was built
    # FORCE=1 goes on to the real step (which fails here on the missing HTML
    # / secrets — the point is that it was not skipped).
    r = subprocess.run(['bash', '-c', head + 'DEGRADED=1; FORCE=1\n' + _run_sh_fn(fn) + f'{fn}\n'],
                       env=env, capture_output=True, text=True, timeout=30)
    assert skip_text not in r.stdout


@pytest.mark.skipif(shutil.which('bash') is None, reason='needs bash')
def test_cloudflare_step_refuses_a_missing_site(tmp_path):
    script = ('set -uo pipefail\nSMOKE=0; DRY_RUN=0\n'
              f'PAGES={tmp_path}; PYTHON=false; REPO={REPO}; WORK={tmp_path}; RUNDATE=2031-11-03\n'
              + _publish_cloudflare_fn() + 'publish_cloudflare\n')
    env = {**os.environ, 'CLOUDFLARE_API_TOKEN': 't', 'CLOUDFLARE_ACCOUNT_ID': 'a', 'CF_PAGES_PROJECT': 'p'}
    r = subprocess.run(['bash', '-c', script], env=env, capture_output=True, text=True, timeout=30)
    assert r.returncode == 1 and 'did not build the site' in r.stdout
