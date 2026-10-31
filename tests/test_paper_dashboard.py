"""Portfolios › paper › Paper account: the Alpaca paper account against the
model. The payload builder, the render's staging, the template wiring, and
the renderer evaluated in Node with its helpers lifted from the template."""
import json
import re
import shutil
import subprocess
from datetime import date
from pathlib import Path

import pytest

from models import paper_dashboard as pd
from scripts import report_html

TEMPLATE = Path(__file__).resolve().parents[1] / 'templates' / 'report.html'

NAV = [{'d': '2026-10-29', 'equity': 100_000.0}, {'d': '2026-10-30', 'equity': 100_500.0},
       {'d': '2026-11-02', 'equity': 101_000.0}]
MODEL = [{'d': '2026-10-28', 'nav': 99.0}, {'d': '2026-10-29', 'nav': 100.0},
         {'d': '2026-10-30', 'nav': 101.0}, {'d': '2026-11-02', 'nav': 102.0}]
SPY = [{'d': '2026-10-29', 'nav': 200.0}, {'d': '2026-10-30', 'nav': 201.0},
       {'d': '2026-11-02', 'nav': 202.0}]
RECORDS = [
    {'type': 'reconcile', 'date': '2026-10-29', 'positions': {}, 'cash': 100_000.0},
    {'type': 'rebalance', 'date': '2026-10-30', 'execute': True, 'snapshot': '2026-10-29',
     'portfolio': 'paper', 'weights': {'A': 0.5, 'B': 0.5}, 'halted': [],
     'orders': [{'ticker': 'A'}, {'ticker': 'B'}], 'sent': [{'ticker': 'A'}], 'failed': [{}],
     'turnover': 50_000.0, 'untradable_weight': 0.04,
     'skipped': [{'ticker': 'C', 'reason': 'quote 12 is +20% from the snapshot close 10'}]},
    {'type': 'reconcile', 'date': '2026-10-30', 'filled': 1, 'filled_dollars': 50_000.0,
     'mean_slip_vs_close': 0.002, 'cash': 51_000.0,
     'positions': {'A': {'qty': 100, 'value': 50_250.0, 'avg_cost': 500.0, 'pl': 0.005},
                   'X': {'qty': 1, 'value': 100.0, 'avg_cost': 90.0, 'pl': 0.11}}},
]


def _build(run_date='2026-11-02'):
    return pd.build('paper', RECORDS, NAV, MODEL, SPY, run_date)


def test_returns_rebased_on_the_first_account_day():
    d = _build()
    assert d['start'] == '2026-10-29' and d['as_of'] == '2026-11-02'
    assert d['series']['paper'] == [100.0, 100.5, 101.0]
    assert d['series']['model'] == [100.0, 101.0, 102.0]
    assert d['returns']['paper'] == pytest.approx(0.01)
    assert d['returns']['model'] == pytest.approx(0.02)
    assert d['returns']['delta'] == pytest.approx(-0.01)
    assert d['returns']['vs_spy'] == pytest.approx(0.0)


def test_run_date_cuts_later_points():
    d = _build('2026-10-30')
    assert d['series']['d'] == ['2026-10-29', '2026-10-30']
    assert d['returns']['model'] == pytest.approx(0.01)


def test_holdings_against_targets():
    h = {x['t']: x for x in _build()['holdings']}
    assert h['A']['status'] == 'held' and h['A']['target'] == 0.5
    assert h['A']['actual'] == pytest.approx(50_250 / 101_000, abs=1e-6)
    assert h['B']['status'] == 'not yet bought' and h['B']['actual'] == 0
    assert h['X']['status'] == 'to sell' and h['X']['target'] == 0
    b = _build()['build']
    assert (b['targets'], b['held'], b['pending'], b['to_sell']) == (2, 1, 1, 1)
    assert b['target_held'] == pytest.approx(50_250 / 101_000, abs=1e-4)


def test_activity_and_skips():
    d = _build()
    a = {x['d']: x for x in d['activity']}
    assert a['2026-10-30']['sent'] == 1 and a['2026-10-30']['failed'] == 1
    assert a['2026-10-30']['slip'] == 0.002
    assert d['skipped'] == {'quote moved since the snapshot': 1}
    assert d['untradable'] == 0.04


def test_no_equity_means_no_dashboard():
    assert pd.build('paper', RECORDS, [], MODEL, SPY) is None


def test_other_portfolios_records_are_ignored():
    recs = [dict(r, portfolio='other') if r['type'] == 'rebalance' else r for r in RECORDS]
    d = pd.build('paper', recs, NAV, MODEL, SPY)
    assert all(h['target'] == 0 for h in d['holdings'])


# -- staging ----------------------------------------------------------------

def _seed(tmp_path, monkeypatch):
    paper = tmp_path / 'paper'
    paper.mkdir()
    (paper / 'ledger.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in RECORDS),
                                        encoding='utf-8')
    (paper / 'nav.json').write_text(json.dumps(NAV), encoding='utf-8')
    monkeypatch.setenv('PAPER_DIR', str(paper))
    out = tmp_path / 'out'
    out.mkdir()
    (out / 'portfolio_nav.json').write_text(json.dumps(
        {'version': 1, 'portfolios': {'paper': MODEL}, 'bench': {'spy': SPY, 'universe': []}}),
        encoding='utf-8')
    return out


def test_stage_writes_paper_json(tmp_path, monkeypatch):
    out = _seed(tmp_path, monkeypatch)
    meta = report_html._stage_paper_dashboard(str(out), date(2026, 11, 2), [{'id': 'paper'}])
    assert meta == {'portfolio': 'paper', 'as_of': '2026-11-02', 'start': '2026-10-29'}
    data = json.loads((out / 'paper.json').read_text(encoding='utf-8'))
    assert data['version'] == pd.PAPER_DASHBOARD_VERSION and data['holdings']


def test_stage_removes_a_stale_file(tmp_path, monkeypatch):
    out = _seed(tmp_path, monkeypatch)
    (out / 'paper.json').write_text('{}', encoding='utf-8')
    assert report_html._stage_paper_dashboard(str(out), date(2026, 11, 2), [{'id': 'semis'}]) is None
    assert not (out / 'paper.json').exists()


def test_stage_without_records(tmp_path, monkeypatch):
    monkeypatch.setenv('PAPER_DIR', str(tmp_path / 'none'))
    assert report_html._stage_paper_dashboard(str(tmp_path), date(2026, 11, 2),
                                              [{'id': 'paper'}]) is None


# -- template ---------------------------------------------------------------

def _tpl():
    return TEMPLATE.read_text(encoding='utf-8')


def _fn_src(name, src):
    return re.search(r'^function ' + re.escape(name) + r'\(.*?^\}', src, re.S | re.M).group(0)


def test_wiring():
    src = _tpl()
    assert "var PAPER_META={{ paper_meta|default('null') }};" in src
    assert "PAPER_META.portfolio===id&&['paper','Paper account'" in _fn_src('_pfDetailHtml', src)
    assert "_loadSidecar('paper.json'," in _fn_src('_ppaEnsure', src)
    assert '_ppaEnsure()' in _fn_src('renderPfView', src)
    assert "k==='paper'" in _fn_src('_pfSubTab', src)
    # The pane is display:none unless its data-tab is listed.
    assert '.pf-tabbed[data-tab="paper"]>[data-pane="paper"]' in src
    sh = (TEMPLATE.parents[1] / 'scripts' / 'run_daily.sh').read_text(encoding='utf-8')
    assert 'cp output/paper.json "$docs/paper.json"' in sh and 'rm -f "$docs/paper.json"' in sh


def test_display_only_never_writes_data():
    src = _tpl()
    body = ''.join(_fn_src(n, src) for n in ('_ppaEnsure', '_ppaHtml', '_ppaHoldHtml',
                                             '_ppaActHtml', '_ppaDraw', '_ppaSort'))
    assert not re.search(r'\bDATA\b', body)


def test_render_in_node():
    if shutil.which('node') is None:
        pytest.skip('node not installed')
    src = _tpl()
    rb = re.search(r'^function rb\(r,d\)\{.*?\}$', src, re.M).group(0)
    rc = re.search(r"^var RC=\{.*?\};$", src, re.M).group(0)
    start = src.index('var _PPA=null,')
    block = src[start:src.index('/* The chosen section carries across portfolios', start)]
    fns = [_fn_src(n, src) for n in ('_pfEsc', '_pfJsArg', '_pfTkLink', '_pfMoney',
                                     '_attr', '_qHelp', '_pfNoteQ')]
    fmt = re.search(r'^function _pfFmtRet\(v\)\{.*?\}$', src, re.M).group(0)
    js = '\n'.join([
        'function isCapped(){return false;}', rb, rc, fmt, *fns, block,
        "var PAPER_META={portfolio:'paper',as_of:'2026-11-02'};",
        "var _pfRow={A:{rating:'BUY'},B:{rating:'LEAN BUY'}};",
        f'_PPA={json.dumps(_build())};',
        'process.stdout.write(_ppaHtml());'])
    html = subprocess.run(['node', '-e', js], capture_output=True, text=True, check=True).stdout
    for needle in ('Paper − model', '-1.00pp', 'Book bought', '1 of 2 names held',
                   'not yet bought', 'to sell', 'Held vs target', 'Activity',
                   'not tradable at Alpaca', 'quote moved since the snapshot', '+0.20%'):
        assert needle in html, needle
