"""Financial Model › Bonds vs Stocks: wiring, display-only rule, and the
renderer evaluated in Node with its helpers lifted from the template."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

TEMPLATE = Path(__file__).resolve().parents[1] / 'templates' / 'report.html'


def _tpl():
    return TEMPLATE.read_text(encoding='utf-8')


def _fn_src(name, src):
    return re.search(r'^function ' + re.escape(name) + r'\(.*?^\}', src, re.S | re.M).group(0)


def test_subtab_is_wired():
    src = _tpl()
    assert "{k:'bond',l:'Bonds vs Stocks'}" in src
    assert '<div id="m-bond-view" style="display:none"></div>' in src
    show = _fn_src('_mShow', src)
    assert "k==='bond'" in show and 'renderBondView()' in show
    assert "var BOND_RV={{ bond_json|default('null') }};" in src


def test_display_only_never_writes_data():
    src = _tpl()
    body = ''.join(_fn_src(n, src) for n in
                   ('renderBondView', 'renderBondsVsStocks', '_bndLadder', '_bndSector'))
    assert not re.search(r'\b(DATA|FD)\b', body)


def test_sector_row_is_a_drill_with_a_way_back():
    fn = _fn_src('_bndSector', _tpl())
    assert fn.index('_drillPush(') < fn.index("showView('pool')") < fn.index('showPoolTab(')
    assert '_drillSeal()' in fn


PAYLOAD = {
    'as_of': '2026-10-23', 'hy_credit_loss': 0.015,
    'bonds': [{'id': 'ust10Y', 'l': '10-year Treasury', 'kind': 'treasury', 'y': 0.05,
               'chg_1y': 0.01, 'pctile': 0.99, 'pct_win': '20y'},
              {'id': 'tips10', 'l': '10-year TIPS (real)', 'kind': 'tips', 'y': 0.0266,
               'derived': '10Y Treasury − 10Y breakeven inflation'},
              {'id': 'ig', 'l': 'Investment-grade corporate', 'kind': 'corp', 'y': 0.058}],
    'stocks': {'n': 3824, 'mcap': 1.06e14, 'earnings_yield': 0.0443, 'fcf_yield': 0.0288,
               'shareholder_yield': 0.0235, 'coverage': 1.0},
    'model': {'source': 'SPY', 'mos': -0.24},
    'premia': {'vs_real': 0.0177, 'fcf_vs_ig': -0.0292},
    'sectors': [{'sector': 'Energy', 'n': 216, 'status': 'ok', 'earnings_yield': 0.065,
                 'fcf_yield': 0.04, 'vs_real': 0.038, 'fcf_vs_ig': -0.018},
                {'sector': 'Tiny <Sector>', 'n': 2, 'status': 'insufficient',
                 'reason': '2 companies', 'earnings_yield': None, 'fcf_yield': None,
                 'vs_real': None, 'fcf_vs_ig': None}],
}


def _render(payload):
    if shutil.which('node') is None:
        pytest.skip('node not installed')
    src = _tpl()
    attr = re.search(r'^function _attr\(s\)\{.*?\}$', src, re.M).group(0)
    fns = [_fn_src(n, src) for n in ('_esc', '_etfPct', '_ppiV', '_ppiTh', '_bndY', '_bndPp',
                                     '_bndGap', '_bndOrd', '_bndLadder', 'renderBondsVsStocks')]
    js = '\n'.join([
        'function _qHelp(c,l,a){return "";}', attr, *fns,
        f'var BOND_RV={json.dumps(payload)};',
        'process.stdout.write(renderBondsVsStocks());'])
    return subprocess.run(['node', '-e', js], capture_output=True, text=True, check=True).stdout


def test_render_headline_ladder_and_sectors():
    html = _render(PAYLOAD)
    assert 'Stocks earn <b>1.8pp</b> more than inflation-protected Treasuries' in html
    assert '<b>2.9pp</b> below investment-grade' in html
    assert '(P/E 22.6)' in html
    assert '99th percentile, 20y' in html
    assert '≈ 10Y Treasury' in html
    # Highest yield first: the IG bar precedes the 10Y and the stock lines.
    assert html.index('Investment-grade corporate') < html.index('10-year Treasury') \
        < html.index('Stocks: earnings yield')
    assert 'onclick="_bndSector(this.dataset.sec)"' in html
    assert 'Tiny &lt;Sector&gt;' in html and '2 companies' in html


def test_render_empty_without_payload():
    assert _render(None) == ''
