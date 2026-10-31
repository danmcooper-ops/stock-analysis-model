"""The top-bar search finds stocks, ETFs, currencies and bonds (qrRender),
and a pick goes where each lives (_qrGo). The renderer is evaluated in Node
with its helpers lifted from the template and a stub DOM."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

TEMPLATE = Path(__file__).resolve().parents[1] / 'templates' / 'report.html'
node = pytest.mark.skipif(shutil.which('node') is None, reason='node not installed')


def _src():
    return TEMPLATE.read_text(encoding='utf-8')


def _fn(name, src=None):
    return re.search(r'^function ' + re.escape(name) + r'\(.*?^\}', src or _src(), re.S | re.M).group(0)


def _var(name, src):
    return re.search(r'^var ' + re.escape(name) + r'=.*?;$', src, re.M).group(0)


STOCKS = [
    {'ticker': 'SGP', 'company_name': 'SpyGlass Pharma, Inc.'},
    {'ticker': 'EURBF', 'company_name': 'EUROBANK S A'},
    {'ticker': 'AAPL', 'company_name': 'Apple Inc.'},
]
ETFS = {'base': 'SPY', 'funds': {
    'SPY': {'fund_name': 'SPDR S&P 500 ETF Trust', 'sector': None},
    'XLE': {'fund_name': 'Energy Select Sector SPDR ETF', 'sector': 'Energy'}}}
CCY = {'date': '2026-10-28', 'total': 2, 'index': [['EUR', 'Euro'], ['JPY', 'Japanese Yen']]}
BONDS = {'columns': [{'k': 'cusip'}, {'k': 'issuer_name'}, {'k': 'coupon_rate'}, {'k': 'maturity_date'}],
         'rows': [['037833DU1', 'APPLE INC', 0.0165, '2030-05-11'],
                  ['037833AL4', 'APPLE INC', 0.0385, '2043-05-04'],
                  ['29874QEX8', 'European Bank for Reconstruction', 0.04375, '2028-03-14']]}


def _run(tmp_path, view, msub, queries):
    src = _src()
    lifted = '\n'.join(_fn(n, src) for n in (
        '_buildSearchMemo', '_relScore', '_qrActive', '_qrOwn', '_qrScore', '_qrTop',
        'qrRender', '_bvIdx', '_bvHay', '_attr', '_etfCat'))
    body = f"""
var box={{innerHTML:'',cls:{{}},classList:{{add:function(c){{box.cls[c]=1;}},remove:function(c){{delete box.cls[c];}},contains:function(c){{return !!box.cls[c];}}}}}};
var inp={{value:''}};
var document={{getElementById:function(id){{return id==='hdr-qr'?box:inp;}},
  documentElement:{{classList:{{contains:function(){{return false;}}}}}}}};
function esc(s){{return String(s);}}
var DATA={json.dumps(STOCKS)},ETF_LT={json.dumps(ETFS)},CCY_META={json.dumps(CCY)},BOND_META={{total:3}};
var _BV={json.dumps(BONDS)},_BV_FAILED=false,_CX=null,_bvIdxMap=null;
var curView={json.dumps(view)},curMSub={json.dumps(msub)};
{_var('_QR_N', src)}
{_var('_ETF_CATS', src)}
{lifted}
var out={{}};
{json.dumps(queries)}.forEach(function(q){{
  inp.value=q;qrRender();
  var opts=[],re=/data-qk="([^"]*)" data-qr="([^"]*)" class="([^"]*)"/g,m;
  while((m=re.exec(box.innerHTML)))opts.push([m[1],m[2],m[3].indexOf('hi')>=0]);
  out[q]={{open:box.classList.contains('open'),opts:opts,
    groups:(box.innerHTML.match(/class="qr-grp"[^>]*>([^<]*)</g)||[]).map(function(g){{return g.replace(/.*>/,'').replace('<','');}})}};
}});
console.log(JSON.stringify(out));
"""
    script = tmp_path / 't.js'
    script.write_text(body, encoding='utf-8')
    r = subprocess.run(['node', str(script)], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads(r.stdout)


@node
def test_finds_every_kind_and_preselects_the_best(tmp_path):
    out = _run(tmp_path, 'macro', None, ['spy', 'eur', 'apple 2030', 'energy', 'zzzz'])
    # An exact ETF or currency outranks a stock whose name merely contains it.
    assert out['spy']['groups'][0] == 'ETFs' and out['spy']['opts'][0] == ['etf', 'SPY', True]
    assert out['eur']['groups'][0] == 'Currencies' and out['eur']['opts'][0] == ['ccy', 'EUR', True]
    assert ['bond', '2', False] in out['eur']['opts']
    # Every word must match a bond, maturity included; the key is the row.
    bonds = [o for o in out['apple 2030']['opts'] if o[0] == 'bond']
    assert bonds == [['bond', '0', True]]
    assert out['energy']['opts'] == [['etf', 'XLE', True]]
    assert out['zzzz']['open'] and not out['zzzz']['opts']
    assert sum(o[2] for q in out.values() for o in q['opts']) == 4


@node
def test_a_table_view_leaves_its_own_kind_out_and_preselects_nothing(tmp_path):
    out = _run(tmp_path, 'matrix', 'matrix', ['spy', 'apple', 'zzzz'])
    assert out['spy']['opts'] == [['etf', 'SPY', False]]
    assert 'Stocks' not in out['apple']['groups'] and out['apple']['groups'] == ['Bonds']
    assert not out['zzzz']['open']
    out = _run(tmp_path, 'matrix', 'bond', ['apple'])
    assert out['apple']['groups'] == ['Stocks']


def test_picks_go_where_each_kind_lives():
    go = _fn('_qrGo')
    assert 'openDet(key)' in go
    assert "_etfSel=key;_fmKeep('etf','detail')" in go
    assert "_cxSel=key;_cxView='detail';_fmKeep('ccy','detail')" in go
    assert '_bvOpenRow(_BV.rows[+key])' in go
    # A jump away from the page you were on is a drill, so Back returns.
    assert go.index('_drillPush(_qrHere())') < go.index('showView') and '_drillSeal()' in go
    # Enter only opens something chosen; on a table view nothing is.
    assert 'if(i<0)return;' in _fn('qrKey')


def test_tab_searches_share_the_rules():
    src = _src()
    assert '_etfQ()' in _fn('renderEtfLookthrough', src)
    rows = _fn('_bvRows', src)
    assert '_bvHay()' in rows and 'toks.length' in rows
    assert 'maturity_date' in _fn('_bvHay', src)
    assert "_loadSidecar('bonds.json'," in _fn('_bvLoad', src)
