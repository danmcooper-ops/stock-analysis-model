"""The top-bar search finds stocks, ETFs, currencies and bonds (qrRender),
and a pick goes where each lives (_qrGo). The renderer is evaluated in Node
with its helpers lifted from the template and a stub DOM."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from score_band_js import band_js

TEMPLATE = Path(__file__).resolve().parents[1] / 'templates' / 'report.html'
node = pytest.mark.skipif(shutil.which('node') is None, reason='node not installed')


def _src():
    return TEMPLATE.read_text(encoding='utf-8')


def _fn(name, src=None):
    return re.search(r'^function ' + re.escape(name) + r'\(.*?^\}', src or _src(), re.S | re.M).group(0)


def _line(name, src):
    """A one-line function, which _fn's closing-brace search would overrun."""
    return re.search(r'^function ' + re.escape(name) + r'\(.*$', src, re.M).group(0)


def _var(name, src):
    return re.search(r'^var ' + re.escape(name) + r'=.*?;$', src, re.M).group(0)


STOCKS = [
    {'ticker': 'SGP', 'company_name': 'SpyGlass Pharma, Inc.', 'sector': 'Healthcare', 'rating': 'HOLD', 'pe': 30, 'mos': -0.1},
    {'ticker': 'EURBF', 'company_name': 'EUROBANK S A', 'sector': 'Financial Services', 'rating': 'BUY', 'pe': 8, 'mos': 0.3,
     'mcap': 2e10},
    {'ticker': 'AAPL', 'company_name': 'Apple Inc.', 'sector': 'Technology', 'rating': 'LEAN BUY', 'pe': 31, 'mos': -0.2,
     'price': 230.5, 'mcap': 3.4e12},
]
ETFS = {'base': 'SPY', 'funds': {
    'SPY': {'fund_name': 'SPDR S&P 500 ETF Trust', 'sector': None},
    'XLE': {'fund_name': 'Energy Select Sector SPDR ETF', 'sector': 'Energy'}}}
CCY = {'date': '2026-10-28', 'total': 2, 'index': [['EUR', 'Euro'], ['JPY', 'Japanese Yen']]}
BONDS = {'columns': [{'k': 'cusip'}, {'k': 'issuer_name'}, {'k': 'coupon_rate'}, {'k': 'maturity_date'}],
         'rows': [['037833DU1', 'APPLE INC', 0.0165, '2030-05-11'],
                  ['037833AL4', 'APPLE INC', 0.0385, '2043-05-04'],
                  ['29874QEX8', 'European Bank for Reconstruction', 0.04375, '2028-03-14']]}


def _block(src):
    """The query parser, recents, pages and terms: everything qrRender
    calls that sits between the SEARCH QUERIES marker and qrRender."""
    return src[src.index('/* === SEARCH QUERIES ==='):src.index('function qrRender(){')]


CF_OPTS = [{'g': 'Valuation', 'k': 'pe', 'l': 'P/E'},
           {'g': 'Valuation', 'k': 'mos', 'l': 'MoS% (%)'},
           {'g': 'Valuation', 'k': 'ev_ebitda', 'l': 'EV/EBITDA'},
           {'g': 'Profitability', 'k': 'roic', 'l': 'ROIC (%)'},
           {'g': 'Debt', 'k': 'mcap', 'l': 'Market Cap ($)'},
           {'g': 'Text', 'k': 'industry', 'l': 'Industry (text)'}]
CF_FMT = {'pe': 'n1', 'mos': 'pct', 'ev_ebitda': 'n1', 'roic': 'pct', 'mcap': 'ds', 'industry': 'text'}
TT = {'pe': 'P/E = Price / EPS. The second sentence.', 'roic': 'ROIC = NOPAT / invested capital. More.'}
NAV = [{'v': 'pool', 'l': 'Sector Analysis', 'items': [{'k': '__performance__', 'l': 'Overview'}, {'k': 'Energy', 'l': 'Energy'}]},
       {'v': 'matrix', 'l': 'Investing', 'items': [{'k': 'matrix', 'l': 'Stocks'}, {'k': 'bond', 'l': 'Bonds'}]}]


def _run(tmp_path, view, msub, queries):
    src = _src()
    lifted = '\n'.join(_fn(n, src) for n in (
        '_buildSearchMemo', '_relScore', '_qrActive', '_qrOwn', '_qrScore', '_qrTop',
        'qrRender', '_qrIsQuestion', '_bvIdx', '_bvHay', '_attr', '_etfCat'))
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
{_var('_QR_AI_ICON', src)}
var _CF_OPTS={json.dumps(CF_OPTS)},_CF_FMT={json.dumps(CF_FMT)},TT={json.dumps(TT)};
function _navGroups(){{return {json.dumps(NAV)};}}
{_line('_num', src)}
function rb(r){{return '<span class="rb">'+r+'</span>';}}
{_block(src)}
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
    script.write_text(band_js(_src()) + body, encoding='utf-8')
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
    assert out['energy']['opts'] == [['etf', 'XLE', True], ['page', 'pool|Energy', False], ['ai', '', False]]
    # Nothing found: asking the AI Analyst is the one choice, and Enter takes it.
    assert out['zzzz']['open'] and out['zzzz']['opts'] == [['ai', '', True]]
    assert sum(o[2] for q in out.values() for o in q['opts']) == 5


@node
def test_a_table_view_leaves_its_own_kind_out_and_preselects_nothing(tmp_path):
    out = _run(tmp_path, 'matrix', 'matrix', ['spy', 'apple', 'zzzz'])
    assert out['spy']['opts'] == [['etf', 'SPY', False], ['ai', '', False]]
    assert 'Stocks' not in out['apple']['groups'] and out['apple']['groups'] == ['Bonds', 'AI Analyst']
    assert out['zzzz']['opts'] == [['ai', '', False]]
    out = _run(tmp_path, 'matrix', 'bond', ['apple'])
    assert out['apple']['groups'] == ['Stocks', 'AI Analyst']


@node
def test_a_question_asks_the_ai_analyst_first(tmp_path):
    # Even on a table view, where Enter otherwise only filters.
    for view, msub in (('macro', None), ('matrix', 'matrix')):
        out = _run(tmp_path, view, msub, ['which banks look cheap?', 'compare apple and spy', 'spy'])
        for q in ('which banks look cheap?', 'compare apple and spy'):
            assert out[q]['groups'][0] == 'AI Analyst' and out[q]['opts'][0] == ['ai', '', True]
        assert out['spy']['groups'][-1] == 'AI Analyst'


def test_the_ai_analyst_lives_in_the_search_bar():
    src = _src()
    assert 'id="ai-fab"' not in src and '#ai-fab' not in src
    bar = src[src.index('<div class="hdr-search'):src.index('id="hdr-qr"')]
    # One way in, as in the iOS app: the field itself, no sparkle button.
    assert 'hdr-ai' not in src and 'aiToggle()' not in bar
    assert 'placeholder="Search or ask AI"' in bar
    go = _fn('_qrGo', src)
    assert "k==='ai'" in go and 'window.aiAsk' in go
    assert 'window.aiAsk=function' in src and 'window.aiToggle=function' in src


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


def _eval(tmp_path, js):
    """Run js after the query block, with the fixture rows and columns."""
    src = _src()
    body = f"""
var DATA={json.dumps(STOCKS)},_CF_OPTS={json.dumps(CF_OPTS)},_CF_FMT={json.dumps(CF_FMT)},TT={json.dumps(TT)};
{_line('_num', src)}
{_fn('_buildSearchMemo', src)}
{_fn('_qrScore', src)}
function fds(v){{return '$'+(v/1e9)+'B';}}
function rb(r){{return '<span class="rb">'+r+'</span>';}}
function _navGroups(){{return {json.dumps(NAV)};}}
{_block(src)}
console.log(JSON.stringify((function(){{{js}}})()));
"""
    script = tmp_path / 'q.js'
    script.write_text(band_js(_src()) + body, encoding='utf-8')
    r = subprocess.run(['node', str(script)], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads(r.stdout)


@node
def test_a_field_query_parses_into_the_filters_panels_terms(tmp_path):
    out = _eval(tmp_path, """
var a=_fqParse('pe<15 sector:fin rating:lean+ roic>=12.5 mcap>10b apple and ev/ebitda:5-10');
var b=_fqParse('industry:"semi conductors" score>50');
var c=_fqParse('price target for apple');
return {a:a,la:a.conds.map(_fqLabel),b:b,c:c,n:_fqCount(_fqParse('pe<=10')),n2:_fqCount(_fqParse('rating:buy+ mos>-15'))};""")
    a = {c.get('k') or c['ms']: c for c in out['a']['conds']}
    assert a['pe']['max'] == 15 and a['pe']['min'] is None
    assert a['ms-sector']['vals'] == ['Financial Services']
    assert a['ms-rating']['vals'] == ['BUY', 'LEAN BUY']
    # Percent columns take percents; dollar columns take suffixes.
    assert a['roic']['min'] == 12.5 and a['mcap']['min'] == 1e10
    assert (a['ev_ebitda']['min'], a['ev_ebitda']['max']) == (5, 10)
    # Unparsed words stay as search text, joiners dropped.
    assert out['a']['text'] == 'apple'
    assert 'P/E ≤ 15' in out['la'] and 'Market Cap ≥ $10B' in out['la'] and 'ROIC ≥ 12.5%' in out['la']
    assert out['b']['conds'][0]['txt'] == 'semi conductors'
    # A sentence with no FIELD OP VALUE term is not a query.
    assert out['c']['conds'] == []
    assert out['n'] == 1 and out['n2'] == 1


@node
def test_a_field_query_leads_the_list_and_offers_save(tmp_path):
    out = _run(tmp_path, 'matrix', 'matrix', ['pe<15 rating:buy', 'roic'])
    assert out['pe<15 rating:buy']['groups'] == ['Filter', 'AI Analyst']
    assert [o[0] for o in out['pe<15 rating:buy']['opts']] == ['fq', 'fqsave', 'ai']
    assert out['pe<15 rating:buy']['opts'][0][2]
    # A column name alone finds its definition under Terms.
    assert ['term', 'roic', False] in out['roic']['opts']


@node
def test_pages_and_stock_rows(tmp_path):
    out = _eval(tmp_path, """
return {p:_qrPages(['BONDS']).map(function(x){return x[3];}),
        t:_qrTerms(['PE']).map(function(x){return [x[1],x[2]];}),
        x:_qrStockX(DATA[2])};""")
    assert out['p'] == ['matrix|bond']
    assert out['t'] == [['P/E', 'P/E = Price / EPS.']]
    assert '$230.50' in out['x'] and '-20%' in out['x']
    assert 'LEAN BUY' not in out['x']          # a stock row shows its score, not a rating


@node
def test_recent_searches_keep_the_last_picks(tmp_path):
    out = _eval(tmp_path, """
var store={};var localStorage={getItem:function(k){return store[k]||null;},setItem:function(k,v){store[k]=v;},removeItem:function(k){delete store[k];}};
this.localStorage=localStorage;globalThis.localStorage=localStorage;
for(var i=0;i<10;i++)_qrRecentPush({k:'stock',key:'T'+i,t:'T'+i});
_qrRecentPush({k:'stock',key:'T5',t:'T5'});
var a=_qrRecent().map(function(e){return e.key;});
_qrRecentClear();
return {a:a,b:_qrRecent()};""")
    assert out['a'] == ['T5', 'T9', 'T8', 'T7', 'T6', 'T4', 'T3', 'T2']
    assert out['b'] == []


def test_picks_and_ai_actions_are_wired():
    src = _src()
    go, pick = _fn('_qrGo', src), _fn('_qrPick', src)
    assert '_fqApply(_fqParse(qtext))' in go and 'pfSaveFilterOpen()' in go
    assert 'navGo(pv[0]' in go and '_qrRecentPush' in go
    assert "k==='term'" in pick and "k==='rec'" in pick and "_qrPick(" in _fn('qrKey', src)
    # The table's own search ignores the query's FIELD OP VALUE terms.
    assert '_fqParse(q)' in _fn('applyFilters', src)
    # The analyst can act: filter the table, open a ticker.
    assert "name:'apply_filter'" in src and "name:'open_ticker'" in src
    assert "case 'apply_filter':return aiApplyFilter" in src
    flt = _fn('aiApplyFilter', src)
    assert "if(fmt==='pct')v=+(v*100)" in flt and '_fqApply(fq)' in flt
    assert 'openDet(t)' in _fn('aiOpenTicker', src)


def test_analyst_replays_unknown_blocks_whole():
    """Opus 5.5 can return blocks the parser does not build up from deltas
    (redacted_thinking carries its data in the start event); they go back
    as received, so a tool loop's next turn is not rejected."""
    src = _src()
    assert "var AI_MODEL='claude-opus-5-5';" in src
    stream = src[src.index('function aiStream('):src.index('/* ---------- minimal markdown')]
    assert 'else blocks[ev.index]=JSON.parse(JSON.stringify(nb));' in stream
    assert "if(b.type==='thinking')return {type:'thinking',thinking:b.thinking,signature:b.signature};" in stream
