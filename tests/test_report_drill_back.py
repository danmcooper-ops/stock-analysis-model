# tests/test_report_drill_back.py
"""Drill-down return: a chart that jumps elsewhere (Profit Pool industry,
Dashboard card / rating / map country -> Gate Matrix; Sector Heatmap or the
Cross-Sector Profit Pool -> a sector page) leaves a way back that restores
the origin's view, tab, scroll and filters.

Source-level assertions against templates/report.html, plus the stack
helpers evaluated in Node with the DOM and navigation stubbed.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

TEMPLATE = Path(__file__).resolve().parents[1] / 'templates' / 'report.html'


def _tpl():
    return TEMPLATE.read_text(encoding='utf-8')


def _fn_src(name, src=None):
    src = src or _tpl()
    return re.search(r'^function ' + name + r'\(.*?^\}', src, re.S | re.M).group(0)


@pytest.mark.parametrize('name,mutation', [
    ('dashIndustryJump', "_cfPut('industry'"),
    ('dashCatJump', '_cfPut(scoreKey'),
    ('dashRatingJump', 'cb.checked=cb.value===r'),
])
def test_every_jump_records_its_origin_before_touching_filters(name, mutation):
    fn = _fn_src(name)
    assert '_drillPush(' in fn
    assert fn.index('_drillPush(') < fn.index(mutation)


def test_map_and_sector_launchers_record_their_origin():
    src = _tpl()
    i = src.index('var isSoleSelection=_restricted')
    seg = src[i:i + 400]
    assert "if(!isSoleSelection)_drillPush(" in seg
    assert seg.index('_drillPush(') < seg.index('boxes.forEach')
    # The heatmap cell and the cross-sector pool bars go through drillPoolTab.
    assert "\\'Sector Heatmap\\')" in src
    assert "\\'Cross-Sector Profit Pool\\')" in src
    assert 'onclick="showPoolTab(' not in src


def test_back_link_lives_in_the_subnav_and_hides_when_empty():
    src = _tpl()
    i = src.index('<div id="subnav"')
    assert 'id="sn-back" onclick="drillBack()"' in src[i:i + 300]
    # .d-back:empty is what hides it with no drill pending.
    assert '.d-back:empty{display:none;}' in src


def test_every_navigation_leaves_pending_drills_and_clear_empties_them():
    """Sidebar, chips, hotkeys, swipe and in-page links all go through navGo,
    so leaving a drill there undoes its filter whichever route is taken;
    drillBack's own navGo is guarded by _drillRestoring."""
    src = _tpl()
    nav = _fn_src('navGo', src)
    assert nav.index('_drillLeave();') < nav.index('if(k==null)')
    assert 'if(_drillRestoring||!_drillStack.length)return;' in _fn_src('_drillLeave', src)
    assert '_drillClear();' in _fn_src('clearFilters', src)
    # Every launcher seals the post-drill state the undo compares against.
    assert '_drillSeal();' in _fn_src('_dashJump', src)
    assert '_drillSeal();' in src[src.index('function drillPoolTab('):][:120]


def test_b_hotkey_is_listed_and_dispatched():
    src = _tpl()
    assert "{g:'Navigation',k:['b']," in src
    assert "else if(k==='b'){if(_drillStack.length)" in _fn_src('_hkDispatch', src)


def _run(tmp_path, body):
    if shutil.which('node') is None:
        pytest.skip('node not installed')
    src = _tpl()
    start = src.index('var _drillStack=[]')
    helpers = src[start:src.index('function drillPoolTab(', start)]
    stub = r"""
var calls=[],curView='pool',_poolTab='Energy',curMSub='matrix',curMacSub='overview',
    curPfSub='overview',curTSub='prices',_hashTimer=null,
    _filt={search:'',colFilters:[]},_FM_SUB={},_bvPopRow=null,_bvSkipPop=false;
function _fmSubTab(v,k){_FM_SUB[v]=k;calls.push(['fm',v,k]);}
var backBtn={textContent:'',title:''};
var document={getElementById:function(id){return id==='sn-back'?backBtn:null;}};
var listeners={};
var window={scrollY:420,_ppSubTabSel:null,scrollTo:function(x,y){calls.push(['scroll',y]);},
            addEventListener:function(k,f){listeners[k]=f;}};
var history={pushState:function(){calls.push(['push']);},
             back:function(){calls.push(['back']);listeners.popstate();}};
var location={href:'x'};
function setTimeout(f){f();}
function clearTimeout(){}
function _syncHashNow(){}
function ppHide(){}
function closeDetail(){}
function _ppSubTabCur(){return 'pool';}
function _getCurrentFilterState(){return JSON.parse(JSON.stringify(_filt));}
function _applyFilterState(st){_filt=JSON.parse(JSON.stringify(st));calls.push(['apply']);}
function navGo(v,k){curView=v;if(v==='pool')_poolTab=k;else if(v==='matrix')curMSub=k;calls.push(['nav',v,k]);}
"""
    js = stub + helpers + body
    p = tmp_path / 'drill.js'
    p.write_text(js, encoding='utf-8')
    out = subprocess.run(['node', str(p)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_back_restores_view_section_scroll_and_filters(tmp_path):
    out = _run(tmp_path, r"""
_drillPush('Energy · Profit Pool');
var label=backBtn.textContent;
_filt.colFilters=[{key:'industry',txt:'Oil & Gas E&P'}];curView='matrix';curMSub='matrix';
_drillSeal();window.scrollY=0;
drillBack();
process.stdout.write(JSON.stringify({label:label,after:backBtn.textContent,view:curView,
  tab:_poolTab,pp:window._ppSubTabSel,cf:_filt.colFilters,calls:calls,n:_drillStack.length}));
""")
    assert out['label'] == '‹ Energy · Profit Pool'
    assert out['after'] == ''
    assert (out['view'], out['tab'], out['pp']) == ('pool', 'Energy', 'pool')
    assert out['cf'] == []                        # the drill's filter is gone
    assert ['back'] in out['calls']               # the history entry is spent
    assert out['calls'][-1] == ['scroll', 420]
    assert out['n'] == 0


def test_stack_unwinds_one_hop_at_a_time(tmp_path):
    out = _run(tmp_path, r"""
curView='pool';_poolTab='__performance__';
_drillPush('Sector Heatmap');
_poolTab='Energy';
_drillPush('Energy · Profit Pool');
curView='matrix';
drillBack();var first=[curView,_poolTab,backBtn.textContent];
drillBack();var second=[curView,_poolTab,backBtn.textContent];
process.stdout.write(JSON.stringify([first,second]));
""")
    assert out[0] == ['pool', 'Energy', '‹ Sector Heatmap']
    assert out[1] == ['pool', '__performance__', '']


def test_popstate_without_a_drill_does_nothing(tmp_path):
    out = _run(tmp_path, r"""
listeners.popstate();
_drillPush('Dashboard · BUY');_drillClear();listeners.popstate();
process.stdout.write(JSON.stringify(calls.filter(function(c){return c[0]==='nav';}).length));
""")
    assert out == 0


def test_back_keeps_a_filter_the_user_added_after_the_drill(tmp_path):
    out = _run(tmp_path, r"""
_filt.colFilters=[{key:'pe',min:null,max:20}];
_drillPush('Dashboard · BUY');
_filt.ratings=['BUY'];curView='matrix';_drillSeal();
_filt.search='oil';_filt.colFilters.push({key:'mos',min:0.1,max:null});
drillBack();
process.stdout.write(JSON.stringify(_filt));
""")
    assert 'ratings' not in out                   # the drill's rating filter is undone
    assert out['search'] == 'oil'                 # the user's own edits stay
    assert out['colFilters'] == [{'key': 'pe', 'min': None, 'max': 20},
                                 {'key': 'mos', 'min': 0.1, 'max': None}]


def test_a_filter_the_user_retuned_is_theirs(tmp_path):
    out = _run(tmp_path, r"""
_drillPush('Dashboard · Quality');
_filt.colFilters=[{key:'q_score',min:62,max:null}];curView='matrix';_drillSeal();
_filt.colFilters=[{key:'q_score',min:70,max:null}];
drillBack();
process.stdout.write(JSON.stringify(_filt.colFilters));
""")
    assert out == [{'key': 'q_score', 'min': 70, 'max': None}]


def test_leaving_by_hand_undoes_every_pending_drill(tmp_path):
    out = _run(tmp_path, r"""
_drillPush('Dashboard · BUY');
_filt.ratings=['BUY'];curView='matrix';_drillSeal();
_drillPush('Dashboard · World map');
_filt.countries=['Canada'];_drillSeal();
_drillLeave();
process.stdout.write(JSON.stringify({f:_filt,n:_drillStack.length,back:backBtn.textContent,
  nav:calls.filter(function(c){return c[0]==='nav';}).length}));
""")
    assert out['f'] == {'search': '', 'colFilters': []}
    assert (out['n'], out['back'], out['nav']) == (0, '', 0)


def test_a_sector_page_hop_engages_no_filter(tmp_path):
    out = _run(tmp_path, r"""
_filt.ratings=['HOLD'];
_drillPush('Sector Heatmap');_poolTab='Energy';_drillSeal();
drillBack();
process.stdout.write(JSON.stringify({f:_filt,applied:calls.filter(function(c){return c[0]==='apply';}).length}));
""")
    assert out['f']['ratings'] == ['HOLD']
    assert out['applied'] == 0


def test_detail_drill_returns_to_the_sub_tab_it_came_from(tmp_path):
    """A currency row (or ETF, or map country) opening Detail is a drill: back
    lands on Investing › Currencies' Screener, not the Detail it left."""
    out = _run(tmp_path, r"""
curView='matrix';curMSub='ccy';_FM_SUB.ccy='map';
_drillPush('Map');_FM_SUB.ccy='detail';
var label=backBtn.textContent;
drillBack();
process.stdout.write(JSON.stringify({label:label,sub:_FM_SUB.ccy,calls:calls,n:_drillStack.length}));
""")
    assert out['label'] == '‹ Map'
    assert ['nav', 'matrix', 'ccy'] in out['calls']
    assert out['calls'].index(['nav', 'matrix', 'ccy']) < out['calls'].index(['fm', 'ccy', 'map'])
    assert out['sub'] == 'map' and out['n'] == 0


def test_detail_launchers_push_a_drill_and_the_tab_row_leaves_it():
    src = _tpl()
    assert "_fmDrill('ccy');" in _fn_src('_cxPick', src)
    assert "_fmDrill('etf');" in _fn_src('_etfPick', src)
    assert '_drillLeave();_fmSubTab(view,k);' in _fn_src('_fmTab', src)
    assert 'onclick="_fmTab(' in _fn_src('_fmBar', src)
