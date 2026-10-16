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


def test_manual_navigation_and_clear_empty_the_stack():
    src = _tpl()
    i = src.index("e.target.closest('#sidebar [data-v],#subnav [data-v]')")
    assert '_drillClear();' in src[i:i + 900]
    for name in ('_hkGo', '_swipeTab', 'clearFilters'):
        assert '_drillClear();' in _fn_src(name, src), name
    # drillBack restores through navGo, so navGo itself must not clear.
    assert '_drillClear' not in _fn_src('navGo', src)


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
    curPfSub='overview',curTSub='prices',_hashTimer=null,_filt={ind:null};
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
function _getCurrentFilterState(){return {ind:_filt.ind};}
function _applyFilterState(st){_filt.ind=st.ind;}
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
_filt.ind='Oil & Gas E&P';curView='matrix';curMSub='matrix';window.scrollY=0;
drillBack();
process.stdout.write(JSON.stringify({label:label,after:backBtn.textContent,view:curView,
  tab:_poolTab,pp:window._ppSubTabSel,ind:_filt.ind,calls:calls,n:_drillStack.length}));
""")
    assert out['label'] == '‹ Energy · Profit Pool'
    assert out['after'] == ''
    assert (out['view'], out['tab'], out['pp']) == ('pool', 'Energy', 'pool')
    assert out['ind'] is None                     # the drill's filter is gone
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
