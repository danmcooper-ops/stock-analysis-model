# tests/test_report_live_quote.py
"""The report's live-quote overlay (the `_lq*` block in templates/report.html).

On the Mac mini's LAN copy (scripts/live_quotes.py) the page polls
api/quote for what is on screen. These pin the invariants that would break
silently: the overlay never writes DATA (so filters, sorts, portfolio rules
and the Summary PDF keep the close), a live cell keeps the close as its sort
value, MoS is recomputed on the scoring rule, stale trades say so, a host
without the API switches it off for good, and a newer build is announced.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

TEMPLATE = Path(__file__).resolve().parents[1] / 'templates' / 'report.html'

pytestmark = pytest.mark.skipif(shutil.which('node') is None, reason='node not installed')


def _src():
    return TEMPLATE.read_text(encoding='utf-8')


def _block():
    s = _src()
    start = s.index('/* --- Live quotes ----')
    end = s.index("document.addEventListener('visibilitychange'", start)
    return s[start:s.index('\n', end) + 1]


def _fn(name):
    return re.search(r'^function ' + name + r'\(.*?^\}', _src(), re.S | re.M).group(0)


# Minimal browser stand-ins; `ELS` holds the elements the block touches.
STUBS = r"""
var ELS={}, APPENDED=[], TIMERS=[], META=LOC_META;
function El(id){this.id=id;this.textContent='';this.className='';this.title='';this.style={};
  var cl={};this.classList={add:function(c){cl[c]=1;},remove:function(c){delete cl[c];},
  contains:function(c){return !!cl[c];}};this.removeAttribute=function(a){this[a]='';};
  this.setAttribute=function(a,v){this[a]=v;};this.querySelector=function(){return null;};}
var document={hidden:false,addEventListener:function(){},
  getElementById:function(id){return ELS[id]||null;},
  querySelector:function(sel){if(sel.indexOf('live-quotes')>=0)return META;
    var m=sel.match(/data-lq="(\w+)"/);return m?(ELS['km-'+m[1]]||null):null;},
  createElement:function(){return new El('');},
  body:{appendChild:function(e){APPENDED.push(e.id);ELS[e.id]=e;}}};
var location={protocol:LOC_PROTO,pathname:LOC_PATH};
function setTimeout(f,ms){TIMERS.push(ms);return TIMERS.length;}
function clearTimeout(){}
var FETCH_Q=[], FETCH_URLS=[];
function fetch(url){FETCH_URLS.push(url);var r=FETCH_Q.shift();
  if(r instanceof Error)return Promise.reject(r);
  return Promise.resolve({ok:r.status===200,status:r.status,json:function(){return Promise.resolve(r.body);}});}
var curView='table',curTSub='val',_pfm={id:null},_dpTicker=null,RENDERS={dt:0,pf:0};
function _renderDtWindow(){RENDERS.dt++;_lqSet('tbl',_lqTargets.tbl.slice());}
function _pfmRedraw(){RENDERS.pf++;}
function esc(s){return String(s).replace(/"/g,'&quot;');}
function _pfEsc(s){return esc(s);}
function _starHtml(){return'';}function _revHtml(){return'';}function _pfDots(){return'';}
function rb(v){return v;}function fds(v){return v;}
var DATA=[{ticker:'AAA',price:80,mos:0.2,_fv_effective:100,dcf_fv:90},
          {ticker:'BBB',price:10,mos:null,_fv_effective:null,dcf_fv:20}];
var NOW=Date.UTC(2026,9,21,15,0);   // 11:00 ET
function Q(c,t){return{c:c,d:1.25,dp:0.0125,t:t};}
async function flush(){for(var i=0;i<5;i++)await Promise.resolve();}
"""


def _run(body, tmp_path, path='/b/100/', extra=(), proto='http:', meta=False):
    js = (STUBS.replace('LOC_PATH', json.dumps(path)).replace('LOC_PROTO', json.dumps(proto))
          .replace('LOC_META', '{}' if meta else 'null') + '\n'
          + '\n'.join(extra) + '\n' + _block()
          + '\n(async function(){var R=await (async function(){' + body + '})();'
          + 'process.stdout.write(JSON.stringify(R));})();')
    p = tmp_path / 'lq.js'
    p.write_text(js, encoding='utf-8')
    r = subprocess.run(['node', str(p)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def test_block_never_assigns_to_row_fields():
    # The overlay may read DATA rows but never write them: filters, sorting,
    # _pfRuleMatch, portfolio stats and the Summary PDF all read DATA.
    assert not re.search(r'\bd\.\w+\s*=[^=]', _block())
    assert not re.search(r'DATA\[[^\]]*\]\.\w+\s*=[^=]', _block())


def test_mos_follows_the_scoring_rule(tmp_path):
    out = _run("""
      return [_lqMos(DATA[0],80), _lqMos(DATA[0],125), _lqMos(DATA[1],10),
              _lqMos({_fv_effective:null,dcf_fv:null},10), _lqMos(DATA[0],0)];
    """, tmp_path)
    # (fv - px) / fv on _fv_effective, falling back to dcf_fv (scoring.py:836).
    assert out[0] == pytest.approx(0.2)
    assert out[1] == pytest.approx(-0.25)
    assert out[2] == pytest.approx(0.5)
    assert out[3:] == [None, None]


def test_poll_fills_overlay_without_touching_data(tmp_path):
    out = _run("""
      var before=JSON.stringify(DATA);
      _lqSet('tbl',['AAA','BBB']);
      FETCH_Q.push({status:200,body:{open:true,build:'100',asof:NOW,
        symbols:{AAA:Q(90,NOW-60000),BBB:null}}});
      _lqPoll();await flush();
      return {same:JSON.stringify(DATA)===before, price:_lqView(DATA[0],'price'),
              mos:_lqView(DATA[0],'mos'), none:_lqView(DATA[1],'price')===undefined,
              url:FETCH_URLS[0], next:TIMERS[TIMERS.length-1], dt:RENDERS.dt,
              tip:_lqTip(DATA[0],'price')};
    """, tmp_path)
    assert out['same'] is True
    assert out['price'] == 90 and out['mos'] == pytest.approx(0.1)
    assert out['none'] is True
    assert out['url'] == 'api/quote?symbols=AAA%2CBBB'
    assert out['next'] == 15000 and out['dt'] == 1
    assert out['tip'].startswith('Live') and 'Close: $80' in out['tip']


def test_stale_trade_is_labelled(tmp_path):
    out = _run("""
      _lqSet('tbl',['AAA']);
      FETCH_Q.push({status:200,body:{open:true,build:'100',asof:NOW,
        symbols:{AAA:Q(90,NOW-16*60000)}}});
      _lqPoll();await flush();
      return [_LQ.AAA.stale, _lqTip(DATA[0],'price')];
    """, tmp_path)
    assert out[0] is True and out[1].startswith('Last trade ')


def test_closed_market_polls_slowly(tmp_path):
    out = _run("""
      _lqSet('tbl',['AAA']);
      FETCH_Q.push({status:200,body:{open:false,build:'100',asof:NOW,symbols:{AAA:null}}});
      _lqPoll();await flush();return TIMERS[TIMERS.length-1];
    """, tmp_path)
    assert out == 300000


@pytest.mark.parametrize('first', ['{status:404,body:null}', "new Error('net')",
                                   "{status:200,body:'<html>'}"])
def test_host_without_the_api_switches_off(tmp_path, first):
    out = _run(f"""
      _lqSet('tbl',['AAA']);FETCH_Q.push({first});
      _lqPoll();await flush();
      var n=FETCH_URLS.length;_lqKick();_lqPoll();
      return [_lqOff, FETCH_URLS.length===n, _lqView(DATA[0],'price')===undefined];
    """, tmp_path)
    assert out == [True, True, True]


def test_blip_after_success_keeps_quotes(tmp_path):
    out = _run("""
      _lqSet('tbl',['AAA']);
      FETCH_Q.push({status:200,body:{open:true,build:'100',asof:NOW,symbols:{AAA:Q(90,NOW)}}});
      _lqPoll();await flush();
      FETCH_Q.push(new Error('wifi'));_lqPoll();await flush();
      return [_lqOff, _lqView(DATA[0],'price'), TIMERS[TIMERS.length-1]];
    """, tmp_path)
    assert out == [False, 90, 300000]


def test_file_url_never_fetches(tmp_path):
    out = _run("""
      _lqSet('tbl',['AAA']);_lqPoll();
      return [_lqOff, FETCH_URLS.length];
    """, tmp_path, proto='file:')
    assert out == [True, 0]


def test_new_build_is_announced_once(tmp_path):
    out = _run("""
      _lqSet('tbl',['AAA']);
      for(var i=0;i<2;i++){
        FETCH_Q.push({status:200,body:{open:true,build:'200',asof:NOW,symbols:{AAA:null}}});
        _lqPoll();await flush();_lqBusy=false;
      }
      return [_LQ_BUILD, APPENDED];
    """, tmp_path)
    assert out == ['100', ['lq-new']]


def test_only_on_screen_targets_are_asked_for(tmp_path):
    out = _run("""
      ELS['det-modal']=new El('det-modal');
      _lqTargets={det:['DDD'],tbl:['AAA','BBB'],pf:['PPP']};
      var r=[];
      r.push(_lqWant());                                  // table view, no popup
      ELS['det-modal'].classList.add('open');r.push(_lqWant());
      curView='pf';_pfm.id='x';r.push(_lqWant());
      curView='table';curTSub='prices';ELS['det-modal'].classList.remove('open');r.push(_lqWant());
      return r;
    """, tmp_path)
    assert out == [['AAA', 'BBB'], ['DDD', 'AAA', 'BBB'], ['DDD', 'PPP'], []]


def test_table_cell_shows_live_but_sorts_on_close(tmp_path):
    out = _run("""
      _LQ.AAA={c:90,d:1,dp:0.01,t:NOW,stale:false};
      _dtCols=[{k:'price',f:'dollar'},{k:'mos',f:'pct'}];
      return _dtBuildRow(DATA[0],0);
    """, tmp_path, extra=[_fn('ftc'), _fn('_dtBuildRow'), 'var _dtCols;'])
    cells = re.findall(r'<td([^>]*)>([^<]*)</td>', out)
    price, mos = cells[-2], cells[-1]
    assert 'data-value="80"' in price[0] and price[1] == '$90' and 'lq-live' in price[0]
    assert 'data-value="0.2"' in mos[0] and mos[1] == '10.0%'


def test_ticker_page_paint(tmp_path):
    out = _run("""
      ['d-px','d-chg','d-lq','km-price','km-mos'].forEach(function(id){ELS[id]=new El(id);});
      var n=new El('n');ELS['km-mos'].querySelector=function(){return n;};
      _dpTicker='AAA';_LQ.AAA={c:90,d:-1.5,dp:-0.0164,t:NOW,stale:false};
      _lqPaintDet('AAA');
      var r={px:ELS['d-px'].textContent,chg:ELS['d-chg'].textContent,
             stamp:ELS['d-lq'].textContent,mos:n.textContent,
             live:ELS['d-px'].classList.contains('lq-live')};
      delete _LQ.AAA;_lqPaintDet('AAA');
      r.cleared=!ELS['d-px'].classList.contains('lq-live')&&ELS['d-lq'].textContent==='';
      _dpTicker='ZZZ';ELS['d-px'].textContent='kept';_LQ.AAA={c:1,d:0,dp:0,t:NOW};
      _lqPaintDet('AAA');r.guard=ELS['d-px'].textContent;
      return r;
    """, tmp_path)
    assert out['px'] == '$90.00' and out['chg'] == '−1.50 (−1.64%)'
    assert out['stamp'].startswith('Live · ') and out['stamp'].endswith(' ET')
    assert out['mos'] == '10.0%' and out['live'] is True
    assert out['cleared'] is True
    assert out['guard'] == 'kept'          # a late answer for another ticker


def test_hooks_are_wired():
    s = _src()
    assert '_detDayChg(tk);\n  _lqDet(tk);' in s
    assert '_detOpenState(false);_lqDet(null);}' in _fn('closeDetail')
    assert "_lqSet('tbl'," in _fn('_renderDtWindow')
    assert "_lqSet('pf'," in _fn('_pfmTable')
    assert '_lqView(d,c.k)' in _fn('_pfmCell')
    assert 'data-lq="\'+r[4]+\'"' in s and "'','','price']" in s and "'','mos']" in s
    assert 'id="d-lq"' in s


def test_one_timer_per_poll_and_idle_views_resume(tmp_path):
    out = _run("""
      _lqSet('tbl',['AAA']);var kicks=TIMERS.length;       // 250 ms kick
      FETCH_Q.push({status:200,body:{open:true,build:'100',asof:NOW,symbols:{AAA:Q(90,NOW)}}});
      _lqPoll();await flush();
      // The repaint re-rendered the table (same rows): no extra kick.
      var afterPoll=TIMERS.slice(kicks);
      // Leave the table: the next poll finds nothing and goes idle.
      curView='macro';_lqPoll();
      // Come back to the same rows: polling resumes.
      curView='table';var n=TIMERS.length;_lqSet('tbl',['AAA']);
      return [kicks, afterPoll, TIMERS.length-n, RENDERS.dt];
    """, tmp_path)
    assert out == [1, [15000], 1, 1]


@pytest.mark.parametrize('path', ['/', '/index.html', '/stock-report/'])
def test_hosted_sites_never_ask(tmp_path, path):
    # Cloudflare and GitHub Pages do not serve /b/<build>/ paths.
    out = _run("""
      _lqSet('tbl',['AAA']);_lqPoll();
      return [_lqOff, FETCH_URLS.length];
    """, tmp_path, path=path)
    assert out == [True, 0]


def test_cloudflare_worker_tag_turns_quotes_on(tmp_path):
    # The Worker adds <meta name="live-quotes"> when it can answer api/quote;
    # its pages are not under /b/<build>/ and carry no build to compare.
    out = _run("""
      _lqSet('tbl',['AAA']);
      FETCH_Q.push({status:200,body:{open:true,asof:NOW,symbols:{AAA:Q(90,NOW)}}});
      _lqPoll();await flush();
      return [_lqOff, FETCH_URLS[0], _lqView(DATA[0],'price'), APPENDED];
    """, tmp_path, path='/', meta=True)
    assert out == [False, 'api/quote?symbols=AAA', 90, []]


def test_left_out_symbols_keep_their_quote(tmp_path):
    # The Worker leaves out symbols it did not fetch this time.
    out = _run("""
      _lqSet('tbl',['AAA','BBB']);
      FETCH_Q.push({status:200,body:{open:true,asof:NOW,symbols:{AAA:Q(90,NOW),BBB:Q(11,NOW)}}});
      _lqPoll();await flush();
      FETCH_Q.push({status:200,body:{open:true,asof:NOW,symbols:{AAA:Q(91,NOW)}}});
      _lqPoll();await flush();
      return [_lqView(DATA[0],'price'), _lqView(DATA[1],'price')];
    """, tmp_path)
    assert out == [91, 11]
