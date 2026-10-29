"""Charts › Compare by Category: the charted stocks side by side over any of
the table's column sets, with the best and worst of them marked per metric.
"""
import os
import re
import shutil
import subprocess

import pytest

TEMPLATE = os.path.join(os.path.dirname(__file__), '..', 'templates', 'report.html')
node = pytest.mark.skipif(shutil.which('node') is None, reason='node not installed')


def _src():
    return open(TEMPLATE, encoding='utf-8').read()


def _fn(name):
    return re.search(r'^function ' + re.escape(name) + r'\(.*?^\}', _src(), re.S | re.M).group(0)


def test_compare_renders_with_the_chart():
    src = _src()
    assert "+'<div id=\"ph-cmp\"></div>'" in src
    assert re.search(r'function renderPHChart\(\)\{\s*_renderPHCompare\(\);', src)


@node
def test_compare_table(tmp_path):
    stubs = r"""
var _phPalette=['#1','#2','#3'],DV_POS={pe:1},DV_NORANK={price:1};
var DV_DIR={roic:1,pe:-1};
var TL={prices:'Charts',val:'Valuation',prof:'Profitability'};
var TC={prof:[{k:'ticker',f:'text'},{k:'rating',f:'rating'},{k:'roic',l:'ROIC',f:'pct'},{k:'pe',l:'P/E',f:'n1'},{k:'price',l:'Price',f:'dollar'},{k:'gone',l:'Gone',f:'n1'}],
        val:[{k:'ticker',f:'text'},{k:'pe',l:'P/E',f:'n1'}]};
var DATA=[{ticker:'A',rating:'BUY',_composite_score:70,roic:0.3,pe:-5,price:10},
          {ticker:'B',rating:'HOLD',_composite_score:40,roic:0.1,pe:20,price:20},
          {ticker:'C',rating:'PASS',_composite_score:10,roic:0.2,pe:12,price:30}];
var _phStocks=['A','SPY','B','C'],_phCat='prof';
function esc(s){return String(s);} function _attr(s){return String(s);}
function tt(){return '';} function _dvTipAttr(){return '';}
function rb(v){return '<b>'+v+'</b>';}
function ftc(v,f){if(v==null)return{t:'N/A'};if(f==='pct')return{t:(v*100).toFixed(1)+'%'};if(f==='rating')return{t:rb(v)};return{t:String(v)};}
function _dvNum(v){return typeof v==='number'&&isFinite(v)?v:null;}
function _dvVals(k){return DATA.map(function(d){return d[k];}).filter(function(v){return typeof v==='number'&&!(DV_POS[k]&&v<=0);}).sort(function(a,b){return a-b;});}
function _dvMedian(a){if(!a.length)return null;var m=a.length>>1;return a.length%2?a[m]:(a[m-1]+a[m])/2;}
var el={innerHTML:''};var document={getElementById:function(){return el;}};
var localStorage={setItem:function(){}};
"""
    script = tmp_path / 't.js'
    script.write_text(stubs + _fn('_phCats') + '\n' + _fn('phSetCat') + '\n'
                      + _fn('_renderPHCompare') + '\n'
                      + "_renderPHCompare();console.log(el.innerHTML);", encoding='utf-8')
    r = subprocess.run(['node', str(script)], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    h = r.stdout
    # Tabs follow the table's own order; Charts is not a category.
    assert h.index('Valuation') < h.index('Profitability') and 'Charts' not in h
    # The index (no row) is left out; the stocks keep their chart colours.
    assert 'SPY' not in h
    assert re.findall(r"openDet\('(\w+)'\)", h) == ['A', 'B', 'C']
    rows = dict(re.findall(r'<tr><td>([^<]+)</td>(.*?)</tr>', h))
    # Higher ROIC is better: A best, B worst.
    roic = re.findall(r'<td( class="[^"]+")?>([^<]*)</td>', rows['ROIC'])
    assert roic[0] == (' class="sa-chrome ph-best"', '30.0%')
    assert roic[1] == (' class="sa-chrome ph-worst"', '10.0%')
    assert roic[3][1] == '20.0%'  # the run's median
    # A negative P/E is not "cheapest": C best, B worst, A unmarked.
    pe = re.findall(r'<td( class="[^"]+")?>([^<]*)</td>', rows['P/E'])
    assert pe[0] == ('', '-5') and pe[2][0].endswith('ph-best"') and pe[1][0].endswith('ph-worst"')
    # Price has no direction and no median; a column every stock lacks is dropped.
    assert 'ph-' not in rows['Price'] and 'Gone' not in rows
    assert '<b>BUY</b>' in rows['Rating']
