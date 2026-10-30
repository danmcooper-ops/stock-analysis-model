"""Charts › Stock Comparison: one dropdown picks the chart's metric, listing
every numeric column of the table's column sets plus the filed history
(Financials, after Valuation). A metric with a filed history charts over time
(↗), any other today's values as bars. There is no comparison table.
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


def test_picker_renders_with_the_chart_and_no_table():
    src = _src()
    # The metric, the stock pins and the range all sit in the hamburger panel.
    panel = re.search(r"id=\"ph-cm-panel\">'(.*?)\+'<div id=\"ph-tip-bar\"", src, re.S).group(1)
    assert 'id="ph-metric"' in panel and 'id="ph-chips"' in panel and 'id="ph-ranges"' in panel
    assert re.search(r'function renderPHChart\(\)\{\s*_renderPHMetricSel\(\);', src)
    assert 'ph-cmp' not in src and '_renderPHCompare' not in src


@node
def test_metric_dropdown(tmp_path):
    stubs = r"""
var TL={prices:'Charts',val:'Valuation',prof:'Profitability'};
var TC={prof:[{k:'ticker',f:'text'},{k:'rating',f:'rating'},{k:'roic',l:'ROIC',f:'pct'},{k:'pe',l:'P/E',f:'n1'},{k:'price',l:'Price',f:'dollar'},{k:'roe',l:'ROE',f:'pct'},{k:'sector',l:'Sector',f:'text'}],
        val:[{k:'ticker',f:'text'},{k:'pe',l:'P/E',f:'n1'}]};
var _PH_TC2H={price:'price',roe:'roe'};
var _PH_METRICS=[{k:'price',lbl:'Price',src:'price'},{k:'rev',lbl:'Revenue',src:'hist'},{k:'roe',lbl:'ROE',src:'hist'}];
var _phMetric='s:roic',_HIST_AVAILABLE=true,_HIST_TICKERS=['A'],PRICES=null;
function _phMetricDef(){return {src:'snap'};}
function esc(s){return String(s);} function _attr(s){return String(s);}
var el={innerHTML:'',style:{},classList:{toggle:function(){}}};var document={getElementById:function(){return el;}};
"""
    script = tmp_path / 't.js'
    script.write_text(stubs + _fn('_phCats') + '\n' + _fn('_phCatLbl') + '\n'
                      + _fn('_renderPHMetricSel') + '\n'
                      + "_renderPHMetricSel();console.log(el.innerHTML);", encoding='utf-8')
    r = subprocess.run(['node', str(script)], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    h = r.stdout
    assert 'onchange="phSetMetric(this.value)"' in h and '<table' not in h
    groups = re.findall(r'<optgroup label="([^"]+)">', h)
    assert groups == ['Valuation', 'Financials', 'Profitability']
    values = re.findall(r'<option value="([^"]+)"', h)
    # Price leads; each metric once, under its first tab; text/rating columns left out.
    assert values == ['price', 's:pe', 'rev', 'roe', 's:roic']
    assert '<option value="s:roic" selected>ROIC</option>' in h
    assert 'Revenue ↗' in h
