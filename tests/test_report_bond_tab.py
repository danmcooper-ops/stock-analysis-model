"""Financial Model › Bonds: the bond model's report
(bond-analysis-model's bond_report_<date>.json) staged beside the page and
drawn in this report's styling. Staging, wiring, the display-only rule, and
the renderer evaluated in Node with its helpers lifted from the template."""
import json
import re
import shutil
import subprocess
from datetime import date
from pathlib import Path

import pytest

from scripts import report_html

TEMPLATE = Path(__file__).resolve().parents[1] / 'templates' / 'report.html'


def _tpl():
    return TEMPLATE.read_text(encoding='utf-8')


def _fn_src(name, src):
    return re.search(r'^function ' + re.escape(name) + r'\(.*?^\}', src, re.S | re.M).group(0)


REPORT = {
    'version': 1, 'run_date': '2026-10-26', 'curve_date': '2026-10-26',
    'generated': '2026-10-26 17:06', 'total': 2, 'capped_pct': 50, 'mark_age': 179,
    'vintage': {'date': '2026-04-30', 'age_days': 179},
    'cards': [{'k': 'Instruments', 'v': '2', 't': 'Every bond priced.'},
              {'k': 'BUY', 'v': '1', 't': 'Top of the ranking.'}],
    'caveats': [{'lead': 'Prices are monthly.', 'text': 'Marks are <old>.'}],
    'ratings': [['BUY', 1], ['HOLD', 1]], 'buckets': [['A', 1]],
    'bucket_tips': {'A': 'Solid investment grade.'},
    'curve': [[0.25, 0.042], [10.0, 0.05]],
    'term': [{'name': 'mid', 'points': [[2.0, 0.7], [10.0, 1.1]]},
             {'name': 'FRED slices', 'points': [[2.0, 0.59]]}],
    'cap_reasons': [['stale mark', 1]],
    'categories': [{'label': 'Valuation', 'n': 2, 'mean': 34.4}],
    'tips': {'ratings': 'r', 'buckets': 'b', 'curve': 'c', 'term': 't', 'caps': 'k',
             'categories': 'g'},
    'notes': {'term': 'Fitted from spreads.', 'categories': 'Dropped, not zero.'},
    'sources': 'SEC Form N-PORT.',
    'columns': [{'k': 'cusip', 'l': 'CUSIP', 'f': 'text', 'w': 90, 't': 'The id.'},
                {'k': 'issuer_name', 'l': 'Issuer', 'f': 'text', 'w': 210, 't': 'Who.'},
                {'k': 'asset_class', 'l': 'Class', 'f': 'cls', 'w': 78, 't': 'Class.'},
                {'k': 'implied_bucket', 'l': 'Model', 'f': 'bucket', 'w': 62, 't': 'm'},
                {'k': 'spread_mispricing', 'l': 'Mispr', 'f': 'bp', 'w': 66, 't': 'Cheap.'},
                {'k': '_composite_score', 'l': 'Comp', 'f': 'n1', 'w': 60, 't': 'c'},
                {'k': 'rating', 'l': 'Rating', 'f': 'rating', 'w': 88, 't': 'r'},
                {'k': '_caps', 'l': 'Caps', 'f': 'caps', 'w': 200, 't': 'Why.'}],
    'rows': [['A1', 'ALPHA CORP', 'CORP_IG', 'A', 0.0212, 86.7, 'BUY', ''],
             ['T1', None, 'TREASURY', None, -0.0003, 40.1, 'HOLD', 'stale mark (210d old)']],
}


def _write_reports(d, dates):
    for day in dates:
        (d / f'bond_report_{day}.json').write_text(
            json.dumps(dict(REPORT, run_date=day)), encoding='utf-8')


def test_stage_picks_newest_on_or_before_the_run(tmp_path):
    src, out = tmp_path / 'bond', tmp_path / 'out'
    src.mkdir()
    out.mkdir()
    _write_reports(src, ['2026-10-23', '2026-10-26', '2026-10-27'])
    meta = report_html._stage_bond_report(str(out), date(2026, 10, 26), str(src))
    assert meta == {'date': '2026-10-26', 'total': 2}
    assert json.loads((out / 'bonds.json').read_text(encoding='utf-8'))['run_date'] == '2026-10-26'


def test_stage_removes_a_stale_copy_when_there_is_nothing_to_show(tmp_path):
    src, out = tmp_path / 'bond', tmp_path / 'out'
    src.mkdir()
    out.mkdir()
    (out / 'bonds.json').write_text('{}', encoding='utf-8')
    _write_reports(src, ['2026-10-27'])
    assert report_html._stage_bond_report(str(out), date(2026, 10, 26), str(src)) is None
    assert not (out / 'bonds.json').exists()
    assert report_html._stage_bond_report(str(out), date(2026, 10, 26),
                                          str(tmp_path / 'missing')) is None


def test_stage_refuses_another_shape(tmp_path):
    src, out = tmp_path / 'bond', tmp_path / 'out'
    src.mkdir()
    out.mkdir()
    (src / 'bond_report_2026-10-26.json').write_text(json.dumps(dict(REPORT, version=2)),
                                                     encoding='utf-8')
    assert report_html._stage_bond_report(str(out), date(2026, 10, 26), str(src)) is None
    assert not (out / 'bonds.json').exists()


def test_subtab_is_wired():
    src = _tpl()
    assert "{k:'bond',l:'Bonds'}" in src
    assert '<div id="m-bond-view" style="display:none"></div>' in src
    show = _fn_src('_mShow', src)
    assert "k==='bond'" in show and 'renderBondView()' in show
    assert "var BOND_META={{ bond_meta|default('null') }};" in src
    assert "_loadSidecar('bonds.json'," in _fn_src('renderBondView', src)


def test_publish_ships_the_sidecar():
    sh = (TEMPLATE.parents[1] / 'scripts' / 'run_daily.sh').read_text(encoding='utf-8')
    assert 'cp output/bonds.json "$docs/bonds.json"' in sh
    assert 'rm -f "$docs/bonds.json"' in sh


def test_display_only_never_writes_data():
    src = _tpl()
    body = ''.join(_fn_src(n, src) for n in
                   ('renderBondView', '_bvHtml', '_bvRows', '_bvDraw', '_bvSet', '_bvReset'))
    assert not re.search(r'\b(DATA|FD)\b', body)


def _render(report, after=''):
    if shutil.which('node') is None:
        pytest.skip('node not installed')
    src = _tpl()
    attr = re.search(r'^function _attr\(s\)\{.*?\}$', src, re.M).group(0)
    rb = re.search(r'^function rb\(r,d\)\{.*?\}$', src, re.M).group(0)
    rc = re.search(r"^var RC=\{.*?\};$", src, re.M).group(0)
    start = src.index('var _BV=null,')
    block = src[start:src.index('function renderBondView(){', start)]
    fns = [_fn_src(n, src) for n in ('_esc', '_bvCls', '_bvFmt', '_bvDist', '_bvLine',
                                     '_bvCard', '_bvHtml', '_bvIdx', '_bvRows')]
    js = '\n'.join([
        'function isCapped(){return false;}', attr, rb, rc, block, *fns,
        f'_BV={json.dumps(report)};',
        after or 'process.stdout.write(_bvHtml(_BV));'])
    return subprocess.run(['node', '-e', js], capture_output=True, text=True, check=True).stdout


def test_render_draws_every_section_from_the_report():
    html = _render(REPORT)
    assert 'Run 2026-10-26 · curve 2026-10-26 · 2 instruments' in html
    assert 'title="Every bond priced."><span>Instruments</span><b>2</b>' in html
    assert 'Marks are 179 days old' in html and '2026-04-30' in html
    assert '<b>Prices are monthly.</b> Marks are &lt;old&gt;.' in html
    for t in ('Rating distribution', 'Implied credit bucket', 'Treasury par curve',
              'Spread term structure, fitted vs FRED', 'Why rows are capped',
              'Mean score by category', 'All 2 Instruments'):
        assert t in html
    assert 'title="Solid investment grade."' in html
    assert 'Fitted from spreads.' in html and 'Dropped, not zero.' in html
    assert html.count('<svg class="bv-svg"') == 2
    assert '<option value="CORP_IG">IG</option>' in html


def test_rows_filter_and_sort_like_the_bond_page():
    out = _render(REPORT, after=(
        "var r=[];"
        "r.push(_bvRows().map(function(x){return x[0];}).join(','));"
        "_bvF.u=true;r.push(_bvRows().length);_bvF.u=false;"
        "_bvF.q='alpha';r.push(_bvRows().length);_bvF.q='';"
        "_bvF.c='TREASURY';r.push(_bvRows()[0][0]);_bvF.c='';"
        "_bvSortK='spread_mispricing';_bvAsc=true;r.push(_bvRows()[0][0]);"
        "r.push(_bvFmt(0.0212,'bp'),_bvFmt(-0.0003,'bp'),_bvFmt(null,'n1'),_bvFmt('TREASURY_BILL','cls'));"
        "process.stdout.write(JSON.stringify(r));"))
    r = json.loads(out)
    assert r[0] == 'A1,T1'            # composite, high to low
    assert r[1] == 1 and r[2] == 1 and r[3] == 'T1' and r[4] == 'T1'
    assert r[5] == '<span class="bv-pos">212</span>'
    assert r[6] == '<span class="bv-neg">-3</span>'
    assert 'bv-mut' in r[7] and r[8] == 'T-BILL'
