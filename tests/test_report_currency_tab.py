"""Investing › Currencies: the currency screener's report (currencies/'s
currency_report_<date>.json) staged beside the page and drawn in this
report's styling. Staging, wiring, the display-only rule, and the renderer
evaluated in Node with its helpers lifted from the template."""
import json
import re
import shutil
import subprocess
from datetime import date
from pathlib import Path

import pytest

from scripts import report_html

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / 'templates' / 'report.html'


def _tpl():
    return TEMPLATE.read_text(encoding='utf-8')


def _fn_src(name, src):
    return re.search(r'^function ' + re.escape(name) + r'\(.*?^\}', src, re.S | re.M).group(0)


GATES = [
    {'name': 'Valuation: REER Z-Score', 'category': 'Valuation', 'field': 'reer_z20',
     'label': 'REER Z-Score', 'threshold': '< -0.5', 'fmt': 'num', 'tip': 'Cheapness.',
     'key': '_gate_reer_z', 'gpKey': '_gp_reer_z', 'scoreKey': '_gs_reer_z'},
    {'name': 'Carry: Real Carry', 'category': 'Carry', 'field': 'real_carry',
     'label': 'Real Carry', 'threshold': '> +1pp', 'fmt': 'pp', 'tip': 'What it pays.',
     'key': '_gate_real_carry', 'gpKey': '_gp_real_carry', 'scoreKey': '_gs_real_carry'},
]
REPORT = {
    'version': 1, 'run_date': '2026-10-27', 'generated': '2026-10-27 17:02', 'total': 2,
    'gate_meta': {
        'gates': GATES,
        'categories': [
            {'name': 'Valuation', 'weight': 0.3, 'dark': '#123', 'light': '#abc',
             'scoreKey': '_score_pillar_valuation', 'tip': 'Cheap or rich.'},
            {'name': 'Carry', 'weight': 0.25, 'dark': '#456', 'light': '#def',
             'scoreKey': '_score_pillar_carry', 'tip': 'Pays to hold.'}],
        'ui': {'rating': 'Bands.', 'regime': 'Float or peg.', 'na': 'Two kinds.'},
        'ratings': ['LONG', 'LEAN LONG', 'NEUTRAL', 'AVOID'],
        'thresholds': {'long': 57, 'lean': 45, 'neutral': 30}},
    'provenance': {'stale_slots': 0},
    'world_svg': '<svg><path data-iso="JPN" data-name="Japan"/><path data-iso="FRA"/></svg>',
    'rows': [
        {'code': 'JPY', 'name': 'Japanese Yen', 'regime': 'float', 'rating': 'LONG',
         'rating_raw': 'LONG', '_composite_score': 59.0, 'reer_z20': -1.68,
         '_gate_reer_z': -1.68, '_gp_reer_z': True, '_gs_reer_z': 92,
         '_gate_real_carry': -2.8, '_gp_real_carry': False, '_gs_real_carry': 17,
         '_score_pillar_valuation': 89, '_score_pillar_carry': 15, 'map_iso3': ['JPN']},
        {'code': 'HKD', 'name': 'Hong Kong Dollar', 'regime': 'peg', 'rating': 'NEUTRAL',
         'rating_raw': 'LONG', '_rating_cap': 'NEUTRAL', '_rating_cap_reasons': ['Pegged <regime>'],
         '_composite_score': 65.5, 'reer_z20': None, '_gp_reer_z': None,
         '_gate_real_carry': 1.9, '_gp_real_carry': True, '_gs_real_carry': 70,
         '_score_pillar_valuation': None, '_score_pillar_carry': 41, 'map_iso3': ['HKG']},
    ],
}


def _write(d, dates, **over):
    for day in dates:
        (d / f'currency_report_{day}.json').write_text(
            json.dumps(dict(REPORT, run_date=day, **over)), encoding='utf-8')


def test_stage_picks_newest_on_or_before_the_run(tmp_path):
    src, out = tmp_path / 'ccy', tmp_path / 'out'
    src.mkdir()
    out.mkdir()
    _write(src, ['2026-10-23', '2026-10-27', '2026-10-28'])
    meta = report_html._stage_currency_report(str(out), date(2026, 10, 27), str(src))
    assert meta == {'date': '2026-10-27', 'total': 2,
                    'index': [['JPY', 'Japanese Yen'], ['HKD', 'Hong Kong Dollar']]}
    assert json.loads((out / 'currencies.json').read_text(encoding='utf-8'))['run_date'] == '2026-10-27'


def test_stage_removes_a_stale_copy_and_refuses_another_shape(tmp_path):
    src, out = tmp_path / 'ccy', tmp_path / 'out'
    src.mkdir()
    out.mkdir()
    (out / 'currencies.json').write_text('{}', encoding='utf-8')
    _write(src, ['2026-10-27'], version=2)
    assert report_html._stage_currency_report(str(out), date(2026, 10, 27), str(src)) is None
    assert not (out / 'currencies.json').exists()
    assert report_html._stage_currency_report(str(out), date(2026, 10, 27),
                                              str(tmp_path / 'missing')) is None


def test_default_sources_are_the_subprojects(monkeypatch):
    monkeypatch.delenv('BOND_MODEL_OUTPUT', raising=False)
    monkeypatch.delenv('CURRENCY_MODEL_OUTPUT', raising=False)
    assert Path(report_html._bond_output_dir()) == ROOT / 'bonds' / 'output'
    assert Path(report_html._currency_output_dir()) == ROOT / 'currencies' / 'output'


def test_subtab_is_wired():
    src = _tpl()
    assert "{k:'ccy',l:'Currencies'}" in src
    assert '<div id="m-ccy-view" style="display:none"></div>' in src
    show = _fn_src('_mShow', src)
    assert "k==='ccy'" in show and 'renderCcyView()' in show
    assert "var CCY_META={{ ccy_meta|default('null') }};" in src
    assert "_loadSidecar('currencies.json'," in _fn_src('renderCcyView', src)


def test_publish_ships_the_sidecar():
    sh = (ROOT / 'scripts' / 'run_daily.sh').read_text(encoding='utf-8')
    assert 'cp output/currencies.json "$docs/currencies.json"' in sh
    assert 'rm -f "$docs/currencies.json"' in sh


def test_display_only_never_writes_data():
    src = _tpl()
    start = src.index('var _CX=null,')
    block = src[start:src.index('\n/* A sector page', start)]
    assert not re.search(r'\b(DATA|FD)\b', block)


def _render(after):
    if shutil.which('node') is None:
        pytest.skip('node not installed')
    src = _tpl()
    start = src.index('var _CX=null,')
    block = src[start:src.index('\n/* A sector page', start)]
    pick = [re.search(p, src, re.M).group(0) for p in (
        r'^function _attr\(s\)\{.*?\}$', r"^var RC=\{.*?\};$")]
    fm = src[src.index('var _FM_SUB='):src.index('function _fmApply(')]
    # The tab's search is the header's #f-search; Q stands in for it.
    stub = ("var Q='';var document={getElementById:function(id){return id==='f-search'?{value:Q}:null;},"
            "documentElement:{classList:{contains:function(){return false;}}}};")
    js = '\n'.join([stub, _fn_src('_esc', src), *pick, fm, block,
                    f'_CX=_cxPrep({json.dumps(REPORT)});', after])
    return subprocess.run(['node', '-e', js], capture_output=True, text=True, check=True).stdout


def test_render_draws_the_header_tiles_and_views():
    html = _render('process.stdout.write(_cxHtml(_CX));')
    assert 'Run 2026-10-27 · 2 currencies ranked rich to cheap on 2 gates in 2 weighted pillars' in html
    assert '<span>LONG</span><b>1</b>' in html and '<span>NEUTRAL</span><b>1</b>' in html
    assert '<span>Top ranked</span><b>HKD · 66</b>' in html
    assert '<span>Capped</span><b>1</b>' in html
    for v in ('Screener', 'Gate Matrix', 'Map', 'Detail', 'Summary'):
        assert '>' + v + '</button>' in html
    # Opens on the table: the grid pane serves the three table-like tabs;
    # the tiles and notes are the Summary tab's.
    assert re.findall(r'<div data-pane="([\w ]+)"', html) == ['table matrix map', 'detail', 'summary']
    grid = html.split('<div data-pane="table matrix map">')[1].split('data-pane="detail"')[0]
    assert 'pp-tile' not in grid and 'bv-note' not in grid


def test_table_matrix_and_detail():
    out = _render(
        "var r={};"
        "r.table=_cxTable(_cxRows());"
        "r.matrix=_cxMatrix(_cxRows());"
        "r.order=_cxRows().map(function(x){return x.code;});"
        "_cxF.fl=true;r.float=_cxRows().map(function(x){return x.code;});_cxF.fl=false;"
        "Q='yen';r.q=_cxRows().map(function(x){return x.code;});Q='';"
        "_cxSortK='reer_z20';_cxDir=1;r.nulls=_cxRows().map(function(x){return x.code;});"
        "process.stdout.write(JSON.stringify(r));")
    r = json.loads(out)
    assert r['order'] == ['HKD', 'JPY']        # composite, high to low
    assert r['float'] == ['JPY']
    assert r['q'] == ['JPY']                     # the header search matches the name
    assert r['nulls'] == ['JPY', 'HKD']        # missing sorts last either way
    t = r['table']
    assert 'class="fixed-header"' in t and 'data-col-tip-body="Cheapness."' in t
    assert 'background:' in t and '>LONG</span>' in t
    assert 'Pegged &lt;regime&gt;' in t        # cap reason escaped into the ⚠ title
    m = r['matrix']
    assert 'class="cat-header" colspan="2" style="background:#123"' in m
    assert '<td class="gp">-1.68</td>' in m and '<td class="gf">-2.8</td>' in m
    assert '<td class="gn">—</td>' in m        # HKD's REER gate is N/A


def test_ratings_take_the_stock_colours():
    out = _render("process.stdout.write(JSON.stringify(_CX_RATINGS.map(function(r){return RC[_CX_RC[r]];})));")
    src = _tpl()
    rc = json.loads(re.search(r"^var RC=(\{.*?\});$", src, re.M).group(1).replace("'", '"'))
    assert json.loads(out) == [rc['BUY'], rc['LEAN BUY'], rc['HOLD'], rc['PASS']]
