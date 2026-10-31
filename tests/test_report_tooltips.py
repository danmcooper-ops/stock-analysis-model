"""Floating tooltips: placement, dismissal and the text they show.

Every tip is position:fixed and placed once from its source's rect, so a
scroll, a swipe to another tab or another ticker used to leave it stranded
over unrelated content (a Sector Heatmap header tip sitting on its own
column's rows, 2026-10-31). The placement and header-text helpers run in
Node, lifted from the template."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.scoring import gate_metadata

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / 'templates' / 'report.html'


def _tpl():
    return TEMPLATE.read_text(encoding='utf-8')


def _fn_src(name, src):
    return re.search(r'^function ' + re.escape(name) + r'\(.*?^\}', src, re.S | re.M).group(0)


def _node(js):
    if shutil.which('node') is None:
        pytest.skip('node not installed')
    return subprocess.run(['node', '-e', js], capture_output=True, text=True, check=True).stdout


def _place(rect, avoid=None, tip=(360, 130), vw=375, vh=812):
    """Place a tip of size `tip` against `rect` on a vw x vh screen; returns
    the tip's {left, top}. offsetWidth stands in for the shrink-wrap: the
    stub reads narrower unless it was moved to the origin first."""
    stub = ('var window={innerWidth:%d,innerHeight:%d};'
            'var el={style:{left:"300px",top:"500px"},'
            'get offsetWidth(){return this.style.left==="0px"?%d:%d;},offsetHeight:%d};'
            % (vw, vh, tip[0], tip[0] - 120, tip[1]))
    opts = json.dumps({'avoid': avoid}) if avoid else 'null'
    js = '\n'.join([stub, _fn_src('_placeTip', _tpl()),
                    f'_placeTip({json.dumps(rect)},el,{opts});',
                    'process.stdout.write(JSON.stringify(el.style));'])
    st = json.loads(_node(js))
    return float(st['left'][:-2]), float(st['top'][:-2])


def _rect(left, top, w=64, h=36):
    return {'left': left, 'top': top, 'right': left + w, 'bottom': top + h, 'width': w, 'height': h}


def test_header_tip_sits_above_its_header_when_there_is_room():
    left, top = _place(_rect(250, 400), avoid='below')
    assert top + 130 <= 400


def test_header_near_the_top_never_gets_its_tip_on_top_of_it():
    # A phone with the header near the top: no room above, below is the
    # column, no side fits. The clamp used to land the tip on the header.
    hdr = _rect(250, 70)
    left, top = _place(hdr, avoid='below')
    assert top >= hdr['bottom'] or top + 130 <= hdr['top']
    assert top + 130 <= 812 - 8   # pinned to the far edge, on screen


def test_tip_is_measured_at_the_origin_so_it_keeps_the_gutter():
    left, _ = _place(_rect(250, 400), avoid='below', tip=(300, 130))
    assert left + 300 <= 375 - 8


def test_every_tip_closes_on_scroll_and_on_view_or_ticker_change():
    src = _tpl()
    assert "document.addEventListener('scroll',_hideFloatTips,{capture:true" in src
    assert 'window._hidePopupTip=_hideTipP;' in src
    hide = _fn_src('_hideFloatTips', src)
    for part in ('_hideColTip()', '_hideTip()', '_hidePopupTip', "'chart-tip'", "'pool-tt'", "'pp-quad-tip'"):
        assert part in hide
    # Swipes, hotkeys and ticker steps reach these without a click.
    for fn in ('navGo', 'showView', '_swipeTab', 'openDet', 'stkGo',
               'swTSub', 'showPoolTab', '_fmSubTab', '_ppSubTab'):
        body = _fn_src(fn, src)
        assert '_hideFloatTips();' in body.split('\n')[0] + body.split('\n')[1], fn
    close = re.search(r'^function closeDetail\(\)\{.*\}$', src, re.M).group(0)
    assert '_hideFloatTips();' in close


def _col_tip(key, label='L', body=None):
    meta = gate_metadata()
    src = _tpl()
    attrs = {'data-col-tip': key, 'data-col-label': label}
    if body:
        attrs['data-col-tip-body'] = body
    js = '\n'.join([
        f'var GM={json.dumps(meta)};var TT={{}};',
        f'var th={{getAttribute:function(k){{var a={json.dumps(attrs)};return k in a?a[k]:null;}}}};',
        _fn_src('_colTipContent', src),
        'process.stdout.write(_colTipContent(th).body);'])
    return _node(js)


def test_category_total_header_is_not_called_a_gate():
    body = _col_tip('_score_moat', 'Moat Total')
    assert body.startswith('Moat category score')
    assert 'gate.' not in body.split(':')[0]
    assert 'N/A' in body and 'composite' in body


def test_gate_score_fallback_describes_the_weighting():
    body = _col_tip('_score_mos')
    assert 'mos gate' in body and 'Averaged' not in body


def test_mos_tip_agrees_with_the_gate_threshold():
    src = _tpl()
    tt = re.search(r"^mos:'(.*?)',$", src, re.M).group(1)
    assert 'Demand 20' not in tt and 'passes above 10%' in tt
    gate = next(g for g in gate_metadata()['gates'] if g['key'] == '_gate_mos')
    assert gate['threshold'] == 'MoS > 10%'


def test_accruals_threshold_is_one_sided():
    # The test is v < 0.08: large negative accruals pass, so not |Acr|.
    gate = next(g for g in gate_metadata()['gates'] if g['key'] == '_gate_accruals')
    assert gate['threshold'] == 'Accruals < 8%'


def test_sector_heatmap_tips_match_what_it_computes():
    src = _tpl()
    start = src.index('var _sumSectors={};')
    block = src[start:src.index('var _heatTableHtml=', start) + 6000]
    assert 'var fcfMVals=nonLend.map' in block and 'var altZVals=nonLend.map' in block
    assert 'Values are sector medians;' not in block
    assert "'GICS sector." not in block and 'vs. DCF' not in block
    assert 'lenders left out' in block
