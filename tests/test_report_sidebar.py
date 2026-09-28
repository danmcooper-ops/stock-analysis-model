# tests/test_report_sidebar.py
"""The Seeking Alpha-style navigation: sidebar, view chips and phone tab bar.

Like test_report_hotkeys, these are source-level assertions against
templates/report.html (the behaviour lives in inline JS/CSS), plus a render
check and one Node-evaluated helper. They pin the invariants that break
silently: one renderer for every navigation surface, one delegated click
path, the frozen table columns clearing the sidebar, and the layout state
read defensively from localStorage.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.report_html import build_html

TEMPLATE = Path(__file__).resolve().parents[1] / 'templates' / 'report.html'


def _tpl():
    return TEMPLATE.read_text(encoding='utf-8')


def _fn(src, name):
    """Source of a top-level `function name(...){...}` (ends at the first
    line that is exactly `}`)."""
    start = src.index('function ' + name + '(')
    return src[start:src.index('\n}\n', start) + 2]


def test_navigation_surfaces_exist_and_the_views_dropdown_is_gone():
    src = _tpl()
    for needle in ('<nav id="sidebar"', '<div id="subnav"',
                   'id="sn-chips"', 'id="sb-views"', 'id="sb-pf"'):
        assert needle in src, needle
    for gone in ('id="nav-panel"', 'id="nav-cols"', 'id="nav-btn"', 'id="theme-toggle"', 'id="tabbar"'):
        assert gone not in src, gone


def test_one_renderer_feeds_every_navigation_surface():
    """Sidebar rows and chips all come from _navGroups() in
    renderNavMenu, so they can never disagree with each other or with the
    digit hotkeys."""
    body = _fn(_tpl(), 'renderNavMenu')
    assert '_navGroups()' in body
    for target in ("'sb-views'", "'sb-pf'", "'sn-chips'"):
        assert target in body, target


def test_one_delegated_listener_dispatches_navigation():
    src = _tpl()
    assert src.count("closest('#sidebar [data-v],#subnav [data-v]')") == 1
    listener = src.split("closest('#sidebar [data-v],#subnav [data-v]')")[1][:700]
    assert 'navGo(' in listener
    # Navigating from over an open ticker page closes it.
    assert 'closeDetail()' in listener


def test_views_hotkey_drives_the_sidebar():
    src = _tpl()
    assert 'sbToggle()' in _fn(src, 'toggleNavMenu')
    assert 'sbClose()' in _fn(src, 'closeNavMenu')


def test_sidebar_state_is_read_defensively():
    """localStorage can throw (private mode, blocked storage): both the
    pre-paint read and the runtime read sit inside try/catch."""
    src = _tpl()
    prepaint = src[src.index("localStorage.getItem('stock_sidebar_v1')") - 200:]
    assert 'try{' in prepaint[:260]
    assert 'try{' in _fn(src, '_sbStoredRail')


def test_frozen_table_columns_clear_the_sidebar():
    """The page pans horizontally under a fixed sidebar, so every frozen
    column of the two page-scrolling tables offsets by --sb-cur."""
    src = _tpl()
    for tbl in ('#dtbl', '#mtx'):
        offs = re.findall(re.escape(tbl) + r'[^{]*\{left:(?:var\(--sb-cur\)|calc\(var\(--sb-cur\) \+ \d+px\));\}', src)
        assert len(offs) >= 6, (tbl, offs)
    # The page-width math adds the sidebar padding back in.
    assert 'paddingLeft' in _fn(src, '_syncWideTable')


@pytest.mark.skipif(shutil.which('node') is None, reason='node not installed')
def test_sub_tabs_are_underline_tabs():
    """Sub-tabs are text on a hairline with an accent bar under the current
    one (not filled pills), matching the ticker page's #d-tabs."""
    src = _tpl()

    def rule(sel):
        return re.search(re.escape(sel) + r'\{([^}]*)\}', src).group(1)
    assert 'border-bottom:1px solid var(--tab-line)' in rule('.sn-chips')
    base = rule('.sn-chip')
    assert 'background:none' in base and 'border:0' in base and 'border-radius:0' in base
    assert 'box-shadow:inset 0 -3px 0 var(--chrome-accent)' in rule('.sn-chip.cur')
    assert 'chip-on-bg' not in rule('.sn-chip.cur')


def test_statements_share_one_finances_tab_after_summary():
    """Balance Sheet, Income Statement and Cash Flow Statement live under one
    Finances tab, second in the ticker page's tab bar, with a switch inside
    its pane; a statement key passed to swDetTab still lands there."""
    src = _tpl()
    assert "var DT_FIN={k:'fin',l:'Finances'};" in src
    assert "return [{k:'overview',l:'Overview'},DT_FIN].concat(_detDataItems());" in src
    assert '.concat(DT_STMTS)' not in src
    sw = _fn(src, 'swDetTab')
    assert 'if(DT_STMTS[si].k===k){_detStmt=k;k=DT_FIN.k;}' in sw
    assert '_renderDetStmtTab(_detStmt)' in sw
    assert 'data-dstmt' in _fn(src, '_stmtSwitchHtml')
    assert "closest('#d-pane-stmt [data-dstmt]')" in src


def test_portfolio_day_change_from_nav_ledger(tmp_path):
    src = _tpl()
    js = ('var NAV={};function _pfNavRaw(id){return NAV[id]||null;}\n'
          + _fn(src, '_pfDayChg')
          + '\nNAV.a=[["2026-09-24",1.00,0],["2026-09-25",1.02,0]];'
          + 'NAV.one=[["2026-09-25",1.0,0]];NAV.zero=[["d",0,0],["e",1,0]];\n'
          + 'var out=[_pfDayChg("a"),_pfDayChg("one"),_pfDayChg("missing"),_pfDayChg("zero")];\n'
          + 'console.log(JSON.stringify(out));\n')
    script = tmp_path / 'daychg.js'
    script.write_text(js, encoding='utf-8')
    r = subprocess.run(['node', str(script)], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    got = json.loads(r.stdout)
    assert got[0] == pytest.approx(0.02)
    assert got[1:] == [None, None, None]


def test_rendered_report_ships_the_chrome(tmp_path):
    out = tmp_path / 'report.html'
    build_html([{'ticker': 'AAA', 'price': 1.0, 'rating': 'HOLD'}], str(out), prices_dir=None)
    html = out.read_text(encoding='utf-8')
    assert '<nav id="sidebar"' in html
    assert 'id="sn-chips"' in html
    assert 'id="hdr-qr"' in html
    # The search moved to the top bar and kept its id (applyFilters reads it).
    assert html.count('id="f-search"') == 1
    assert html.index('id="f-search"') < html.index('id="filt-panel"')


def test_peers_pane_and_card_share_one_peer_list():
    src = _tpl()
    assert '_detPeerList(d)' in _fn(src, '_detPeersCardHtml')
    open_det = src[src.index('function openDet(tk){'):src.index('function closeDetail(){')]
    assert 'var _allPeers=_detPeerList(d);' in open_det
    assert '_detFillSide(d);' in open_det


def test_ticker_page_search_and_phone_back_button():
    src = _tpl()
    # Search on the ticker page switches ticker instead of filtering the
    # table hidden underneath it.
    assert "classList.contains('det-open')" in _fn(src, '_qrActive')
    tap = _fn(src, 'hdrMenuTap')
    assert tap.index('closeDetail()') < tap.index('sbOpen()')
    assert '_detOpenState(false)' in src[src.index('function closeDetail(){'):][:300]


def _range_js(tmp_path, rows):
    src = _tpl()
    js = ('function esc(s){return String(s);}\n'
          + re.search(r'^function _num\(v\)\{.*?\}$', src, re.M).group(0) + '\n'
          + re.search(r'^function fd\(v\)\{.*?\}$', src, re.M).group(0) + '\n'
          + _fn(src, 'fd2') + _fn(src, '_rngPct') + _fn(src, '_detRangeHtml')
          + 'var rows=' + json.dumps(rows) + ';\n'
          + 'console.log(JSON.stringify(rows.map(_detRangeHtml)));\n')
    script = tmp_path / 'range.js'
    script.write_text(js, encoding='utf-8')
    r = subprocess.run(['node', str(script)], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


@pytest.mark.skipif(shutil.which('node') is None, reason='node not installed')
def test_value_range_card(tmp_path):
    mc, sens, none = _range_js(tmp_path, [
        {'price': 100, 'mc_p10_fv': 80, 'mc_p90_fv': 160, '_fv_effective': 120,
         'low_52w': 70, 'high_52w': 110},
        {'price': 100, 'dcf_sens_range': [90, 140], 'dcf_fv': 115},
        {'price': 100},
    ])
    assert 'Monte Carlo P10' in mc and '52-week range' in mc

    def left(html, cls):
        return float(re.search(r'class="' + cls + r'" style="left:([\d.]+)%', html).group(1))
    band = re.search(r'class="rg-band" style="left:([\d.]+)%;width:([\d.]+)%', mc)
    b0, bw = float(band.group(1)), float(band.group(2))
    assert 0 <= b0 and b0 + bw <= 100.0001
    assert b0 <= left(mc, 'rg-tick') <= b0 + bw   # base inside bear..bull
    assert 'DCF sensitivity' in sens and '52-week' not in sens
    assert none == ''
