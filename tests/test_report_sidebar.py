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
    for needle in ('<nav id="sidebar"', '<nav id="tabbar"', '<div id="subnav"',
                   'id="sn-chips"', 'id="sb-views"', 'id="sb-pf"'):
        assert needle in src, needle
    for gone in ('id="nav-panel"', 'id="nav-cols"', 'id="nav-btn"', 'id="theme-toggle"'):
        assert gone not in src, gone


def test_one_renderer_feeds_every_navigation_surface():
    """Sidebar rows, chips and tab bar all come from _navGroups() in
    renderNavMenu, so they can never disagree with each other or with the
    digit hotkeys."""
    body = _fn(_tpl(), 'renderNavMenu')
    assert '_navGroups()' in body
    for target in ("'sb-views'", "'sb-pf'", "'sn-chips'", "'tabbar'"):
        assert target in body, target


def test_one_delegated_listener_dispatches_navigation():
    src = _tpl()
    assert src.count("closest('#sidebar [data-v],#tabbar [data-v],#subnav [data-v]')") == 1
    listener = src.split("closest('#sidebar [data-v],#tabbar [data-v],#subnav [data-v]')")[1][:700]
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
