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


def test_statements_share_one_finances_tab_after_profile():
    """Balance Sheet, Income Statement and Cash Flow Statement live under one
    Finances tab, third in the ticker page's tab bar (after Summary and the
    one-page Profile), with a switch inside its pane; a statement key passed
    to swDetTab still lands there."""
    src = _tpl()
    assert "var DT_FIN={k:'fin',l:'Finances'};" in src
    assert "return [{k:'overview',l:'Overview'},DT_PROFILE,DT_FIN].concat(_detDataItems());" in src
    assert '.concat(DT_STMTS)' not in src
    sw = _fn(src, 'swDetTab')
    assert 'if(DT_STMTS[si].k===k){_detStmt=k;k=DT_FIN.k;}' in sw
    assert '_renderDetStmtTab(_detStmt)' in sw
    assert 'data-dstmt' in _fn(src, '_stmtSwitchHtml')
    assert "closest('#d-pane-stmt [data-dstmt]')" in src


def test_ticker_page_actions_share_the_button_grid():
    """Folder / check / flag sit in squares the size of the ‹ › × buttons,
    spaced by the arrows' gap and right-aligned with ×, each an 18px SVG —
    the flag no longer a font glyph whose size varies."""
    src = _tpl()
    assert '#det-modal .d-mini-nav-arrows{gap:6px;}' in src
    assert 'top:calc(46px + env(safe-area-inset-top));bottom:auto;right:24px;gap:6px;' in src
    assert '#det-modal .d-mini-nav-actions>*{position:relative;box-sizing:border-box;width:28px;height:28px;' in src
    assert '#det-modal .d-mini-nav-actions svg{width:18px !important;height:18px !important;' in src
    sync = _fn(src, '_syncFlagDet')
    assert 'el.innerHTML=_flagSvg(f)' in sync and '\\u2691' not in sync


def test_phone_ticker_page_closes_from_the_top_bar():
    """On phones the ticker page's × is the top bar's own button (the bar
    stacks above the modal, so the band's × could not be lifted into it),
    and the collapsed band keeps no blank row: the zero-width rating-row
    tails are display:none so they cannot wrap onto a line of their own."""
    src = _tpl()
    assert '<button type="button" class="hdr-close sa-chrome" id="hdr-close" onclick="closeDetail()"' in src
    assert '.hdr-close{display:none;' in src
    assert 'html.det-open .hdr-close{display:flex;}' in src
    assert 'html.det-open #det-modal .detail-close{display:none;}' in src
    assert ('#det-modal.hdr-min .d-gp,#det-modal.hdr-min .d-rsince,#det-modal.hdr-min .det-nav-pos'
            '{max-width:0;opacity:0;margin:0;display:none;}') in src
    assert 'html.det-open #det-modal.hdr-min .d-rating-row{margin-top:6px;gap:8px;flex-wrap:nowrap;' in src


def test_review_feature_is_gone():
    """The Reviewed marks were unused and are removed everywhere: the table
    and matrix column, the Reviewed/Unreviewed filters and their hash key,
    the ticker page's check icon, the Enter "mark reviewed" hotkey. The
    frozen columns close up to # · flag · ticker (· rating · Δ)."""
    src = _tpl()
    for gone in ('isReviewed', 'toggleReviewed', '_revHtml', '_revFilter', 'rev-chk',
                 'f-reviewed', 'f-unreviewed', 'id="d-rev"', "(\\'_reviewed\\')", '_reviewed:', '_CHK_SVG',
                 'Mark reviewed', 'p.rv'):
        assert gone not in src, gone
    # The old stored marks are cleared rather than left behind.
    assert "localStorage.removeItem('stock_reviewed_v1')" in src
    # Ticker is now the 3rd frozen column, 32px further left.
    assert '#dtbl th:nth-child(3),#dtbl td.tk2{left:calc(var(--sb-cur) + 68px);}' in src
    assert '#dtbl th:nth-child(4),#dtbl tbody td:nth-child(4){left:calc(var(--sb-cur) + 136px);}' in src
    assert '#mtx thead tr:first-child th:nth-child(5),#mtx tbody td:nth-child(5){left:calc(var(--sb-cur) + 200px);}' in src


def test_chart_controls_are_underline_tabs():
    """The ticker page's chart and the Financial Data Charts view share one
    inline control style (.cm-inline): Metric / Range / Compare / index rows
    as underline tabs like the section tabs and #subnav sub-tabs, out in the
    open rather than behind the ☰ menu; an index that is on carries its bar
    in its own line colour."""
    src = _tpl()
    p = 'html body .cm-panel.cm-inline'
    assert '<div class="cm-panel cm-inline" id="dtc-cm-panel">' in src
    assert """'<div class="cm-panel cm-inline" id="ph-cm-panel">'""" in src
    for tabs in ('dtc-metric-btns', 'dtc-ranges', 'dtc-bench-btns',
                 'ph-metric-btns', 'ph-ranges', 'ph-index-btns'):
        assert f'<div class="cm-tabs" id="{tabs}"></div>' in src, tabs
    assert '.cm-wrap:has(.cm-inline)>.cm-btn,.cm-wrap:has(.cm-inline)>.cm-sum{display:none;}' in src
    assert (f'{p} .ph-metric-btn.active,{p} .ph-range-btn.active,{p} .ph-index-btn.active'
            '{--sac:var(--chrome-heading);font-weight:700;box-shadow:inset 0 -3px 0 var(--chrome-accent);}') in src
    assert f'{p} .ph-index-btn.active{{box-shadow:inset 0 -3px 0 var(--ix,var(--chrome-accent));}}' in src
    assert src.count("""style="--ix:'+col+'">""") == 2
    assert '#det-modal #dtc-cm-panel' not in src


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


def test_summary_has_no_value_range_or_peers_cards():
    """The Summary's side column (Value Range and Peers cards) was removed;
    peers live on the Peers tab only, still fed by _detPeerList."""
    src = _tpl()
    for gone in ('d-range-card', 'd-peers-card', 'd-sum-side', 'd-sum-grid', 'd-side-card',
                 'function _detRangeHtml(', 'function _detPeersCardHtml(', 'function _detFillSide('):
        assert gone not in src, gone
    open_det = src[src.index('function openDet(tk){'):src.index('function closeDetail(){')]
    assert 'var _allPeers=_detPeerList(d);' in open_det


def test_ticker_page_search_and_phone_back_button():
    src = _tpl()
    # Search on the ticker page switches ticker instead of filtering the
    # table hidden underneath it.
    assert "classList.contains('det-open')" in _fn(src, '_qrActive')
    tap = _fn(src, 'hdrMenuTap')
    assert tap.index('closeDetail()') < tap.index('sbOpen()')
    assert '_detOpenState(false)' in src[src.index('function closeDetail(){'):][:300]
