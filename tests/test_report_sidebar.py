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


def test_ticker_page_actions_share_the_button_grid():
    """Folder / check / flag sit in squares the size of the ‹ › × buttons,
    spaced by the arrows' gap, each an 18px SVG — the flag no longer a font
    glyph whose size varies."""
    src = _tpl()
    assert '#det-modal .d-mini-nav-arrows{gap:6px;}' in src
    assert '#det-modal .d-mini-nav-actions{gap:6px;margin-left:-5px;}' in src
    assert '#det-modal .d-mini-nav-actions>*{position:relative;box-sizing:border-box;width:28px;height:28px;' in src
    assert '#det-modal .d-mini-nav-actions svg{width:18px !important;height:18px !important;' in src
    sync = _fn(src, '_syncFlagDet')
    assert 'el.innerHTML=_flagSvg(f)' in sync and '\\u2691' not in sync


def test_ticker_page_top_row_holds_actions_and_pager():
    """The band's first line is the portfolio / reviewed / flag icons on the
    left and ‹ n / N › with × on the right, all in flow — so the ticker, name
    and meta lines reserve no right padding for floating controls — and it
    collapses away with the name and meta."""
    src = _tpl()
    band = src[src.index('<div class="d-header-band">'):src.index('<div class="d-tabs sa-chrome"')]
    top = band[band.index('<div class="d-toprow">'):band.index('<div class="dh">')]
    order = [top.index(k) for k in ('id="d-pf"', 'id="d-rev"', 'id="d-flag"', 'det-nav-prev',
                                    'class="det-nav-pos"', 'det-nav-next', 'class="detail-close"')]
    assert order == sorted(order)
    assert band.index('<div class="d-toprow">') < band.index('id="d-tk"')
    assert band.count('detail-close') == 1
    for gone in ('.d-mini-nav{position:absolute', '.d-mini-nav-actions{position:absolute',
                 'padding-right:84px', 'padding-right:130px', '.dh{padding-right:44px'):
        assert gone not in src, gone
    assert '.d-toprow .detail-close{position:static;' in src
    # The top row is one of the band lines that slide out as it collapses.
    assert '#det-modal .d-header-band>:not(.d-tabs):not(.d-compact){opacity:' in src
    # Long names and meta wrap rather than overflow.
    assert 'overflow-wrap:anywhere;' in src[src.index('#det-modal .dcname{'):][:200]


def test_phone_ticker_page_closes_from_the_top_bar():
    """On phones the ticker page's × is the top bar's own button (the bar
    stacks above the modal, so the band's × could not be lifted into it),
    and the collapsed band keeps no blank row: it shows the compact row,
    whose lines cannot wrap."""
    src = _tpl()
    assert '<button type="button" class="hdr-close sa-chrome" id="hdr-close" onclick="closeDetail()"' in src
    assert '.hdr-close{display:none;' in src
    assert 'html.det-open .hdr-close{display:flex;}' in src
    assert 'html.det-open #det-modal .detail-close{display:none;}' in src
    # Collapsed, price · change · rating · score sit on the compact row's
    # second line, one line that cannot wrap.
    assert ('#det-modal .d-compact .dc-l1,#det-modal .d-compact .dc-l2'
            '{display:flex;align-items:center;gap:8px;white-space:nowrap;overflow:hidden;}') in src


def test_review_feature_is_back():
    """The Reviewed marks are restored (without the Summary side cards that
    left with them): the table and matrix column, the Reviewed/Unreviewed
    filters and their hash key, the ticker page's check icon, the Enter "mark
    reviewed" hotkey. The frozen columns are # · flag · reviewed · ticker
    (· rating · Δ) again."""
    src = _tpl()
    for back in ('function isReviewed(', 'function toggleReviewed(', 'function _revHtml(',
                 'var _revFilter=', 'rev-chk', 'id="f-reviewed"', 'id="f-unreviewed"',
                 'id="d-rev"', '_reviewed:', 'var _CHK_SVG=', 'p.rv=st.revFilter',
                 "var _REV_KEY='stock_reviewed_v1';"):
        assert back in src, back
    # The 28 Sep removal wiped the stored marks on every load; that must not
    # survive the restore or each new mark would be lost on reload.
    assert "localStorage.removeItem('stock_reviewed_v1')" not in src
    # Ticker is the 4th frozen column again, 32px further right.
    assert '#dtbl th:nth-child(4),#dtbl td.tk2{left:calc(var(--sb-cur) + 100px);}' in src
    assert '#dtbl th:nth-child(5),#dtbl tbody td:nth-child(5){left:calc(var(--sb-cur) + 168px);}' in src
    assert '#dtbl th:nth-child(6),#dtbl tbody td:nth-child(6){left:calc(var(--sb-cur) + 240px);}' in src
    assert '#mtx thead tr:first-child th:nth-child(4),#mtx td.tk{left:calc(var(--sb-cur) + 100px);}' in src
    assert '#mtx thead tr:first-child th:nth-child(6),#mtx tbody td:nth-child(6){left:calc(var(--sb-cur) + 232px);}' in src


def test_chart_controls_are_underline_tabs_behind_the_menu():
    """The ticker page's chart and the Financial Data Charts view keep their
    Metric / Range / Compare controls behind the chart's ☰ button, and inside
    that menu draw them as underline tabs like the section tabs and #subnav
    sub-tabs (.cm-uline / .cm-tabs); an index that is on carries its bar in
    its own line colour."""
    src = _tpl()
    p = 'html body .cm-panel.cm-uline'
    assert '<div class="cm-panel cm-uline" id="dtc-cm-panel">' in src
    assert """'<div class="cm-panel cm-uline" id="ph-cm-panel">'""" in src
    for tabs in ('dtc-metric-btns', 'dtc-ranges', 'dtc-bench-btns',
                 'ph-metric-btns', 'ph-ranges', 'ph-index-btns'):
        assert f'<div class="cm-tabs" id="{tabs}"></div>' in src, tabs
    # Behind the menu: nothing forces the panel open or hides the ☰ button.
    assert 'cm-inline' not in src
    assert 'display:flex !important;flex-wrap:wrap;gap:10px 28px;position:static' not in src
    assert "cmToggle('dtc',event)" in src and "cmToggle(\\'ph\\',event)" in src
    assert (f'{p} .ph-metric-btn.active,{p} .ph-range-btn.active,{p} .ph-index-btn.active'
            '{--sac:var(--chrome-heading);font-weight:700;box-shadow:inset 0 -3px 0 var(--chrome-accent);}') in src
    assert f'{p} .ph-index-btn.active{{box-shadow:inset 0 -3px 0 var(--ix,var(--chrome-accent));}}' in src
    assert src.count("""style="--ix:'+col+'">""") == 2


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


def test_ticker_key_metrics_wrap_to_two_columns_on_phones():
    """The five key-metric cells (Mkt cap, Price, Fair value, MoS, Next
    earnings) used to share one row on a phone, truncating every label and
    running the fair-value band into the MoS badge. The #det-modal rules
    outrank the old <=480px shrink rules, so the phone layout is scoped the
    same way: a two-column grid whose odd last cell spans the row, with
    labels and sub-lines wrapping instead of clipping."""
    src = _tpl()
    m = re.search(r'@media \(max-width:600px\)\{\n  #det-modal \.d-kmetrics\{([^}]*)\}', src)
    assert m, 'phone grid for #det-modal .d-kmetrics'
    assert 'display:grid' in m.group(1)
    assert 'grid-template-columns:repeat(2,minmax(0,1fr))' in m.group(1)
    assert '#det-modal .d-kmetrics .km-cell:last-child:nth-child(odd){grid-column:1/-1;}' in src
    assert '#det-modal .d-kmetrics .km-k{font-size:.66em;white-space:normal;' in src
    assert '#det-modal .d-kmetrics .km-sub{font-size:.64em;white-space:normal;}' in src


def _node_eval(src_fns, setup, expr, tmp_path):
    import json
    import shutil
    import subprocess
    if shutil.which('node') is None:
        import pytest
        pytest.skip('node not installed')
    js = setup + '\n' + '\n'.join(src_fns) + '\nprocess.stdout.write(JSON.stringify(' + expr + '));'
    p = tmp_path / 'sb.js'
    p.write_text(js, encoding='utf-8')
    return json.loads(subprocess.run(['node', str(p)], capture_output=True, text=True,
                                     check=True).stdout)


def _fn_src(name):
    import re
    src = open(TEMPLATE, encoding='utf-8').read()
    return re.search(r'^function ' + name + r'\(.*?^\}', src, re.S | re.M).group(0)


def test_sidebar_mode_follows_the_css_media_queries_not_zoom(tmp_path):
    """Zoomed out on an iPad mini, window.innerWidth read ~1,500px and the
    menu button decided it was on a desktop; the CSS (media queries) still
    showed the phone bar. The mode now asks the same media queries."""
    setup = ('var window={innerWidth:1500,matchMedia:function(q){return {matches:'
             '/max-width:768px/.test(q)||/max-width:1023px/.test(q)};}};'
             'var document={documentElement:{clientWidth:744}};')
    assert _node_eval([_fn_src('_sbMode')], setup, '_sbMode()', tmp_path) == 'phone'


def test_wide_views_hold_a_full_size_zoom_floor_on_ios(tmp_path):
    """A brief maximum-scale cap did not hold on an iPad: Safari re-fitted
    the wide table to the 0.5 floor once the viewport was restored. The
    floor itself is raised to 1.0 while a wide view is open, then restored."""
    fn = _fn_src('_syncWideTable')
    assert 'if(wide!==_wasWide)_setIosZoomFloor(wide);' in fn
    setup = ('var navigator={userAgent:"iPad",maxTouchPoints:5};'
             'var meta={a:{content:"width=device-width, initial-scale=1.0, minimum-scale=0.5"},'
             'getAttribute:function(k){return this.a[k];},setAttribute:function(k,v){this.a[k]=v;}};'
             'var document={querySelector:function(){return meta;}};')
    out = _node_eval([_fn_src('_isIos'), _fn_src('_setIosZoomFloor')], setup,
                     '(function(){_setIosZoomFloor(true);var a=meta.a.content;'
                     '_setIosZoomFloor(false);return [a,meta.a.content];})()', tmp_path)
    assert out[0].endswith('minimum-scale=1.0') and out[1].endswith('minimum-scale=0.5')


def test_no_all_portfolios_row():
    """The sidebar's portfolio heading opens the overview, so an "All
    Portfolios" row under it is redundant."""
    tpl = _tpl()
    assert 'All Portfolios</span>' not in tpl
    assert 'class="sb-head" data-v="pf" data-k="overview"' in tpl


def test_ticker_band_collapse_follows_the_scroll():
    """The band collapses with the finger, not on a timer: its height is set
    from the scroll position each frame, a spacer at the top of .d-body
    takes up what the band gives back (so content moves 1:1), and nothing
    animates on its own clock. Scroll anchoring would cancel the spacer."""
    src = _tpl()
    apply = _fn(src, '_detHdrApply')
    assert "e.band.style.height=c>0?(_hdr.hExp-c)+'px':'';" in apply
    assert "e.hc.style.height=(_hdr.src==='body'?c:0)+'px';" in apply
    assert "e.dm.style.setProperty('--hp',p.toFixed(4));" in apply
    on = _fn(src, '_detHdrOnScroll')
    assert 'requestAnimationFrame(_detHdrApply)' in on
    assert '<div class="d-hc" aria-hidden="true"></div>' in src
    assert '#det-modal .d-body{overflow-anchor:none;}' in src
    assert '#det-modal .d-header-band>*{flex-shrink:0;}' in src
    assert 'justify-content:flex-end;' in src[src.index('#det-modal .d-header-band{display:flex;'):][:120]
    # The old threshold class and its .28s transitions are gone.
    for gone in ('_hdrHoldUntil', 'st>40&&st>last+2', 'transition:max-height .28s',
                 '#det-modal.hdr-min .dcname'):
        assert gone not in src, gone
    # The compact row mirrors the band and never re-triggers its own sync.
    assert 'if(!(e.cp&&e.cp.contains(t))){later();return;}' in src
    assert "closeDetail(){closeDetMenus();var _dm=document.getElementById('det-modal');_dm.classList.remove('open');_detHdrReset();" in src
