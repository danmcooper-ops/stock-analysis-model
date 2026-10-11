# tests/test_report_hotkeys.py
"""Tests for the report's global keyboard shortcuts.

The bindings live entirely in templates/report.html, so these are source-level
assertions in the manner of test_epv_tooltips_keyed_to_row_fields — they pin the
invariants that are easy to break by editing the template and impossible to
notice without a browser, plus a render check that the overlay actually ships.
"""
import re
from pathlib import Path

from scripts.report_html import build_html

TEMPLATE = Path(__file__).resolve().parents[1] / 'templates' / 'report.html'


def _tpl():
    return TEMPLATE.read_text(encoding='utf-8')


def test_hotkeys_table_is_the_only_source_of_bindings():
    """The cheat sheet renders from HOTKEYS, so a binding can't be documented
    in one place and dispatched from another."""
    src = _tpl()
    assert re.search(r'^var HOTKEYS=\[', src, re.M)
    # The overlay builder walks HOTKEYS; nothing else may hand-roll the sheet.
    assert src.count('HOTKEYS.forEach(') == 1
    assert '<div id="hk-body"></div>' in src


def test_single_global_keydown_owns_navigation():
    """Escape precedence (overlay -> menus -> popup) only holds while one
    handler decides it. A second navigation listener would make the order
    depend on registration order instead. The AI panel keeps its own handler
    for its own Escape, which is why the count is two and not one."""
    src = _tpl()
    assert src.count("document.addEventListener('keydown'") == 2
    assert src.count('_hkDispatch(e);') == 1
    # The popup must consume the keystroke rather than fall through to the
    # page-level hotkeys while it is open.
    modal_block = src.split("if(document.getElementById('det-modal').classList.contains('open')){")[1]
    assert modal_block.split('_hkBlocked')[0].count('return;') >= 1


def test_hotkeys_never_fire_while_typing():
    """One target test has to cover every text field on the page — the ticker
    search, the column-picker search and the AI textarea."""
    guard = _tpl().split('function _hkBlocked(e){')[1].split('\n}')[0]
    assert 'e.ctrlKey||e.metaKey||e.altKey' in guard
    assert 'isComposing' in guard
    assert 'isContentEditable' in guard
    for tag in ('input', 'textarea', 'select'):
        assert f"'{tag}'" in guard
    # Shift must NOT be rejected, or '{' and '}' become unreachable.
    assert 'shiftKey' not in guard


def test_digit_keys_index_the_live_nav_group_list():
    """Views are addressed by position in _navGroups(), which drops Macro
    Outlook on snapshots without macro data. A hardcoded name map would leave
    '1' dead there, and the menu badge would lie."""
    src = _tpl()
    assert '_navGroups().length' in src.split('function _hkDispatch(e){')[1]
    # The badge is printed from the same list the dispatcher indexes.
    assert "_navGroups().forEach(function(g,gi){" in src
    assert '<kbd class="navmenu-key">\'+(gi+1)+\'</kbd>' in src


def test_filter_hotkeys_inert_where_the_filter_bar_is_hidden():
    """Sector Analysis and Macro Outlook get .nav-only, which hides the
    Filters button; opening its panel there would anchor to an invisible
    element and put the caret out of sight. '/' is NOT gated: the search
    sits in the top bar on every view (views without a table list quick
    results instead of filtering)."""
    src = _tpl()
    assert "classList.contains('nav-only')" in src.split('function _hkFiltersUsable(){')[1]
    dispatch = src.split('function _hkDispatch(e){')[1].split('\n}\n')[0]
    assert dispatch.count('_hkFiltersUsable()') == 1
    slash = dispatch.split("k==='/'")[1].split("k==='f'")[0]
    assert '_hkFiltersUsable' not in slash
    assert "getElementById('f-search')" in slash


def test_overlay_ships_in_the_rendered_report(tmp_path):
    out = tmp_path / 'report.html'
    build_html([{'ticker': 'AAA', 'price': 1.0, 'rating': 'HOLD'}], str(out),
               prices_dir=None)
    html = out.read_text(encoding='utf-8')
    assert 'id="hk-overlay"' in html
    assert 'var HOTKEYS=[' in html
    assert 'Keyboard shortcuts' in html


def test_overlay_ships_in_an_empty_report(tmp_path):
    """No rows still means a usable page — and the shortcuts still apply."""
    out = tmp_path / 'report_empty.html'
    build_html([], str(out), prices_dir=None)
    html = out.read_text(encoding='utf-8')
    assert 'id="hk-overlay"' in html
    assert 'var HOTKEYS=[' in html


def _det_dispatch():
    return _tpl().split('function _hkDetDispatch(e){')[1].split('\n}\n')[0]


def test_popup_hotkeys_stand_aside_for_typing():
    """The popup branch hands every key but Escape to _hkDetDispatch, and only
    past the same _hkBlocked guard the page-level keys use — the membership
    menu's new-portfolio field lives inside the popup."""
    modal_block = _tpl().split(
        "if(document.getElementById('det-modal').classList.contains('open')){")[1].split('\n  }\n')[0]
    assert 'else if(!_hkBlocked(e))_hkDetDispatch(e);' in modal_block
    assert '_hkDispatch(e)' not in modal_block


def test_every_popup_hotkey_is_on_the_cheat_sheet():
    """Each key _hkDetDispatch matches must appear in a 'Ticker popup' row of
    HOTKEYS, and vice versa, so the ? sheet stays a complete map."""
    src = _tpl()
    table = src.split('var HOTKEYS=[')[1].split('\n];')[0]
    popup_rows = [ln for ln in table.splitlines() if "g:'Ticker popup'" in ln]
    listed = set(re.findall(r"'([^']+)'", ''.join(r.split(",l:")[0] for r in popup_rows)))
    dispatch = _det_dispatch()
    matched = set(re.findall(r"k===('([^']+)')", dispatch))
    matched = {m[1] for m in matched}
    names = {'ArrowLeft': '←', 'ArrowRight': '→', 'ArrowUp': '↑', 'ArrowDown': '↓'}
    shown = {names.get(k, k) for k in matched}
    listed.discard('Ticker popup')
    # '?' is listed once, under Search & panels, and works in the popup too.
    shown.discard('?')
    assert shown <= listed, shown - listed
    # Everything listed (digits aside, which come from a function) is dispatched.
    assert listed <= shown, listed - shown
    assert "k>='1'&&k<='9'" in dispatch
    assert '_detMenuItems().length' in dispatch


def test_popup_section_keys_index_the_rendered_tab_list():
    """Digits and brackets address _detMenuItems(), the list _renderDetTabBar
    draws, so '3' always lands on the third tab shown."""
    src = _tpl()
    assert '_detMenuItems().forEach(function(it){' in src.split('function _renderDetTabBar(){')[1]
    assert 'var its=_detMenuItems()' in src.split('function _hkDetTab(i){')[1].split('\n}')[0]


def test_cheat_sheet_from_the_popup_keeps_its_scroll_lock():
    """Closing the overlay opened over the popup must not release the body
    lock the popup still holds."""
    close = _tpl().split('function hkOverlayClose(){')[1].split('\n')[0]
    assert "det-modal').classList.contains('open')" in close
    assert '_unlockBody()' in close


def test_enter_marks_reviewed_then_next_ticker():
    """Enter in the ticker popup marks the stock reviewed (never unmarks it)
    and steps to the next ticker; the cheat sheet lists it."""
    src = _tpl()
    assert "{g:'Ticker popup',k:['Enter'],l:'Mark reviewed, then next ticker'}" in src
    disp = src.split('function _hkDetDispatch(e){')[1].split('\n}')[0]
    enter = disp.split("k==='Enter'")[1].split('\n')[0]
    assert 'if(tk&&!isReviewed(tk)){toggleReviewed(tk);_syncRevDet();}navDet(1);' in enter
