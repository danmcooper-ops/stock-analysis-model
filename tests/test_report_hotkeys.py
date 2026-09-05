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
    element and put the caret out of sight."""
    src = _tpl()
    assert "classList.contains('nav-only')" in src.split('function _hkFiltersUsable(){')[1]
    dispatch = src.split('function _hkDispatch(e){')[1]
    assert dispatch.count('_hkFiltersUsable()') == 2


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
