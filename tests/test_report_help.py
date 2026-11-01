"""The top bar's ? holds each page's explanatory text (templates/report.html)."""
import re
from pathlib import Path

TPL = Path(__file__).resolve().parent.parent / 'templates' / 'report.html'


def _tpl():
    return TPL.read_text(encoding='utf-8')


def test_help_button_sits_in_the_page_corner_not_the_top_bar():
    """The ? is the page's upper-right corner (the view title row, or the
    ticker band's top row), as each card's ? is its card's; not the top bar."""
    src = _tpl()
    header = src.split('<div class="header">', 1)[1].split('<!-- Seeking Alpha-style navigation', 1)[0]
    assert 'page-help-btn' not in header
    sn_head = src.split('<div class="sn-head">', 1)[1].split('</div>', 1)[0]
    assert 'id="page-help-btn"' in sn_head
    place = src.split('function _helpPlace(det){', 1)[1].split('\n}', 1)[0]
    assert "'#det-modal .d-mini-nav'" in place and "'#subnav .sn-head'" in place


def test_every_page_key_has_text():
    src = _tpl()
    body = src.split('function _helpKey(){', 1)[1].split('\n}', 1)[0]
    keys = set(re.findall(r"return '(\w+)'", body)) | set(re.findall(r"\?'(\w+)':'(\w+)'", body)[0])
    keys |= {'macro', 'charts', 'etf', 'bond', 'ccy', 'dashboard'}  # _navView / curMSub values
    table = src.split('var _HELP_TXT={', 1)[1].split('\n};', 1)[0]
    for k in keys:
        assert re.search(r'^\s*' + k + r':\[', table, re.M), k


def test_page_text_moves_behind_the_icon():
    src = _tpl()
    assert '.pg-help{display:none !important;}' in src
    assert 'class="pfv-desc pg-help"' in src and 'class="pfv-meta pg-help"' in src
