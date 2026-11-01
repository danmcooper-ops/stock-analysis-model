"""The top bar's ? holds each page's explanatory text (templates/report.html)."""
import re
from pathlib import Path

TPL = Path(__file__).resolve().parent.parent / 'templates' / 'report.html'


def _tpl():
    return TPL.read_text(encoding='utf-8')


def test_help_button_sits_in_the_top_bar():
    src = _tpl()
    header = src.split('<div class="header">', 1)[1].split('<!-- Seeking Alpha-style navigation', 1)[0]
    assert 'id="hdr-help"' in header and 'id="help-pop"' in header


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
