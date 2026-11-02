"""Every titled section takes the sector page's look (templates/report.html,
SECTION TEMPLATE): a light panel with a navy stripe down its left edge and the
navy title box inside it."""
import re
from pathlib import Path

TPL = Path(__file__).resolve().parent.parent / 'templates' / 'report.html'


def _tpl():
    return TPL.read_text(encoding='utf-8')


def test_one_rule_styles_every_section_container():
    src = _tpl()
    block = src.split('/* === SECTION TEMPLATE', 1)[1]
    assert ':root{--sx-bg:' in block and '[data-theme="dark"]{--sx-bg:' in block
    rule = block[block.index('.sx,'):]
    sel = rule.split('{', 1)[0]
    for s in ('.sx', '.xsect-dashboard', '.mac-regime', '.mac-card.mac-narrative', '#det-modal .dgroup'):
        assert s in sel, s
    body = rule.split('{', 1)[1].split('}', 1)[0]
    assert 'border-left:5px solid var(--hdr-bg)' in body and 'background:var(--sx-bg)' in body


def test_no_title_sits_in_an_untemplated_white_card():
    """A navy title must not open an inline white card: wrap it in .sx."""
    src = _tpl()
    bad = re.findall(r"background:#fff;border-radius:\d+px[^']*'\s*;?\s*\n?\s*html\+='<div class=\"sec-label\"", src)
    assert not bad
    assert "var _mapHtml='<div class=\"sx\">'" in src
    assert "html+='<div class=\"sx\"><div class=\"qh-card qh-flush\"" in src
    assert "phv.innerHTML='<div class=\"sx\"><div class=\"sec-label\">Stock Comparison</div>'" in src


def test_portfolio_sections_are_templated():
    src = _tpl()
    assert '<div class="sx"><h3 class="pfv-h3"><span class="sec-label">Your Portfolios</span></h3>' in src
    assert '<div class="sx"><h3 class="pfv-h3"><span class="sec-label">Alerts</span>' in src
    assert "if(ovl)h+='<div class=\"sx\">'+ovl+'</div>';" in src
    # every portfolio sub-tab pane is one section
    assert "'<div data-pane=\"'+x[0]+'\"><div class=\"sx\">'+x[2]+'</div></div>'" in src
