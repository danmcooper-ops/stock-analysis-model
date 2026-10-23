"""Every section of the report carries a title in the same navy box the
sector pages use (.pp-section-label there, .sec-label everywhere else)."""
import re
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parents[1] / 'templates' / 'report.html'


def _tpl():
    return TEMPLATE.read_text(encoding='utf-8')


def test_sec_label_shares_the_sector_box_style():
    src = _tpl()
    assert '.pp-section-label,.sec-label{display:inline-flex;' in src
    assert re.search(r'\.sec-label\{background:var\(--hdr-bg\);', src)
    assert '[data-theme="dark"] .sec-label{background:var(--hdr-bg) !important;}' in src


def test_old_unboxed_title_classes_are_gone():
    src = _tpl()
    for cls in ('xsect-title', 'xsect-chart-title', 'mac-regime-k'):
        assert f'class="{cls}"' not in src, cls


def test_section_headings_are_boxed():
    src = _tpl()
    # Portfolio and Macro headings put their label in the box.
    assert not re.search(r'<h3 class="pfv-h3"[^>]*>(?!<span class="sec-label">)', src)
    assert not re.search(r'<h3 class="mac-h"[^>]*>(?!<span class="sec-label">)', src)
    # Ticker-page cards: every title bar holds a boxed label.
    assert src.count('<span class="dg-icon"></span>') == \
        src.count('<span class="dg-icon"></span><span class="sec-label">')
    for title in ('Gates Summary', 'Composite Score Distribution', 'Gate Pass Rates',
                  'World MoS Map', 'Stock Price Comparison', 'Market regime',
                  'Your Portfolios', 'Summary'):
        assert re.search(r'class="sec-label"[^>]*>' + re.escape(title) + '<', src), title


def test_market_regime_card_is_on_the_macro_overview_only():
    src = _tpl()
    body = src.split('function renderMacro(){', 1)[1].split('\n}\n', 1)[0]
    assert "var h=curMacSub==='overview'?_macRegimeHTML(true):'';" in body
