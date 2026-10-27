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
                  'World MoS Map', 'Stock Price Comparison', 'The US Economy',
                  'The International Economy',
                  'Your Portfolios', 'Summary'):
        assert re.search(r'class="sec-label"[^>]*>' + re.escape(title) + '<', src), title


def test_market_regime_card_is_on_the_macro_overview_only():
    src = _tpl()
    body = src.split('function renderMacro(){', 1)[1].split('\n}\n', 1)[0]
    assert "var h=curMacSub==='overview'?_macOverviewHTML():_macSectionHTML(curMacSub);" in body
    assert '_macRegimeHTML' not in body


def test_sector_etf_section_charts_the_fund_against_spy():
    src = _tpl()
    fn = src.split('function renderPoolEtf(sec){', 1)[1].split('\n}\n', 1)[0]
    assert '_etfcBox(t)+_etfDetail(r,true)' in fn
    # Redrawn with the pool charts: on render, sub-tab switch and resize.
    assert src.split('function _pphDrawAll(root){', 1)[1].lstrip().startswith('_etfcDrawAll(root);')
    py = (Path(__file__).resolve().parents[1] / 'scripts' / 'report_html.py').read_text(encoding='utf-8')
    assert 'for _ftk, _fsec in etf_holdings.SPDR_FUNDS.items():' in py


def test_etf_views_carry_no_rating_mix():
    src = _tpl()
    assert '_etfMix' not in src and 'Rating mix by weight' not in src
