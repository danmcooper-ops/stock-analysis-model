# tests/test_report_portfolio_news.py
"""Portfolios › a portfolio › News: a sub-tab right after Summary that pools
the members' company headlines and keeps only those naming a member."""
import os
import re

_TEMPLATE = os.path.join(os.path.dirname(__file__), '..', 'templates', 'report.html')


def _src():
    return open(_TEMPLATE, encoding='utf-8').read()


def test_news_pane_follows_summary():
    s = _src()
    i = s.index("var panes=[['summary','Summary'")
    block = s[i:i + 2500]
    assert block.index("['news','News'") < block.index("['members','Members'")
    assert 'id="pf-news"' in block


def test_news_pane_is_shown_by_the_tab_css():
    assert '.pf-tabbed[data-tab="news"]>[data-pane="news"]' in _src()


def test_news_is_curated_to_headlines_naming_a_member():
    s = _src()
    body = s[s.index('function _pfNewsItems('):s.index('function _pfNewsHtml(')]
    # sector context is left out, duplicates are folded, and an item that
    # names no member is dropped
    assert "n.scope==='sector'" in body
    assert 'seen[' in body
    assert 'if(!tks.length)return;' in body
    # waits for the details parts that carry news_headlines
    html = s[s.index('function _pfNewsHtml('):s.index('function _pfNewsFilt(')]
    assert '_loadDetails(' in html and re.search(r'_PFN_MAX', html)
