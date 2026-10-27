# tests/test_report_sector_compare.py
"""Guard: the Sector Price Comparison chart offers every sector SPDR.

The template keeps its own sector -> fund map (_SECTOR_SPDR) because the
funds ship only as px/ shards, with nothing in prices_meta.json naming their
sector. If data/etf_holdings.SPDR_FUNDS gains, drops or renames a fund, the
chart would silently stop offering it, so pin the two together.
"""
import os
import re

from data import etf_holdings

_TEMPLATE = os.path.join(os.path.dirname(__file__), '..', 'templates',
                         'report.html')


def _tpl():
    return open(_TEMPLATE, encoding='utf-8').read()


def _js_map():
    m = re.search(r'var _SECTOR_SPDR=\{(.*?)\};', _tpl(), re.S)
    assert m, 'could not find _SECTOR_SPDR in the template'
    return dict(re.findall(r"'([^']+)':'([A-Z]+)'", m.group(1)))


def test_sector_spdr_map_matches_etf_holdings():
    want = {sec: tk for tk, sec in etf_holdings.SPDR_FUNDS.items() if sec}
    assert _js_map() == want


def test_comparisons_sit_behind_a_dropdown():
    tpl = _tpl()
    # The indices and funds are rows of one collapsible list, not open chips.
    assert 'id="sector-bench-list"' in tpl
    assert 'onclick="sectorBenchDD(this)"' in tpl
    assert 'id="sector-bench-btns"' not in tpl
