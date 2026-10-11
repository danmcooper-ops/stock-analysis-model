# tests/test_report_profit_pool.py
"""The Cross-Sector Profit Pool counts each issuer once.

The universe lists some issuers several times — share classes, ordinary
shares beside their ADR, and preferred series carrying the parent's
statements (FNMA plus 14 preferreds, FMCC plus 21). Summing rows counted
Fannie Mae 15 times, and with bank operating income also overstated, gave
Financial Services a 124% operating margin and 63% of all profit on
2026-10-06. Node evaluates the template's own aggregation.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

TEMPLATE = Path(__file__).resolve().parents[1] / 'templates' / 'report.html'


def _fn(src, name):
    """Source of a top-level function: a one-liner, or up to the first line
    that is exactly `}`."""
    one = re.search(r'^function ' + re.escape(name) + r'\(.*\}$', src, re.M)
    if one and one.group(0).count('{') == one.group(0).count('}'):
        return one.group(0)
    m = re.search(r'^function ' + re.escape(name) + r'\(.*?^\}', src, re.S | re.M)
    assert m, name
    return m.group(0)


def _pool(rows, tmp_path):
    src = TEMPLATE.read_text(encoding='utf-8')
    js = '\n'.join([
        'var window={};function setTimeout(){}function _ppDraw(){}',
        'function _num(v){return typeof v==="number"&&isFinite(v)?v:null;}',
        'function _esc(s){return s;}function _attr(s){return s;}',
        _fn(src, '_ppColM'), _fn(src, '_ppIssuerKey'),
        _fn(src, 'renderCrossSectorProfitPool'),
        'var DATA=' + json.dumps(rows) + ';',
        'renderCrossSectorProfitPool();',
        'console.log(JSON.stringify(window._xsPoolAll));',
    ])
    script = tmp_path / 'pool.js'
    script.write_text(js, encoding='utf-8')
    out = subprocess.run(['node', str(script)], capture_output=True, text=True, check=True)
    return {a['name']: a for a in json.loads(out.stdout)}


@pytest.mark.skipif(shutil.which('node') is None, reason='node not installed')
def test_duplicate_listings_count_once(tmp_path):
    fannie = {'sector': 'Financial Services', 'revenue': 29.2e9, 'operating_income': 18.0e9}
    rows = ([dict(fannie, ticker=t) for t in ('FNMA', 'FNMAO', 'FNMAS', 'FNMAT')]
            + [{'ticker': 'JPM', 'sector': 'Financial Services',
                'revenue': 182.4e9, 'operating_income': 72.6e9},
               {'ticker': 'AAPL', 'sector': 'Technology',
                'revenue': 416.2e9, 'operating_income': 133.1e9},
               {'ticker': 'MSFT', 'sector': 'Technology',
                'revenue': 281.7e9, 'operating_income': 128.5e9}])
    pool = _pool(rows, tmp_path)
    fs = pool['Financial Services']
    assert fs['n'] == 2
    assert fs['rev'] == pytest.approx(29.2e9 + 182.4e9)
    assert fs['oi'] == pytest.approx(18.0e9 + 72.6e9)
    assert pool['Technology']['n'] == 2


@pytest.mark.skipif(shutil.which('node') is None, reason='node not installed')
def test_distinct_issuers_with_different_figures_both_count(tmp_path):
    rows = [{'ticker': 'A', 'sector': 'Energy', 'revenue': 10.0, 'operating_income': 1.0},
            {'ticker': 'B', 'sector': 'Energy', 'revenue': 10.0, 'operating_income': 2.0},
            {'ticker': 'C', 'sector': 'Utilities', 'revenue': 10.0, 'operating_income': 1.0}]
    pool = _pool(rows, tmp_path)
    assert pool['Energy']['n'] == 2 and pool['Energy']['oi'] == 3.0
    assert pool['Utilities']['n'] == 1
