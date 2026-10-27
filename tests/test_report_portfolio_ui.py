"""Report Portfolios view: private holdings, value & risk, breakdowns, CSV.

Holdings (lots, cash, realized sales) must never reach the published page;
the browser computes P&L from a positions file the reader loads locally,
with a twin of models.portfolio_positions.holdings_summary.
"""
import json
import os
import re
import shutil
import subprocess
from datetime import date

import pytest

from models import portfolio_groups as pg
from models import portfolio_positions as pp

TEMPLATE = os.path.join(os.path.dirname(__file__), '..', 'templates', 'report.html')
node = pytest.mark.skipif(shutil.which('node') is None, reason='node not installed')


def _src():
    return open(TEMPLATE, encoding='utf-8').read()


def _fn(name):
    return re.search(r'^function ' + re.escape(name) + r'\(.*?^\}', _src(), re.S | re.M).group(0)


def _run_js(tmp_path, body):
    num = re.search(r'^function _num\(v\)\{.*?\}$', _src(), re.M).group(0)
    script = tmp_path / 't.js'
    script.write_text(num + '\n' + body, encoding='utf-8')
    r = subprocess.run(['node', str(script)], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads(r.stdout)


BOOK = {
    'positions': {
        'AAA': [{'shares': 10.0, 'cost': 10.0, 'date': '2024-01-01', 'note': 'core'},
                {'shares': 5.0, 'cost': 16.0, 'date': '2025-02-03'}],
        'BRK-A': [{'shares': 1.0, 'cost': 100.0, 'date': '2024-01-01'}],
        'GONE': [{'shares': 3.0, 'cost': 2.0, 'date': '2024-05-05'}],
    },
    'realized': [{'ticker': 'Z', 'shares': 2.0, 'cost': 1.0, 'price': 4.0, 'date': '2026-03-01'},
                 {'ticker': 'Z', 'shares': 1.0, 'cost': 5.0, 'price': 4.0, 'date': '2025-03-01'}],
    'cash': 250.0,
}
ROWS = {'AAA': {'ticker': 'AAA', 'price': 20.0},
        'BRK-B': {'ticker': 'BRK-B', 'price': '300', 'listing_aliases': ['BRK-A']},
        'ZERO': {'ticker': 'ZERO', 'price': 0}}


@node
def test_holdings_math_matches_python(tmp_path):
    book = pp.normalize_book(BOOK, 'p')
    want = pp.holdings_summary(book, ROWS, pg.listing_aliases(ROWS), today='2026-10-24')
    got = _run_js(tmp_path, _fn('_pfToday') + '\n' + _fn('_pfHoldCalc') + '\n'
                  + f'var byTk={json.dumps(ROWS)},alias={{"BRK-A":"BRK-B"}};\n'
                  + f'console.log(JSON.stringify(_pfHoldCalc({json.dumps(book)},byTk,alias,"2026-10-24")));')
    assert [r['ticker'] for r in got['rows']] == [r['ticker'] for r in want['rows']]
    for a, b in zip(got['rows'], want['rows'], strict=True):
        for k in ('row', 'shares', 'cost', 'cost_value', 'first', 'lots', 'price', 'market_value',
                  'pnl', 'pnl_pct', 'weight', 'note'):
            assert a[k] == pytest.approx(b[k]) if isinstance(b[k], float) else a[k] == b[k], k
    for k, v in want['totals'].items():
        assert got['totals'][k] == (pytest.approx(v) if isinstance(v, float) else v), k


@node
def test_parse_tickers_size_buckets_and_csv(tmp_path):
    body = '\n'.join(_fn(n) for n in ('_pfParseTickers', '_pfSizeBucket', '_pfBrkKey',
                                      '_pfBrkCount', '_pfCsvCell'))
    sizes = re.search(r"var _PF_SIZES=\[.*?\];", _src()).group(0)
    out = _run_js(tmp_path, sizes + '\n' + body + '\nconsole.log(JSON.stringify({'
                  'tk:_pfParseTickers("aapl, $msft\\nbrk-b;; aapl  x/y TOOLONGTICKER123"),'
                  'sz:[3e11,5e10,3e9,1e8,null,-1].map(function(m){return _pfSizeBucket({mcap:m});}),'
                  'bc:_pfBrkCount([{country:"US"},{country:"US"},{country:""},{}],"country"),'
                  'csv:[_pfCsvCell(\'a,"b"\'),_pfCsvCell(0.1+0.2),_pfCsvCell(null),_pfCsvCell("x")]}));')
    assert out['tk'] == ['AAPL', 'MSFT', 'BRK-B']
    assert out['sz'] == ['Mega (≥ $200B)', 'Large ($10–200B)', 'Mid ($2–10B)', 'Small (< $2B)', None, None]
    assert out['bc'] == {'c': {'US': 2}, 'n': 2, 'none': 2}
    assert out['csv'] == ['"a,""b"""', '0.3', '', 'x']


@node
def test_book_validation(tmp_path):
    out = _run_js(tmp_path, _fn('_pfBookCheck') + '\nconsole.log(JSON.stringify(['
                  '_pfBookCheck({version:1,portfolios:{p:' + json.dumps(BOOK) + '}}),'
                  '_pfBookCheck({portfolios:[]}),'
                  '_pfBookCheck({version:2,portfolios:{}}),'
                  '_pfBookCheck({portfolios:{p:{positions:{X:[{shares:0,cost:1,date:"2024-01-01"}]}}}}),'
                  '_pfBookCheck({portfolios:{p:{cash:-1}}})]));')
    assert out[0] is None
    assert 'Expected portfolio/positions.json' in out[1]
    assert 'version 2' in out[2]
    assert '(p X)' in out[3] and '(p cash)' in out[4]


class TestPrivacy:
    def test_book_fields_never_in_definitions_or_links(self):
        fields = re.search(r"var _PF_FIELDS=\[(.*?)\];", _src()).group(1)
        for k in ('positions', 'realized', 'cash'):
            assert f"'{k}'" not in fields
        for f in ('pfExport', 'pfShare', '_pfClean', '_pfAddIncoming'):
            assert '_pfBook' not in _fn(f), f
        # The book key is only ever read, written or removed locally.
        s = _src()
        local = re.findall(r'localStorage\.(?:get|set|remove)Item\(_PF_BOOK_KEY', s)
        assert local and s.count('_PF_BOOK_KEY') == len(local) + 1      # + its definition

    def test_render_never_publishes_a_book(self, tmp_path, monkeypatch):
        """With a real positions file beside a held portfolio, no output file
        carries a single number from it."""
        from data.snapshot_store import write_snapshot_file
        from scripts.report_html import build_html
        res = tmp_path / 'out'
        res.mkdir()
        rows = [{'ticker': 'AAA', 'price': 20.0, 'rating': 'BUY', 'sector': 'Energy'},
                {'ticker': 'BBB', 'price': 30.0, 'rating': 'HOLD', 'sector': 'Energy'}]
        write_snapshot_file(str(res / 'results_2026-10-23.json'), {'results': rows})
        pf, pos = tmp_path / 'pf.json', tmp_path / 'positions.json'
        pg.save_portfolios({'version': 1, 'portfolios': [
            {'id': 'mine', 'name': 'Mine', 'tickers': ['AAA', 'BBB'], 'weighting': 'held'}]}, str(pf))
        pp.save_books({'mine': {'positions': {
            'AAA': [{'shares': 4321.0, 'cost': 98.7654, 'date': '2024-01-02'}],
            'BBB': [{'shares': 1234.0, 'cost': 54.3219, 'date': '2024-01-02'}]},
            'cash': 77777.0}}, str(pos))
        monkeypatch.setenv('PORTFOLIOS_FILE', str(pf))
        monkeypatch.setenv('POSITIONS_FILE', str(pos))
        build_html([dict(r) for r in rows], str(res / 'r.html'), run_date=date(2026, 10, 23))
        secret = ('4321', '98.7654', '1234.0', '54.3219', '77777')
        for root, _, files in os.walk(res):
            for f in files:
                text = open(os.path.join(root, f), encoding='utf-8', errors='ignore').read()
                for s in secret:
                    assert s not in text, f'{s} leaked into {f}'


def test_hooks_are_wired():
    s = _src()
    assert '_pfBookManageHtml()' in _fn('_pfRenderManage')
    assert '_pfRiskHtml(id)' in _fn('_pfDetailHtml') and '_pfBrkPanel(id,st)' in _fn('_pfDetailHtml')
    assert '_pfAttrib(id,r)' in _fn('_pfContribDraw')
    assert '_pfOverlapHtml(defs)' in _fn('_pfOverviewHtml')
    assert '_pfLivePaint()' in _fn('_lqPaint')
    assert "_lqSet('pf'," in _fn('_pfmTable')
    # No prompt() left in the portfolio flows.
    start, end = s.index('/* === PORTFOLIOS ==='), s.index('/* === FILTER STATE SNAPSHOT')
    assert not re.search(r"[=(;]\s*prompt\(", s[start:end])
