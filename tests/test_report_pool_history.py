# tests/test_report_pool_history.py
"""The sector page's Pool History and Share Shifts render what
models/sector_pool.py computed. Node evaluates the template's own functions
on a history built by the real module."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from models.sector_pool import sector_pool_history

TEMPLATE = Path(__file__).resolve().parents[1] / 'templates' / 'report.html'
YEARS = list(range(2016, 2026))


def _fn(src, name):
    one = re.search(r'^function ' + re.escape(name) + r'\(.*\}$', src, re.M)
    if one and one.group(0).count('{') == one.group(0).count('}'):
        return one.group(0)
    m = re.search(r'^function ' + re.escape(name) + r'\(.*?^\}', src, re.S | re.M)
    assert m, name
    return m.group(0)


def _row(t, rev, margin, growth, rating='BUY'):
    rev_by = {y: rev * (1 + growth) ** (y - 2016) for y in YEARS}
    oi_by = {y: v * margin for y, v in rev_by.items()}
    return {'ticker': t, 'company_name': t + ' Corp', 'sector': 'Tech',
            'rating': rating, 'revenue': rev_by[2025], 'operating_income': oi_by[2025],
            'edgar_history': {'revenue_history': rev_by,
                              'operating_income_history': oi_by}}


def _render(history, call, tmp_path):
    src = TEMPLATE.read_text(encoding='utf-8')
    names = ['_ppFy', '_ppFy1', '_ppPct', '_ppSgnPct', '_ppBn', '_ord',
             'renderPoolHistory', '_ppShiftRows', 'renderPoolShifts']
    js = '\n'.join(
        ["var RC={'BUY':'#1a9850','PASS':'#de2d26'};",
         'function _esc(s){return String(s);}function _attr(s){return String(s);}']
        + [_fn(src, n) for n in names]
        + ['var SECTOR_POOL=' + json.dumps({'Tech': {'history': history}}) + ';',
           'process.stdout.write(' + call + ');'])
    script = tmp_path / 'h.js'
    script.write_text(js, encoding='utf-8')
    return subprocess.run(['node', str(script)], capture_output=True, text=True,
                          check=True).stdout


pytestmark = pytest.mark.skipif(shutil.which('node') is None,
                                reason='node not installed')


def test_history_draws_every_year_and_states_the_split(tmp_path):
    rows = [_row('AAA', 100, 0.10, 0.12), _row('BBB', 80, 0.15, 0.0, 'PASS'),
            _row('CCC', 60, 0.08, 0.03)]
    h = sector_pool_history(rows)
    html = _render(h, "renderPoolHistory('Tech')", tmp_path)
    assert html.count('class="pph-bar') == len(h['points']) == 10
    assert 'Where the growth came from' in html
    assert 'Cycle position' in html and 'Profit concentration' in html
    assert 'FY2018–20 with FY2023–25' in html
    d = h['decomposition']
    assert '%+.1f%%/yr' % (d['pool_cagr'] * 100) in html


def test_incomplete_year_is_hollow(tmp_path):
    rows = [_row('AAA', 100, 0.10, 0.12), _row('BBB', 80, 0.15, 0.0),
            _row('CCC', 60, 0.08, 0.03)]
    rows[0]['edgar_history']['revenue_history'][2026] = 200
    rows[0]['edgar_history']['operating_income_history'][2026] = 30
    html = _render(sector_pool_history(rows), "renderPoolHistory('Tech')", tmp_path)
    assert html.count('class="pph-bar inc"') == 1


def test_shifts_list_gainers_and_losers_with_click_through(tmp_path):
    rows = [_row('AAA', 100, 0.10, 0.12), _row('BBB', 80, 0.15, -0.02, 'PASS'),
            _row('CCC', 60, 0.08, 0.03)]
    html = _render(sector_pool_history(rows), "renderPoolShifts('Tech')", tmp_path)
    up, down = html.split('Lost share')
    assert 'data-tk="AAA"' in up and 'data-tk="BBB"' in down
    assert 'onclick="openDet(this.dataset.tk)"' in html
    assert 'border-left-color:#de2d26' in down     # BBB's PASS rating


def test_no_history_renders_nothing(tmp_path):
    assert _render(None, "renderPoolHistory('Tech')+'|'+renderPoolShifts('Tech')",
                   tmp_path) == '|'
