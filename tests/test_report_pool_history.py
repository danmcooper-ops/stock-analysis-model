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

from models.sector_pool import (economic_pool, industry_pools, pool_structure,
                                 sector_pool_history, universe_totals)

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


def _render(history, call, tmp_path, entry=None):
    src = TEMPLATE.read_text(encoding='utf-8')
    names = ['_ppFy', '_ppFy1', '_ppPct', '_ppSgnPct', '_ppBn', '_ord',
             'renderPoolHistory', '_ppShiftRows', 'renderPoolShifts',
             '_ppIndCls', 'renderPoolIndustries', '_ppEpRows', 'renderPoolEcon',
             '_ppStructureStats']
    sp = dict(entry or {}, history=history)
    js = '\n'.join(
        ["var RC={'BUY':'#1a9850','PASS':'#de2d26'};",
         'function _esc(s){return String(s);}function _attr(s){return String(s);}']
        + [_fn(src, n) for n in names]
        + ['var SECTOR_POOL=' + json.dumps({'Tech': sp}) + ';',
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


def _step2_rows():
    ic = {'2025': 400.0}
    return [dict(_row('AAA', 100, 0.30, 0.12), industry='Semis', spread=0.25,
                 _ic_by_year=ic, mcap=5000),
            dict(_row('AAB', 90, 0.25, 0.10), industry='Semis', spread=0.15,
                 _ic_by_year=ic, mcap=3000),
            dict(_row('AAC', 85, 0.20, 0.08), industry='Semis', spread=0.08,
                 _ic_by_year=ic, mcap=2000),
            dict(_row('BBA', 80, 0.05, 0.0, 'PASS'), industry='Services', spread=-0.04,
                 _ic_by_year=ic, mcap=300),
            dict(_row('BBB', 70, 0.06, 0.01), industry='Services', spread=-0.02,
                 _ic_by_year=ic, mcap=350),
            dict(_row('BBC', 60, 0.04, 0.02), industry='Services', spread=0.01,
                 _ic_by_year=ic, mcap=250)]


def test_industries_view_draws_and_tabulates_each_industry(tmp_path):
    rows = _step2_rows()
    html = _render(None, "renderPoolIndustries('Tech')", tmp_path,
                   {'industries': industry_pools(rows)})
    assert html.count('class="ppi-bar') == 2
    assert 'ppi-pos' in html and 'ppi-neg' in html          # spread colours
    assert html.count('<tr><td>') == 2 and 'Pool CAGR' in html


def test_econ_states_both_market_shares_and_the_exclusions(tmp_path):
    rows = _step2_rows()
    uni = universe_totals(rows)
    e = economic_pool(rows, uni)
    html = _render(None, "renderPoolEcon('Tech')", tmp_path, {'economic': e})
    assert 'of US operating profit but' in html
    assert 'Largest value creators' in html and 'data-tk="AAA"' in html
    assert 'data-tk="BBA"' in html.split('Largest value destroyers')[1]
    assert 'Not measured' not in html                       # nothing excluded


def test_structure_stats_render_the_four_tiles(tmp_path):
    rows = _step2_rows()
    st = pool_structure(rows, universe_totals(rows))
    html = _render(None, "_ppStructureStats(SECTOR_POOL.Tech.structure_stats)",
                   tmp_path, {'structure_stats': st})
    for title in ('Profit concentration', 'Margin spread', 'Loss-makers',
                  'Price of the pool'):
        assert title in html


def _render_signals(entry, tmp_path):
    src = TEMPLATE.read_text(encoding='utf-8')
    names = ['_ppFy', '_ppFy1', '_ppPct', '_ppSgnPct', '_ord', '_ppRankOf', '_ppWin', '_ppfSpark',
             '_ppfEvidence', '_ppfExpVal', '_ppfChips', '_ppfExposure', '_ppfBalance',
             '_ppfCard', '_ppfLive', '_ppfCol', 'renderPoolSectorSignals']
    consts = '\n'.join(re.search(r'^var %s=.*$' % v, src, re.M).group(0)
                       for v in ('_PPF_ST', '_PPF_ORDER', '_PPF_TYPE'))
    js = '\n'.join(["var RC={'BUY':'#1a9850','PASS':'#de2d26'};",
                    'function _esc(s){return String(s);}function _attr(s){return String(s);}'
                    'function _linkifyTickers(s){return s;}',
                    consts] + [_fn(src, n) for n in names]
                   + ['var SECTOR_POOL=' + json.dumps({'Real Estate': entry}) + ';',
                      "process.stdout.write(renderPoolSectorSignals('Real Estate'));"])
    script = tmp_path / 's.js'
    script.write_text(js, encoding='utf-8')
    return subprocess.run(['node', str(script)], capture_output=True, text=True,
                          check=True).stdout


def test_forces_read_as_bulleted_rows_with_plain_word_statuses(tmp_path):
    from models.sector_forces import evaluate_forces
    from tests.test_sector_forces import _sidecar
    side = _sidecar('DGS10', [3.0 + 0.02 * k for k in range(120)])
    side['sector_data'] = {'Real Estate': {'etf': 'XLRE', 'rs_3m': -0.10, 'rs_6m': -0.17}}
    html = _render_signals({'forces': evaluate_forces('Real Estate', side, {})}, tmp_path)
    tw, hw = html.split('>Headwinds<')
    assert html.count('<div class="ppf ppf-') == 6
    assert 'ppf-st ppf-active"><i class="ppf-dot"></i>Acting now' in hw
    assert 'ppf-st ppf-dormant"><i class="ppf-dot"></i>Not acting' in tw
    assert 'Not measured' in html and 'Qualitative' not in html
    card = hw.split('Interest-rate sensitivity')[1].split('<div class="ppf ppf-')[0]
    assert '<ul class="ppf-ul"><li><b>Evidence:</b> Test series 5.38%' in card
    assert '<polyline' in card and '<b>Why it matters:</b> Cap rates' in card
    assert 'XLRE has trailed the market by 10.0% over 3 months' in html
    assert '0 of 3 acting</small>' in tw and '1 of 3 acting</small>' in hw
    assert 'Acting now: the evidence is strong and holding' in html       # the key


def test_nothing_is_hidden_behind_expand_or_collapse(tmp_path):
    from models.sector_forces import evaluate_forces
    from tests.test_sector_forces import _sidecar
    side = _sidecar('DGS10', [3.0 + 0.02 * k for k in range(120)])
    html = _render_signals({'forces': evaluate_forces('Real Estate', side, {})}, tmp_path)
    assert '<details' not in html and '<summary' not in html
    hw = html.split('>Headwinds<')[1]
    # moving forces first, then not acting, then not measured
    assert (hw.index('Interest-rate sensitivity') < hw.index('Remote work')
            and hw.index('ppf-active') < hw.index('ppf-qualitative'))
    assert hw.count('<b>Evidence:</b> No data series tracks this') == 2


def test_signals_fall_back_to_the_plain_lists(tmp_path):
    html = _render_signals({'headwinds': ['A — x'], 'tailwinds': ['B — y']}, tmp_path)
    assert '<li' in html and 'ppf' not in html


def test_market_check_names_the_unit_without_the_3_month_reading(tmp_path):
    from models.sector_forces import evaluate_forces
    side = {'as_of': '2026-10-08', 'sector_data': {'Real Estate': {'etf': 'XLRE', 'rs_6m': 0.251}}}
    html = _render_signals({'forces': evaluate_forces('Real Estate', side, {})}, tmp_path)
    assert 'XLRE has beaten the market by 25.1% over 6 months.' in html

def test_force_cards_show_reach_and_who_feels_it(tmp_path):
    from models.sector_forces import evaluate_forces
    from tests.test_sector_forces import _sidecar, _re_rows
    side = _sidecar('DGS10', [3.0 + 0.02 * k for k in range(120)])
    res = evaluate_forces('Real Estate', side, {}, _re_rows())
    html = _render_signals({'forces': res}, tmp_path)
    assert 'Headwinds outweigh tailwinds' in html
    card = html.split('Interest-rate sensitivity')[1].split('class="ppf ppf-')[0]
    assert 'of sector</span>' in card.split('<ul')[0]                     # reach beside the title
    assert '<li><b>Most exposed</b> (the highest net debt / EBITDA)' in card
    assert '<li><b>Least exposed:</b>' in card and '<li><b>Reach:</b>' in card
    assert 'data-tk="LEV0"' in card and '\u00d7' in card


def test_own_window_label_names_both_blocks(tmp_path):
    inds = [{'industry': 'A', 'n': 5, 'revenue_share': 0.6, 'pool_share': 0.6, 'margin': 0.2,
             'median_spread': 0.05, 'pool_cagr': 0.1, 'window': [2020, 2025, 3], 'own_window': False},
            {'industry': 'B', 'n': 4, 'revenue_share': 0.4, 'pool_share': 0.4, 'margin': 0.1,
             'median_spread': 0.01, 'pool_cagr': 0.2, 'window': [2018, 2023, 3], 'own_window': True}]
    html = _render(None, "renderPoolIndustries('Tech')", tmp_path, {'industries': inds})
    assert '(FY2016\u201318 to FY2021\u201323)' in html
    assert html.count('ppi-dim" title') == 1
