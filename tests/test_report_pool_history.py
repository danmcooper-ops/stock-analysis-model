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
             '_ppIndCls', '_ppiV', '_ppiTh', 'renderPoolIndustries', '_ppEpRows', 'renderPoolEcon',
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
    assert html.count('<tr><td data-v=') == 2 and 'Pool CAGR' in html


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


def _render_signals(entry, tmp_path, stored=None):
    src = TEMPLATE.read_text(encoding='utf-8')
    names = ['_ppFy', '_ppFy1', '_ppPct', '_ppSgnPct', '_ord', '_ppRankOf', '_ppWin', '_ppfSpark',
             '_ppfEvidence', '_ppfExpVal', '_ppfChips', '_ppfExposure', '_ppfBalance',
             '_ppfCard', '_ppfLive', '_ppfCollapsedState', '_ppfCol', 'renderPoolSectorSignals']
    consts = '\n'.join(re.search(r'^var %s=.*$' % v, src, re.M).group(0)
                       for v in ('_PPF_ST', '_PPF_ORDER', '_PPF_TYPE'))
    store = ('var localStorage={getItem:function(){return %s;}};' % json.dumps(json.dumps(stored))
             if stored is not None else '')
    js = '\n'.join([store + "var RC={'BUY':'#1a9850','PASS':'#de2d26'};",
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
    assert 'of sector' not in card                                         # reach only once
    assert card.count('<b>Reach:</b>') == 1
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


def test_an_industry_force_names_its_industry_once(tmp_path):
    from models.sector_forces import evaluate_forces
    from tests.test_sector_forces import _entry
    from tests.test_sector_pool import YEARS, _flat
    rows = [dict(_flat(t, YEARS, 100 + i, 0.2, 0.08), industry='Software - Infrastructure', sector='Technology')
            for i, t in enumerate(('MSFT', 'ORCL', 'X'))]
    html = _render_signals({'forces': evaluate_forces('Technology', None, _entry(0.14), rows)}, tmp_path)
    card = html.split('Software dollar share keeps rising')[1].split('<div class="ppf ppf-')[0]
    assert card.count('Software - Infrastructure') == 1
    assert '<b>Who gains:</b> 3 companies' in card


def _render_co_table(shown, tail, tmp_path):
    src = TEMPLATE.read_text(encoding='utf-8')
    names = ['_ppPct', '_ppSgnPct', '_ppiV', '_ppiTh', '_ppSkewTd', 'renderPoolCompanyTable']
    js = '\n'.join(["var RC={'BUY':'#1a9850','PASS':'#de2d26'};",
                    'function _esc(s){return String(s);}function _attr(s){return String(s);}',
                    re.search(r'^var _PP_BS_INDUSTRIES=.*$', src, re.M).group(0),
                    re.search(r'^var _PPI_RANK=.*$', src, re.M).group(0)]
                   + [_fn(src, n) for n in names]
                   + ['process.stdout.write(renderPoolCompanyTable(%s,%s));'
                      % (json.dumps(shown), json.dumps(tail))])
    script = tmp_path / 'co.js'
    script.write_text(js, encoding='utf-8')
    return subprocess.run(['node', str(script)], capture_output=True, text=True,
                          check=True).stdout


def _co(t, rev, pool, **kw):
    return dict({'ticker': t, 'company_name': t + ' Inc', 'rating': 'BUY', 'sector': 'Basic Materials',
                 'industry': 'Gold', 'pp_revenue_share': rev, 'pp_profit_share': pool,
                 'operating_margin': 0.2, 'spread': 0.05, '_gate_pool_share': 0.03}, **kw)


def test_company_table_mirrors_the_industries_table(tmp_path):
    shown = [_co('A%02d' % k, 0.02, 0.02 + (0.001 * k)) for k in range(25)]
    tail = [_co('T%d' % k, 0.001, 0.0005) for k in range(10)]
    html = _render_co_table(shown, tail, tmp_path)
    assert html.count('<tr class="ppi-co"') == 25
    assert '<td>Others <span class="ppi-dim">(10 companies)</span>' in html
    assert html.index('A24') < html.index('A00')                  # by pool share
    assert '>Pool share trend</th>' in html and '+3.0%/yr' in html
    assert 'data-tk="A00" onclick="openDet(this.dataset.tk)"' in html
    assert 'class="ppi-tbl ppi-sort"' in html


def test_company_table_highlights_and_blanks(tmp_path):
    shown = [_co('HI', 0.10, 0.125), _co('LO', 0.10, 0.075), _co('MID', 0.10, 0.1),
             _co('JPM', 0.10, 0.1, sector='Financial Services', industry='Banks - Diversified'),
             _co('NEW', 0.10, 0.1, _gate_pool_share=None)]
    html = _render_co_table(shown, [], tmp_path)
    row = lambda t: html.split('data-tk="%s"' % t)[1].split('</tr>')[0]   # noqa: E731
    assert 'ppi-hi' in row('HI') and 'ppi-lo' in row('LO') and 'ppi-' not in row('MID').replace('ppi-dim', '').replace('ppi-rt', '')
    assert row('JPM').count('<td data-v="">—</td>') == 1                # spread only
    assert row('NEW').endswith('<td data-v="">—</td>')
    assert '<td>Others' not in html and 'pooled as Others' not in html


def test_each_column_header_toggles_its_bullets(tmp_path):
    """Collapsed, a column keeps one line per force: status and name."""
    from models.sector_forces import evaluate_forces
    from tests.test_sector_forces import _sidecar
    side = _sidecar('DGS10', [3.0 + 0.02 * k for k in range(120)])
    forces = {'forces': evaluate_forces('Real Estate', side, {})}
    html = _render_signals(forces, tmp_path)
    assert html.count('onclick="_ppfToggleCol(this)"') == 2
    assert html.count('aria-expanded="true"') == 2 and 'ppf-collapsed' not in html
    shut = _render_signals(forces, tmp_path, stored={'headwind': True})
    assert '<div class="ppf-colwrap ppf-collapsed" data-kind="headwind">' in shut
    assert '<div class="ppf-colwrap" data-kind="tailwind">' in shut
    # the bullets are still in the page (CSS hides them), so expanding is instant
    assert shut.split('data-kind="headwind"')[1].count('<ul class="ppf-ul">') == 3


def test_collapsed_columns_hide_only_bullets_and_type():
    css = TEMPLATE.read_text(encoding='utf-8')
    assert '.ppf-collapsed .ppf-ul,.ppf-collapsed .ppf-meta{display:none;}' in css


def _sort_rows(html, col, desc=True, numeric=True):
    """Apply _ppiSort's ordering rule to a rendered table (Node has no DOM):
    rows by the column's data-v, blanks last, pinned rows at the bottom."""
    body = html.split('<tbody>')[1].split('</tbody>')[0]
    rows = re.findall(r'<tr[^>]*>.*?</tr>', body, re.S)
    pin = [r for r in rows if 'ppi-pin' in r.split('>')[0]]
    rest = [r for r in rows if r not in pin]
    def key(r):
        v = re.findall(r'<td[^>]*?data-v="([^"]*)"', r)
        return v[col] if col < len(v) else ''
    have = [r for r in rest if key(r) != '']
    blank = [r for r in rest if key(r) == '']
    have.sort(key=lambda r: float(key(r)) if numeric else key(r), reverse=desc)
    return have + blank + pin


def test_company_table_headers_sort_and_pin_others(tmp_path):
    shown = [_co('A', 0.10, 0.20, rating='PASS', _gate_pool_share=None),
             _co('B', 0.20, 0.10, rating='BUY', _gate_pool_share=0.05),
             _co('C', 0.05, 0.05, rating='HOLD', _gate_pool_share=-0.02)]
    html = _render_co_table(shown, [_co('T', 0.01, 0.01)], tmp_path)
    assert html.count('onclick="_ppiSort(this)"') == 7
    assert '<th data-t="n" aria-sort="descending" onclick' in html            # Pool, as rendered
    assert '<th data-t="s" onclick' in html                                    # Company sorts as text
    assert '<tr class="ppi-pin"><td>Others' in html
    rows = _sort_rows(html, 6)                                                 # trend, high to low
    order = [re.search(r'data-tk="(\w+)"', r).group(1) if 'data-tk' in r else 'Others' for r in rows]
    assert order == ['B', 'C', 'A', 'Others']                                  # blank last, Others pinned
    rows = _sort_rows(html, 1)                                                 # rating rank
    assert [re.search(r'data-tk="(\w+)"', r).group(1) for r in rows[:3]] == ['B', 'C', 'A']


def test_sorter_keeps_blanks_last_and_pinned_rows_at_the_bottom():
    src = TEMPLATE.read_text(encoding='utf-8')
    fn = _fn(src, '_ppiSort')
    assert "classList.contains('ppi-pin')" in fn and 'body.concat(pin)' in fn
    assert "if(x===''||x==null)return (y===''||y==null)?0:1;" in fn


def _subtabs(body, has, tmp_path, stored=None):
    src = TEMPLATE.read_text(encoding='utf-8')
    store = ('var localStorage={getItem:function(){return %s;}};' % json.dumps(stored)
             if stored is not None else '')
    js = '\n'.join([store + 'var window={};function _esc(s){return String(s);}',
                    re.search(r'^var _PP_SUBTABS=.*$', src, re.M).group(0),
                    _fn(src, '_ppSubTabCur'), _fn(src, '_ppSubTabs'),
                    'process.stdout.write(_ppSubTabs(%s,%s));' % (json.dumps(body), json.dumps(has))])
    script = tmp_path / 'tabs.js'
    script.write_text(js, encoding='utf-8')
    return subprocess.run(['node', str(script)], capture_output=True, text=True,
                          check=True).stdout


def test_sub_tabs_skip_empty_groups_and_fall_back(tmp_path):
    has = {'overview': True, 'forces': True, 'pool': True, 'history': False,
           'value': True, 'companies': True, 'flow': False}
    html = _subtabs('<div data-pane="overview">x</div>', has, tmp_path)
    assert html.count('role="tab"') == 5
    assert 'data-k="history"' not in html and 'data-k="flow"' not in html
    assert '<div class="pp-tabbed" data-tab="overview">' in html             # default tab
    assert 'class="pp-subtab active" aria-selected="true" data-k="overview"' in html
    # a remembered tab is used when the sector has it, else the first one
    assert 'data-tab="value"' in _subtabs('', has, tmp_path, stored='value')
    assert 'data-tab="overview"' in _subtabs('', has, tmp_path, stored='flow')
