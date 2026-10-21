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
             '_pphNice', '_pphSvgs', 'renderPoolHistory', '_ppShiftRows', 'renderPoolShifts',
             '_ppIndCls', '_ppiV', '_ppiTh', 'renderPoolIndustries', '_ppEpRows', 'renderPoolEcon',
             '_ppStructureStats', '_qHelp', '_noteQ']
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
    # one chart: bars on the $ axis, the margin line on the % axis, with a key
    assert html.count('<svg class="pph-svg"') == 1
    assert 'Operating-profit pool <b>' in html and 'Net operating margin <b>' in html
    assert 'Where the growth came from' in html
    assert 'Cycle position' in html and 'Concentration trend' in html
    assert 'FY2018–20 with FY2023–25' in html
    d = h['decomposition']
    assert '%+.1f%%/yr' % (d['pool_cagr'] * 100) in html


def test_incomplete_year_is_a_footnote_not_a_bar(tmp_path):
    """A year most filers have not reported yet drew as a stub bar that read
    as a collapse; it is left out of both charts and named in a footnote."""
    rows = [_row('AAA', 100, 0.10, 0.12), _row('BBB', 80, 0.15, 0.0),
            _row('CCC', 60, 0.08, 0.03)]
    rows[0]['edgar_history']['revenue_history'][2026] = 200
    rows[0]['edgar_history']['operating_income_history'][2026] = 30
    h = sector_pool_history(rows)
    html = _render(h, "renderPoolHistory('Tech')", tmp_path)
    assert html.count('class="pph-bar') == sum(p['complete'] for p in h['points'])
    assert '>2026<' not in html and 'inc"' not in html
    assert 'FY2026 left out: only part of the sector has reported it' in html


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
    assert len(re.findall(r'<tr class="ppi-ind" data-k="\d+" onclick="dashIndustryJump\(', html)) == 2 and 'Pool CAGR' in html
    # every row and bar opens the Gate Matrix on its industry, by index
    assert html.count("onclick=\"dashIndustryJump('Tech',") == 4


def test_industry_drilldown_filters_by_sector_and_member_names(tmp_path):
    src = TEMPLATE.read_text(encoding='utf-8')
    js = '\n'.join([
        'var _dashFilter={type:"industry",sector:"Tech",label:"Other industries",members:["Odd","Other"]};',
        _fn(src, '_dashPass'), _fn(src, '_dashLabel'),
        'var rows=[{sector:"Tech",industry:"Odd"},{sector:"Tech"},{sector:"Tech",industry:"Semis"},'
        '{sector:"Energy",industry:"Odd"}];',
        'process.stdout.write(JSON.stringify([rows.map(_dashPass),_dashLabel()]));'])
    script = tmp_path / 'df.js'
    script.write_text(js, encoding='utf-8')
    out = json.loads(subprocess.run(['node', str(script)], capture_output=True, text=True,
                                    check=True).stdout)
    assert out == [[True, True, False, False], 'Tech \u203a Other industries']


def test_econ_states_both_market_shares_and_the_exclusions(tmp_path):
    rows = _step2_rows()
    uni = universe_totals(rows)
    e = economic_pool(rows, uni)
    html = _render(None, "renderPoolEcon('Tech')", tmp_path, {'economic': e})
    assert 'of US operating profit but' in html
    assert 'Largest value creators' in html and 'data-tk="AAA"' in html
    assert 'data-tk="BBA"' in html.split('Largest value destroyers')[1]
    assert 'Not measured' not in html                       # nothing excluded


_TILES = "_ppStructureStats(SECTOR_POOL.Tech.structure_stats,SECTOR_POOL.Tech.stats)"


def test_structure_stats_render_the_four_tiles(tmp_path):
    rows = _step2_rows()
    st = pool_structure(rows, universe_totals(rows))
    html = _render(None, _TILES, tmp_path,
                   {'structure_stats': st, 'stats': {'hhi': 0.035, 'cr4': 0.306}})
    for title in ('Concentration', 'Margin spread', 'Loss-makers', 'Price of the pool'):
        assert '<b>%s</b>' % title in html
    tile = html.split('<b>Concentration</b>')[1].split('</div>')[0]
    # the two top-4 sets are named apart, and each HHI says what it is on
    assert 'The top 4 by revenue hold 31% of sector revenue' in tile
    assert 'the top 4 by profit earn' in tile
    assert 'HHI 0.035 on revenue (competitive)' in tile and ' on profit.' in tile


def test_concentration_tile_stands_on_either_half(tmp_path):
    """structure_stats is built under a try and the revenue HHI/CR4 are null
    under three tickers: either half alone still draws the tile, and
    neither draws nothing at all."""
    rows = _step2_rows()
    st = pool_structure(rows, universe_totals(rows))
    rev_only = _render(None, _TILES, tmp_path, {'stats': {'hhi': 0.3, 'cr4': 0.8}})
    assert '<b>Concentration</b>The top 4 by revenue hold 80% of sector revenue. HHI 0.300 on revenue (concentrated).</div>' in rev_only
    prof_only = _render(None, _TILES, tmp_path, {'structure_stats': st, 'stats': {'hhi': None, 'cr4': None}})
    assert '<b>Concentration</b>The top 4 by profit earn' in prof_only and 'on revenue' not in prof_only
    no_pool = dict(st, profit_cr4=None, profit_hhi=None, lorenz=None)
    html = _render(None, _TILES, tmp_path, {'structure_stats': no_pool, 'stats': {'hhi': 0.2, 'cr4': 0.5}})
    assert 'HHI 0.200 on revenue (moderately concentrated).' in html and 'on profit' not in html
    assert _render(None, _TILES, tmp_path, {}) == ''


def _render_signals(entry, tmp_path, stored=None, data=None):
    src = TEMPLATE.read_text(encoding='utf-8')
    names = ['_ppFy', '_ppFy1', '_ppPct', '_ppSgnPct', '_ord', '_ppRankOf', '_ppWin', '_ppfSpark',
             '_ppfEvidence', '_ppfExpVal', '_ppfChips', '_ppfExposure', '_ppfBalance',
             '_ppfCard', '_ppfLive', '_windState', '_windGroup', '_ppfColSum', '_ppfCol', '_qHelp', '_windMatrixLi',
             '_noteQ', 'renderPoolSectorSignals']
    consts = '\n'.join(re.search(r'^var %s=.*$' % v, src, re.M).group(0)
                       for v in ('_PPF_ST', '_PPF_ORDER', '_PPF_TYPE'))
    store = ('var localStorage={getItem:function(){return %s;}};' % json.dumps(json.dumps(stored))
             if stored is not None else '')
    js = '\n'.join([store + "var RC={'BUY':'#1a9850','PASS':'#de2d26'};var DATA=" + json.dumps(data or []) + ";",
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
    tw, hw = html.split('>▼ Headwinds<')
    assert html.count('<div class="ppf wind-row ppf-') == 6
    assert 'ppf-st ppf-active"><i class="ppf-dot"></i>Acting now' in hw
    assert 'ppf-st ppf-dormant"><i class="ppf-dot"></i>Not acting' in tw
    assert 'Not measured' in html and 'Qualitative' not in html
    card = hw.split('class="ppf-th">Interest-rate sensitivity')[1].split('<div class="ppf wind-row ppf-')[0]
    assert '<ul class="ppf-ul"><li><b>Evidence:</b> Test series 5.38%' in card
    assert '<polyline' in card and '<b>Why it matters:</b> Cap rates' in card
    assert 'XLRE has trailed the market by 10.0% over 3 months' in html
    assert '▲ Tailwinds<span>· 0 of 3 acting</span>' in tw and '· 1 of 3 acting</span>' in hw
    assert 'class="wind-h wind-up"' in tw   # the report's one wind header
    assert 'Acting now: the evidence is strong and holding' in html       # the key
    # ...behind the section's one ?, with the method, not a band above the lists
    assert 'ppf-keyrow' not in html and html.count('class="qhelp-btn"') == 1
    note = html.split('<div class="note-q">')[1]
    assert 'class="qhelp-pop"' in note and 'Acting now' in note and 'Each status reads' in note
    assert html.index('Market check:') < html.index('<div class="ppf-cols">')


def test_nothing_is_hidden_behind_expand_or_collapse(tmp_path):
    from models.sector_forces import evaluate_forces
    from tests.test_sector_forces import _sidecar
    side = _sidecar('DGS10', [3.0 + 0.02 * k for k in range(120)])
    html = _render_signals({'forces': evaluate_forces('Real Estate', side, {})}, tmp_path)
    assert '<details' not in html and '<summary' not in html
    hw = html.split('>▼ Headwinds<')[1]
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
    card = html.split('class="ppf-th">Interest-rate sensitivity')[1].split('class="ppf wind-row ppf-')[0]
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
    card = html.split('class="ppf-th">Software dollar share keeps rising')[1].split('<div class="ppf wind-row ppf-')[0]
    assert card.count('Software - Infrastructure') == 1
    assert '<b>Who gains:</b> 3 companies' in card


def _render_co_table(shown, tail, tmp_path):
    src = TEMPLATE.read_text(encoding='utf-8')
    names = ['_ppPct', '_ppSgnPct', '_ppiV', '_ppiTh', '_ppSkewTd', '_qHelp', '_noteQ',
             'renderPoolCompanyTable']
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
    shown = [_co('A%02d' % k, 0.02, 0.02 + (0.001 * k), operating_margin=0.1 + 0.01 * k) for k in range(25)]
    tail = [_co('T%d' % k, 0.001, 0.0005) for k in range(10)]
    html = _render_co_table(shown, tail, tmp_path)
    assert html.count('<tr class="ppi-co"') == 25
    assert '<td>Others <span class="ppi-dim">(10 companies)</span>' in html
    assert html.index('A24') < html.index('A00')                  # by margin
    assert '>Pool share trend</th>' in html and '+3.0%/yr' in html
    assert html.index('>Pool share trend</th>') < html.index('>Value/pool</th>')   # last column
    assert 'data-tk="A00" onclick="openDet(this.dataset.tk)"' in html
    assert 'class="ppi-tbl ppi-sort"' in html


def test_company_table_highlights_and_blanks(tmp_path):
    shown = [_co('HI', 0.10, 0.125), _co('LO', 0.10, 0.075), _co('MID', 0.10, 0.1),
             _co('JPM', 0.10, 0.1, sector='Financial Services', industry='Banks - Diversified'),
             _co('NEW', 0.10, 0.1, _gate_pool_share=None)]
    html = _render_co_table(shown, [], tmp_path)
    row = lambda t: html.split('data-tk="%s"' % t)[1].split('</tr>')[0]   # noqa: E731
    assert 'ppi-hi' in row('HI') and 'ppi-lo' in row('LO') and 'ppi-' not in row('MID').replace('ppi-dim', '').replace('ppi-rt', '')
    assert row('JPM').count('<td data-v="">—</td>') == 2                # spread, and no mcap
    assert row('NEW').endswith('<td data-v="">—</td><td data-v="">—</td>')   # trend, value/pool
    assert '<td>Others' not in html and 'pooled as Others' not in html


def test_company_table_value_over_pool(tmp_path):
    """Share of the sector's market value over share of the pool, over the
    same rows the cards count (shown + tail, mcap > 0); the Others row
    carries the tail's combined figure."""
    shown = [_co('A', 0.5, 0.6, mcap=60.0), _co('B', 0.3, 0.2, mcap=30.0),
             _co('L', 0.1, 0.0, mcap=5.0)]                     # loss-maker: no pool
    tail = [_co('T', 0.1, 0.2, mcap=5.0), _co('U', 0.0, 0.0)]  # U has no mcap
    html = _render_co_table(shown, tail, tmp_path)
    row = lambda t: html.split('data-tk="%s"' % t)[1].split('</tr>')[0]   # noqa: E731
    assert row('A').endswith('<td data-v="1">1.00\u00d7</td>')      # 60% of value / 60% of pool
    assert row('B').endswith('1.50\u00d7</td>')
    assert row('L').endswith('<td data-v="">\u2014</td>')
    others = html.split('<tr class="ppi-pin" data-k="__others">')[1].split('</tr>')[0]
    assert others.endswith('<td>0.25\u00d7</td>')                     # 5% of value / 20% of pool


def test_drivers_use_the_reports_wind_group_and_each_heading_toggles_its_force(tmp_path):
    """Each Drivers kind is the report's one wind group (_windGroup): open
    unless this browser folded it, with a one-sentence summary for when it
    is; every force's heading is a button for its own bullets, closed."""
    from models.sector_forces import evaluate_forces
    from tests.test_sector_forces import _sidecar
    side = _sidecar('DGS10', [3.0 + 0.02 * k for k in range(120)])
    forces = {'forces': evaluate_forces('Real Estate', side, {})}
    html = _render_signals(forces, tmp_path)
    assert html.count('onclick="_windToggle(this)"') == 2
    assert '<div class="wind-grp wind-shut" data-wk="drivers:tailwind">' in html
    assert '<div class="wind-grp wind-shut" data-wk="drivers:headwind">' in html
    assert '<div class="wind-sum">None of its 3 tailwinds is acting.</div>' in html
    assert 'Interest-rate sensitivity and tenant-credit risk is acting now.</div>' in html
    heads = html.count('class="ppf-hdr" role="button"')
    assert heads == html.count('<ul class="ppf-ul">') == 6
    assert html.count('aria-expanded="false" onclick="_ppfToggleOne(this)"') == 6
    opened = _render_signals(forces, tmp_path, stored={'drivers:tailwind': False})
    assert '<div class="wind-grp wind-shut" data-wk="drivers:headwind">' in opened
    assert '<div class="wind-grp" data-wk="drivers:tailwind">' in opened


def test_wind_group_and_force_toggles():
    css = TEMPLATE.read_text(encoding='utf-8')
    assert '.ppf:not(.ppf-open) .ppf-ul,.ppf:not(.ppf-open) .ppf-meta{display:none;}' in css
    assert '.wind-shut .wind-body{display:none;}' in css and '.wind-shut .wind-sum{display:block;}' in css
    one = _fn(css, '_ppfToggleOne')
    assert "classList.toggle('ppf-open')" in one
    tog = _fn(css, '_windToggle')
    assert "localStorage.setItem('stock_wind_shut_v1'" in tog
    # the ticker page uses the same group
    assert "_windGroup({scope:'ticker',k:'tailwind'" in css and "_windGroup({scope:'ticker',k:'headwind'" in css


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
    assert html.count('onclick="_ppiSort(this)"') == 8
    assert '<th data-t="n" aria-sort="descending" onclick' in html            # Pool, as rendered
    assert '<th data-t="s" onclick' in html                                    # Company sorts as text
    assert '<tr class="ppi-pin" data-k="__others"><td>Others' in html
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
                    'var t=_ppSubTabs(%s,%s);process.stdout.write(t.bar+"|BODY|"+t.body);'
                    % (json.dumps(body), json.dumps(has))])
    script = tmp_path / 'tabs.js'
    script.write_text(js, encoding='utf-8')
    return subprocess.run(['node', str(script)], capture_output=True, text=True,
                          check=True).stdout


def test_sub_tabs_skip_empty_groups_and_fall_back(tmp_path):
    has = {'overview': True, 'forces': True, 'pool': True, 'history': False,
           'value': True, 'flow': False}
    html = _subtabs('<div data-pane="overview">x</div>', has, tmp_path)
    assert html.count('role="tab"') == 4
    assert 'data-k="history"' not in html and 'data-k="flow"' not in html
    assert '<div class="pp-tabbed" data-tab="overview">' in html             # default tab
    assert 'class="d-stmt-tab cur" aria-selected="true" data-k="overview"' in html
    bar, body = html.split('|BODY|')
    # the row reuses the ticker page's statement tabs, and sits apart from the body
    assert bar.startswith('<div class="d-stmt-tabs sa-chrome pp-subtabs" role="tablist"')
    assert body.startswith('<div class="pp-tabbed"') and 'role="tab"' not in body
    # a remembered tab is used when the sector has it, else the first one
    assert 'data-tab="value"' in _subtabs('', has, tmp_path, stored='value')
    assert 'data-tab="overview"' in _subtabs('', has, tmp_path, stored='flow')


def test_company_table_shows_tickers_only(tmp_path):
    html = _render_co_table([_co('BHP', 0.04, 0.1, company_name='BHP Group Limited')], [], tmp_path)
    assert '<td data-v="BHP">BHP</td>' in html
    assert 'title="BHP Group Limited"' in html and '>BHP Group Limited<' not in html


def test_both_charts_use_round_ticks_and_label_their_latest_value(tmp_path):
    """One chart, two axes on shared gridlines: the $ pool ticks step
    1/2/2.5/5 x 10^k, and the % margin axis takes the same number of round
    steps, from the complete years only (one early-filer outlier cannot
    stretch it). The latest values sit in the key."""
    hist = {'n_rows': 3, 'n_with_history': 3, 'points': [
        {'year': 2022 + i, 'pool': pool, 'margin': m, 'n': 3, 'coverage': 1.0, 'complete': c}
        for i, (pool, m, c) in enumerate([(47e9, 0.145, True), (99e9, 0.204, True),
                                          (127e9, 0.157, True), (2e9, 0.45, False)])]}
    html = _render(hist, "_pphSvgs(SECTOR_POOL.Tech.history,375)", tmp_path)
    (svg,) = re.findall(r'<svg class="pph-svg".*?</svg>', html, re.S)
    assert 'width="375"' in svg                              # drawn at the real width
    ticks = re.findall(r'class="pph-tx pph-ax-pool"[^>]*>([^<]+)<', svg)
    assert ticks == ['0', '$50B', '$100B', '$150B']
    pct = [float(t) for t in re.findall(r'class="pph-tx pph-ax-m"[^>]*>(-?[\d.]+)%<', svg)]
    assert len(pct) == len(ticks)                            # one % label per gridline
    steps = {round(b - a, 6) for a, b in zip(pct, pct[1:], strict=False)}
    assert len(steps) == 1 and pct[0] <= 14.5 and pct[-1] >= 20.4 and pct[-1] < 45
    assert '<b>$127B</b>' in html and '<b>15.7%</b>' in html
    d = re.search(r'<path class="pph-line" d="([^"]*)"', svg).group(1)
    assert d.count('M') == 1 and d.count('L') == 2

def _render_outlook(pool, tmp_path, macro=None, stored=None, data=None):
    src = TEMPLATE.read_text(encoding='utf-8')
    consts = re.search(r'^var _SO_SUM=.*$', src, re.M).group(0)
    names = ['_soDominant', '_soHealth', '_soNarr', '_qHelp', '_windState', '_windGroup', '_windRow',
             '_ppfBalance', '_windTop', '_windMatrixLi', '_soGroupSummary', 'renderSectorOutlook']
    js = '\n'.join(['function _esc(s){return String(s);}function _attr(s){return String(s);}',
                    'var MACRO_SUM=' + json.dumps(macro) + ';var MACRO=null;var DATA=' + json.dumps(data or []) + ';',
                    'var localStorage={getItem:function(){return %s;}};' % json.dumps(json.dumps(stored)),
                    re.search(r'^var _SO_W=.*$', src, re.M).group(0), consts]
                   + [_fn(src, n) for n in names]
                   + ['var SECTOR_POOL=' + json.dumps(pool) + ';',
                      'process.stdout.write(renderSectorOutlook());'])
    script = tmp_path / 'o.js'
    script.write_text(js, encoding='utf-8')
    return subprocess.run(['node', str(script)], capture_output=True, text=True,
                          check=True).stdout


def test_sector_outlook_names_the_dominant_force_each_way(tmp_path):
    def f(kind, theme, status, reach):
        return {'kind': kind, 'theme': theme, 'status': status,
                'exposure': None if reach is None else {'reach': reach}}
    pool = {
        'Energy': {'forces': {'balance': {'tailwind': 0.48, 'headwind': 0.0, 'net': 0.48},
                              'forces': [f('tailwind', 'Underinvestment', 'active', 0.48),
                                         f('tailwind', 'Discipline', 'easing', 0.9),
                                         f('headwind', 'Commodity cycle', 'dormant', 0.48)]}},
        'Utilities': {'forces': {'balance': {'tailwind': 0.33, 'headwind': 0.29, 'net': 0.04},
                                 'forces': [f('tailwind', 'Defensive bid', 'active', None),
                                            f('tailwind', 'Grid capex', 'building', 0.6),
                                            f('headwind', 'Rate cases', 'active', 0.29)]}},
        'Healthcare': {'forces': {'balance': {'tailwind': 0, 'headwind': 0, 'net': 0},
                                  'forces': [f('tailwind', 'Aging', 'qualitative', None)]}},
        'Bare': {'stats': {}},
    }
    html = _render_outlook(pool, tmp_path)
    assert html.count('<div class="ppf wind-row ') == 3 and 'Bare' not in html
    assert '>Broad Sector Outlook<' in html
    # the Drivers' verdict and bar, weighing sectors
    assert 'Across 3 sectors, tailwinds lead in 1 and headwinds in 0.' in html
    energy = html.split('class="ppf-th">Energy')[1].split('<div class="ppf wind-row')[0]
    assert '<span class="ppf-st ppf-active"><i class="ppf-dot"></i>Tailwinds lead</span>' in html.split('class="ppf-th">Energy')[0][-200:]
    assert '<li><b>Tailwind:</b> Underinvestment.</li>' in energy
    assert '<li><b>Headwind:</b> none acting.</li>' in energy
    assert '1 of 3 forces acting' in energy
    util = html.split('class="ppf-th">Utilities')[1].split('<div class="ppf wind-row')[0]
    # building at half weight x 0.6 reach (0.30) loses to active x 1/3 (0.33)
    assert '<b>Tailwind:</b> Defensive bid.' in util and '<b>Headwind:</b> Rate cases.' in util
    assert 'Grid capex' not in util
    assert 'goSectorDrivers' in util and 'data-sec="Utilities"' in util
    assert 'No force acting</span>' in html
    assert _render_outlook({'X': {}}, tmp_path) == ''
    assert 'data-wk="outlook:all"' in html, 'no narrative: one ungrouped group'


def test_sector_outlook_groups_by_the_macro_stance(tmp_path):
    forces = {'balance': {'tailwind': 0, 'headwind': 0, 'net': 0},
              'forces': [{'kind': 'tailwind', 'theme': 'T', 'status': 'dormant'}]}
    pool = {s: {'forces': forces} for s in ('Energy', 'Healthcare', 'Utilities', 'Technology')}
    macro = {'narrative': {'sectors': [
        {'sector': 'Healthcare', 'stance': 'tailwind', 'headline': 'H', 'influence': 'Inf'},
        {'sector': 'Utilities', 'stance': 'headwind', 'influence': 'U'},
        {'sector': 'Energy', 'stance': '<script>'}]}}
    html = _render_outlook(pool, tmp_path, macro)
    assert html.count('aria-expanded="false" onclick="_windToggle(this)"') == 3
    assert 'class="wind-h wind-up"' in html and 'class="wind-h wind-down"' in html
    assert 'class="wind-h" ' in html
    assert html.count('wind-shut') == 3, 'groups start folded'
    opened = _render_outlook(pool, tmp_path, macro, stored={'outlook:down': False})
    # each group carries its one-sentence summary (CSS shows it only folded)
    assert '<div class="wind-sum">No force is acting in Healthcare.</div>' in html
    assert '<div class="wind-sum">No force is acting in Energy and Technology.</div>' in html
    assert opened.count('wind-shut') == 2 and '<div class="wind-grp" data-wk="outlook:down">' in opened
    up, rest = html.split('▲ Macro tailwind<span>· 1</span>')[1].split('Macro neutral<span>· 2</span>')
    flat, down = rest.split('▼ Macro headwind<span>· 1</span>')
    assert 'Healthcare' in up and '<li><b>Macro read:</b> <i>H.</i> Inf</li>' in up
    assert 'class="ppf-th">Energy' in flat and 'class="ppf-th">Technology' in flat   # unknown stance, absent
    assert 'Macro read' not in flat, 'no influence, no Macro read'
    assert 'Utilities' in down and '<b>Macro read:</b> U</li>' in down
    assert '<script>' not in html
    assert 'class="qhelp-btn"' in html and 'the Macro Outlook’s read on it' in html, \
        'the explainer folds behind the ?'


def test_pool_bars_mirror_their_table():
    """Both pool charts are drawn in their table's order (margin first,
    the pooled row last) and every re-sort lays the bars out again in the
    new row order: bars and rows share a key (data-k / data-tk)."""
    src = TEMPLATE.read_text(encoding='utf-8')
    assert "return (a.others?1:0)-(b.others?1:0)||b.margin-a.margin;" in src
    ind = _fn(src, 'renderPoolIndustries')
    assert "(a.folded?1:0)-(b.folded?1:0)||b.margin-a.margin" in ind
    assert 'data-k="\'+ii+\'" data-w=' in ind and '<tr class="ppi-ind\'+(d.folded?\' ppi-pin\':\'\')+\'" data-k="\'+ii+\'"' in ind
    assert "_ppBarsFollow(th.closest('table'));" in _fn(src, '_ppiSort')
    follow = _fn(src, '_ppBarsFollow')
    assert "row.getAttribute('data-k')||row.getAttribute('data-tk')" in follow
    table = _fn(src, 'renderPoolCompanyTable')
    assert "return (b.operating_margin||0)-(a.operating_margin||0);" in table
    assert "_ppiTh('Pool')+_ppiTh('Margin','n',1)" in table


def test_industry_pool_table_opens_on_margin():
    """The Industries table opens ranked by margin, with the folded Other
    last and each row still addressed by its index."""
    src = TEMPLATE.read_text(encoding='utf-8')
    ind = _fn(src, 'renderPoolIndustries')
    assert "_ppiTh('Pool')+_ppiTh('Margin','n',1)" in ind
    assert "var x=a[0].margin,y=b[0].margin;" in ind
    assert "var pa=a[0].folded?1:0,pb=b[0].folded?1:0" in ind
    assert "var d=p[0],ii=p[1];" in ind and "dashIndustryJump(" in ind


def test_wind_rows_drill_to_the_gate_matrix(tmp_path):
    """A Drivers force and an outlook sector each carry a link into the
    Gate Matrix filtered to who it is about (the industry a force names,
    else the sector), made as a drill so the matrix offers the way back."""
    from models.sector_forces import evaluate_forces
    from tests.test_sector_forces import _entry
    from tests.test_sector_pool import YEARS, _flat
    rows = [dict(_flat(t, YEARS, 100 + i, 0.2, 0.08), industry='Software - Infrastructure', sector='Technology')
            for i, t in enumerate(('MSFT', 'ORCL', 'X'))]
    data = [{'ticker': r['ticker'], 'sector': 'Technology', 'industry': r['industry']} for r in rows] \
        + [{'ticker': 'AAPL', 'sector': 'Technology', 'industry': 'Consumer Electronics'}]
    src = TEMPLATE.read_text(encoding='utf-8')
    # renderPoolSectorSignals is rendered for 'Real Estate' by the harness; use the card directly
    html = _render_signals({'forces': evaluate_forces('Technology', None, _entry(0.14), rows)}, tmp_path, data=data)
    assert 'windMatrixJump' not in html, 'no companies in Real Estate, no link'
    card_src = _fn(src, '_ppfCard')
    assert "_windMatrixLi(sec+' \\u00b7 Drivers',sec,f.exposure&&f.exposure.basis==='industry'?f.exposure.industry:'')" in card_src
    li = _fn(src, '_windMatrixLi')
    assert "windMatrixJump(this.dataset.from,this.dataset.sec,this.dataset.ind)" in li
    jump = _fn(src, 'windMatrixJump')
    assert '_drillPush(from);' in jump and "_cfPut('industry',null,null,ind||null);" in jump and '_dashJump();' in jump
    forces = {'balance': {'tailwind': 0.5, 'headwind': 0, 'net': 0.5},
              'forces': [{'kind': 'tailwind', 'theme': 'T', 'status': 'active', 'exposure': {'reach': 0.5}}]}
    out = _render_outlook({'Technology': {'forces': forces}}, tmp_path, data=data)
    assert '>Research more ›</button></li>' in out
    assert 'title="Open the Gate Matrix filtered to the 4 Technology companies"' in out
    assert 'data-from="Broad Sector Outlook" data-sec="Technology" data-ind=""' in out


def test_stacked_wind_rows_indent_the_heading_under_the_status():
    """On phones a wind row stacks; its heading and bullets step in to the
    status word (after the 9px dot and 6px gap) in every section, which
    all share the .ppf row."""
    src = TEMPLATE.read_text(encoding='utf-8')
    assert ('@media(max-width:520px){.ppf-hdr{grid-template-columns:minmax(0,1fr);}'
            '.ppf-th{padding-left:15px;}.ppf-ul{margin-left:15px;}}') in src
