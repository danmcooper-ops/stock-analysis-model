# tests/test_report_sector_page.py
"""Guard: the per-sector page reads as four arcs, and states each number once.

The page grew to ten stacked sections that had drifted into restating each
other — HHI/CR4 was quoted in three of them, the blended operating margin in
three, the top-3 profit-vs-revenue skew in two, and three separate card grids
listed largely the same companies. It was reorganised into:

    A. Context   — primer, macro read, the sector's own structural forces
    B. The Pool  — the merged structure prose, then the chart of it
    C. Owners    — one company grid, annotating the chart above it
    D. Flow      — liquidity, trimmed to five bullets

Each rule below is one line in a 10k-line template or a few lines of prose
generation, and none of them fails loudly when undone: the page still renders,
it just goes back to saying the same thing three times. So pin them.
"""
import os
import re

_TEMPLATE = os.path.join(os.path.dirname(__file__), '..', 'templates',
                         'report.html')


def _tpl():
    return open(_TEMPLATE, encoding='utf-8').read()


def _assembly(css):
    """The per-sector section assembly inside renderPool."""
    m = re.search(r'var primerHtml=renderPoolPrimer\(sec\);.*?'
                  r'pp-section pp-liquidity[^\n]*\n', css, re.S)
    assert m, 'could not find the per-sector section assembly'
    return m.group(0)


def test_sections_render_in_arc_order():
    body = _assembly(_tpl())
    order = [m.group(1) for m in
             re.finditer(r'pp-section (pp-[a-z]+)"><span class="pp-section-label"',
                         body)]
    assert order == ['pp-primer', 'pp-signals', 'pp-structure',
                     'pp-chart', 'pp-history', 'pp-shifts', 'pp-econ',
                     'pp-companies', 'pp-liquidity'], order


def test_chart_sits_under_the_prose_that_describes_it():
    """The profit-pool chart used to sit four sections below the prose
    describing it. Structure -> chart -> the companies in that chart."""
    body = _assembly(_tpl())
    assert (body.index('pp-section pp-structure')
            < body.index('pp-section pp-chart')
            < body.index('pp-section pp-companies'))


def test_the_three_company_grids_stayed_merged():
    """Company Highlights, CR4 Companies and Top 5 by Model Score were one
    grid each; a sector's mega-cap appeared in all three at once. They are
    one grid now, and a company accumulates a badge per reason it is there."""
    css = _tpl()
    for gone in ('renderPoolKeyPlayers', 'renderPoolCR4Companies',
                 'pp-section pp-players', 'pp-section pp-cr4',
                 'pp-section pp-top'):
        assert gone not in css, '%s came back as a separate grid' % gone
    assert css.count('function renderPoolCompanies(') == 1
    assert css.count('class="pp-co-grid"') == 1, 'one grid, one card builder'


def test_company_cards_are_classed_not_inline_styled():
    """Three copies of the card markup were inlined, which forced dark mode
    to match on `[style*="background:white"]` — including the ` white` form
    the browser rewrites to on hover. Real classes, real overrides."""
    css = _tpl()
    assert 'div[style*="background:white"]' not in css, \
        'the attribute-selector dark-mode override is back'
    fn = re.search(r'function renderPoolCompanies\(sec,cos\).*?\n\}\n', css, re.S)
    assert fn, 'could not find renderPoolCompanies'
    assert 'background:white' not in fn.group(0), \
        'company cards are .pp-co, styled by class'
    assert '[data-theme="dark"] .pp-co{' in css
    for cls in ('.pp-co{', '.pp-co-badge{', '.pp-co-chip{', '.pp-co-note{'):
        assert cls in css, '%s missing' % cls


def test_liquidity_walks_the_same_rows_as_the_rest_of_the_page():
    """_ppLiquidityCache used to scan DATA unfiltered, bucketing on
    `d.sector||'Unknown'` with no gate, so its ticker count and its CR4 were
    measured over a wider universe than the header band, chart and prose on
    the same page — and could contradict them."""
    css = _tpl()
    fn = re.search(r'function _ppLiquidityCache\(\).*?\n\}\n', css, re.S)
    assert fn, 'could not find _ppLiquidityCache'
    fn = fn.group(0)
    assert "d.sector||'Unknown'" not in fn, 'the ungated bucket is back'
    assert 'if(!d.sector||d.pp_revenue_share==null)return;' in fn
    assert '_rev==null||_oi==null||_rev<=0' in fn
    # and the revenue-side CR4 is the sector's canonical one, not a second
    # figure derived here
    liq = re.search(r'function renderPoolLiquidityInsights\(sec\).*?\n\}\n',
                    css, re.S)
    assert liq and 's.cr4Rev' in liq.group(0), \
        'the flow-vs-revenue bullet quotes pp_sector_cr4'


def test_liquidity_is_five_bullets():
    """Nine bullets, four of which restated the company grid or each other."""
    css = _tpl()
    liq = re.search(r'function renderPoolLiquidityInsights\(sec\).*?\n\}\n',
                    css, re.S)
    assert liq
    liq = liq.group(0)
    assert liq.count('bullets.push(') == 5
    for gone in ('Profit extraction vs flow', 'Model conviction × flow',
                 'Margin × flow correlation'):
        assert gone not in liq, '%s duplicated the company grid' % gone


def test_retired_sector_code_stays_retired():
    """~540 lines that nothing reached: the KPI banner and its config, the
    stat banner, and a scatter chart whose container no element ever emitted
    (its sync function also re-registered a window resize listener on every
    render, one per sector switch)."""
    css = _tpl()
    for dead in ('SECTOR_KPI_BANNER', '_renderSectorKpiBanner',
                 '_renderSectorBanner', '_buildAllSectorStats', '_poolSecTint',
                 '_buildScatterSVG', '_syncScatter', '_scatterTip',
                 'xsect-scatter-wrap', 'pool-stat-banner', '.pp-kpis',
                 '_allSecStats', 'pool-sector-table'):
        assert dead not in css, '%s came back' % dead


def test_flow_vs_profit_bullet_compares_positive_pools():
    """The flow-vs-economic-weight bullet divided the sector's NET operating
    income by the universe's POSITIVE-only pool, so a sector with loss-makers
    read as a smaller share of US profit than it is."""
    css = _tpl()
    fn = re.search(r'function _ppLiquidityCache\(\).*?\n\}\n', css, re.S).group(0)
    assert 'bySec[sec].oiPos+=d.operating_income;allOIPos+=d.operating_income;' in fn
    liq = re.search(r'function renderPoolLiquidityInsights\(sec\).*?\n\}\n',
                    css, re.S).group(0)
    assert 'var poolShare=s.oiPos/cache.allOIPos*100;' in liq
    assert 's.oi/cache.allOIPos' not in liq


def test_history_and_shifts_render_from_sector_pool_only():
    """Both sections read SECTOR_POOL[sec].history (built server-side by
    models/sector_pool.py); neither recomputes the pool from DATA."""
    css = _tpl()
    for name in ('renderPoolHistory', 'renderPoolShifts', 'renderPoolIndustries',
                 'renderPoolEcon'):
        fn = re.search(r'function ' + name + r'\(sec\).*?\n\}\n', css, re.S)
        assert fn, name
        assert 'DATA' not in fn.group(0), '%s reads DATA' % name
        assert 'SECTOR_POOL' in fn.group(0)


def test_both_pool_chart_views_carry_a_table():
    """The Companies view had only its chart; it now has the same table as
    the Industries view."""
    css = _tpl()
    assert 'renderPoolCompanyTable(shownCos,tailCos)' in css
    for name in ('renderPoolCompanyTable', 'renderPoolIndustries'):
        fn = re.search(r'function ' + name + r'\(.*?\n\}\n', css, re.S)
        assert fn and 'class="ppi-tbl ppi-sort"' in fn.group(0), name


def test_every_section_belongs_to_a_sub_tab_with_a_hide_rule():
    """The sector body is split into sub-tabs: each section carries a
    data-pane, and every pane name has the CSS rule that hides it when
    another tab is selected."""
    css = _tpl()
    body = _assembly(css)
    sections = re.findall(r'<div data-pane="([a-z]+)" class="pp-section (pp-[a-z]+)"', body)
    assert [s for _, s in sections] == ['pp-primer', 'pp-signals', 'pp-structure', 'pp-chart',
                                        'pp-history', 'pp-shifts', 'pp-econ', 'pp-companies',
                                        'pp-liquidity']
    tabs = re.search(r'^var _PP_SUBTABS=(.*);$', css, re.M).group(1)
    for pane in {p for p, _ in sections}:
        assert "'%s'" % pane in tabs, pane
        assert ('.pp-tabbed:not([data-tab="%s"])>[data-pane="%s"]{display:none;}' % (pane, pane)) in css


def test_sub_tabs_sit_above_the_sector_card():
    """Main tabs, then sub-tabs, stacked like the ticker page's section and
    statement tabs: the bar is spliced in before the sector's card."""
    css = _tpl()
    assert ("chartHtml=chartHtml.slice(0,_secStart)+_tabs.bar+chartHtml.slice(_secStart,_bodyStart)+_tabs.body;"
            in css)
    assert css.index("var _secStart=chartHtml.length;") < css.index("<div class=\"pool-sector\" id=\"pool-sec-")


def test_the_sector_banner_is_gone():
    """The sector tab names the sector and the structure section quotes its
    count, HHI and CR4; the banner repeated them."""
    css = _tpl()
    for gone in ('pool-sec-hdr', 'pool-sec-stats', 'pool-sec-name'):
        assert gone not in css, gone
