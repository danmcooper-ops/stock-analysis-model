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
    assert order == ['pp-primer', 'pp-macro', 'pp-signals', 'pp-structure',
                     'pp-chart', 'pp-companies', 'pp-liquidity'], order


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
