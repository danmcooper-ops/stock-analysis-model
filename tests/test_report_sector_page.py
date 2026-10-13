# tests/test_report_sector_page.py
"""Guard: the per-sector page states each number once.

The page grew to ten stacked sections that had drifted into restating each
other — HHI/CR4 was quoted in three of them, the blended operating margin in
three, the top-3 profit-vs-revenue skew in two, and three separate card grids
listed largely the same companies. It is now a row of sub-tabs (Overview,
Drivers, Profit pool, History, Value creation, Trading), and the Profit pool
table is the one company view.

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
    assert order == ['pp-primer', 'pp-today', 'pp-signals', 'pp-structure',
                     'pp-chart', 'pp-history', 'pp-shifts', 'pp-econ',
                     'pp-liquidity'], order


def test_chart_sits_under_the_prose_that_describes_it():
    """The profit-pool chart used to sit four sections below the prose
    describing it. Structure, then the chart."""
    body = _assembly(_tpl())
    assert body.index('pp-section pp-structure') < body.index('pp-section pp-chart')


def test_no_company_card_grid():
    """The sector page had a company grid three times over (Company
    Highlights, CR4 Companies, Top 5 by Model Score), then one merged grid
    under a Key companies tab — and that one restated the Profit pool table.
    The table is the company view; no grid comes back."""
    css = _tpl()
    for gone in ('renderPoolKeyPlayers', 'renderPoolCR4Companies', 'renderPoolCompanies',
                 'pp-section pp-players', 'pp-section pp-cr4', 'pp-section pp-top',
                 'pp-section pp-companies', 'pp-co-list', "['companies',"):
        assert gone not in css, '%s came back' % gone
    assert 'div[style*="background:white"]' not in css


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
    assert [s for _, s in sections] == ['pp-primer', 'pp-today', 'pp-signals', 'pp-structure', 'pp-chart',
                                        'pp-history', 'pp-shifts', 'pp-econ', 'pp-liquidity']
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


def test_pool_table_sort_arrow_matches_the_gate_matrix():
    """Only the sorted column carries an arrow; no unsorted hint glyph,
    which iOS drew as an emoji box."""
    css = _tpl()
    assert "content:'\\2195'" not in css
    assert '.ppi-sort th[aria-sort="descending"]::after{content:\' \\25BC\\FE0E\';}' in css


def test_sector_sections_share_one_neutral_dark_surface():
    """The old Companies pane was dark green (and others navy, blue, teal),
    which read as a status colour."""
    css = _tpl()
    for sec in ('primer', 'today', 'structure', 'liquidity'):
        assert not re.search(r'\[data-theme="dark"\] \.pp-%s\{background:#(?!161b22)' % sec, css), sec
    assert ('[data-theme="dark"] .pp-primer,[data-theme="dark"] .pp-today,[data-theme="dark"] .pp-structure,'
            '[data-theme="dark"] .pp-liquidity,'
            '[data-theme="dark"] .pp-chart{background:#161b22;}') in css


def test_no_window_state_shadows_a_top_level_function():
    """A top-level `function f(){}` in the page script IS `window.f`, so
    `window.f=value` replaces the function. `_ppSubTabCur` kept its state
    under its own name: the stored sub-tab was never read (the first lookup
    returned the function), and after one click the next sector switch threw
    "_ppSubTabCur is not a function"."""
    css = _tpl()
    funcs = set(re.findall(r'^function ([A-Za-z_$][\w$]*)\(', css, re.M))
    written = set(re.findall(r'\bwindow\.([A-Za-z_$][\w$]*)\s*=(?!=)', css))
    assert not (funcs & written), sorted(funcs & written)


def test_pool_tables_pin_their_first_column():
    """Company / Industry stays put while the table scrolls sideways; the
    sticky cell carries the section's surface in both themes, so the
    columns sliding under it do not show through."""
    css = _tpl()
    assert ('.ppi-tbl th:first-child,.ppi-tbl td:first-child{position:sticky;left:0;z-index:1;'
            'background:var(--ppi-bg);') in css
    assert '.ppi-tbl{--ppi-bg:#f7f9fb;' in css                     # = .pp-chart, light
    assert '[data-theme="dark"] .ppi-tbl{--ppi-bg:#161b22;' in css  # = .pp-chart, dark


def test_gate_matrix_has_an_industry_column_after_sector():
    """Industry sits right of Sector in the Gate Matrix, sorts A-Z first like
    Sector, and is escaped ("Oil & Gas ...")."""
    css = _tpl()
    assert css.index("mSort(\\'sector\\')") < css.index("mSort(\\'industry\\')")
    assert css.index("+(d.sector||'N/A')+'</td>';\n  html+='<td style=\"font-size:0.75em\">'+esc(d.industry||'N/A')") > 0
    assert "col==='sector'||col==='industry'" in css


def test_clear_resets_the_dashboard_drilldown():
    """The drill-down counts in the Filters badge, so the panel's Clear must
    reset it; it used to survive, stranding the matrix on one industry."""
    css = _tpl()
    fn = re.search(r'^function clearFilters\(\)\{.*?^\}', css, re.S | re.M).group(0)
    assert '_dashFilter=null;' in fn and fn.index('_dashFilter=null;') < fn.index('applyFilters();')


def test_drilldown_chip_shows_on_phones_too():
    """Phones hide #f-count, so the chip also rides #sn-count beside the view
    title, in white on blue despite the chrome's --sac text colour."""
    css = _tpl()
    fn = re.search(r'^function _updateCountLabel\(\)\{.*?^\}', css, re.S | re.M).group(0)
    assert "el.innerHTML=h+chipHtml;" in fn
    assert "sn.innerHTML=esc('('+FD.length+' of '+DATA.length+')')+chipHtml;" in fn
    assert '.dd-chip{--sac:#fff;}' in css


def test_top_level_navigation_opens_the_first_sub_tab():
    """A sidebar row (or its number hotkey) carries no sub-tab key; it opens
    the section's first sub-tab, not wherever the section was last left."""
    css = _tpl()
    fn = re.search(r'^function navGo\(v,k\)\{.*?^\}', css, re.S | re.M).group(0)
    assert "if(k==null){" in fn and "k=g.items[0].k;" in fn
    assert fn.index('k=g.items[0].k;') < fn.index("if(v==='pool')")


def test_drilldowns_write_column_filters_not_a_chip():
    """The category cards, the score histogram and a single-industry click
    each set an ordinary column-filter row (visible and editable in the
    Filters menu) instead of the hidden drill-down shown as a blue chip.
    Only the gate pass/fail drill-down and the folded multi-industry Other,
    which no single column filter can express, keep the chip."""
    css = _tpl()
    def fn(name):
        return re.search(r'^function ' + name + r'\(.*?^\}', css, re.S | re.M).group(0)
    assert "_cfPut(scoreKey,avg,null,null);" in fn('dashCatJump')
    assert "_dashFilter={type:'cat'" not in css
    score = fn('toggleDashScore')
    assert "idx>=9?100:idx*10+9.9" in score and "_cfPut('_composite_score',lo,hi,null)" in score
    ind = fn('dashIndustryJump')
    assert "_cfPut('industry',null,null,mem[0]);" in ind
    assert "_dashFilter={type:'industry'" in ind          # the folded Other only
    # the scores are filterable columns
    assert "var _CF_SCORES=[['_composite_score','Composite Score']]" in css


def test_touch_swipe_moves_between_sub_tabs():
    """A flick on a view moves along the innermost tab row (a sector's
    Overview..Trading row, else #subnav), never over the ticker page, a
    chart, the drawer or a pinch-zoomed page, and lets a wide page or
    scroll wrap pan first."""
    css = _tpl()
    fn = re.search(r'^function _swipeTab\(dir\)\{.*?^\}', css, re.S | re.M).group(0)
    assert ".pp-subtabs" in fn and "_ppSubTab(nb)" in fn and "navGo(g.v,nx.k)" in fn
    block = css[css.index('function _swipeTab(dir)'):css.index('/* Velocity-based glide scroll')]
    # anywhere on the page, not just inside a .view (blank space, margins,
    # the title rows) — only the chrome, dialogs and drag-owning marks opt out
    assert "tg.closest('.view')" not in block
    for guard in (".header,#det-modal,#sidebar,.sb-scrim,.fpop-panel,#cfsel-panel,#ai-panel,#hk-overlay,#pf-overlay,svg,canvas,input,select,textarea,[ontouchstart]",
                  "['ai-panel','hk-overlay','pf-overlay'].some(",
                  "vv.scale>1.01", "sx<28||sx>vw-28", "x+vw<se.scrollWidth-1",
                  "Math.abs(dx)<70", "performance.now()-st>650"):
        assert guard in block, guard
    assert "{passive:true}" in block


def test_summary_key_tiles_show_the_52_week_range_not_next_earnings():
    """The ticker page's Summary grid carries the 52-week low – high (two
    decimals, like Price) with where the price sits in it; Next Earnings
    stays in Street View and the table, not in this grid."""
    css = _tpl()
    assert "_krows.push(['Next Earnings'" not in css
    assert "_krows.push(['52-Wk Range',_px(d.low_52w)+' – '+_px(d.high_52w)" in css
    assert "'52-Wk Range':'range_52w_position'" in css
    assert "['Next Earnings',_svEarn]" in css     # Street View keeps it


def test_macro_tab_intros_clamp_with_read_more():
    """Each Macro tab's opening paragraph shows three lines and a Read more /
    Show less toggle, which only appears when the text overflows."""
    css = _tpl()
    assert '<div class="rm"><p class="mac-sec-intro rm-txt">' in css
    assert ".rm:not(.open) .rm-txt{display:-webkit-box;-webkit-line-clamp:3;" in css
    assert ".rm.clip .rm-btn{display:inline-block;}" in css
    assert "  el.innerHTML=h;\n  _rmSync(el);" in css
    tog = re.search(r'^function _rmToggle\(btn\)\{.*?^\}', css, re.S | re.M).group(0)
    assert "'Show less':'Read more'" in tog and 'aria-expanded' in tog


def test_pool_history_axes_wear_their_series_colours():
    """The $ axis labels are the bars' blue and the % axis labels the line's
    orange, so the legend names the series by unit and mark, not 'left' and
    'right', which read as nonsense without an axis tie."""
    css = _tpl()
    assert 'Operating-profit pool (left)' not in css and 'operating margin (right)' not in css
    assert 'Operating-profit pool ($, bars)' in css and 'Net weighted operating margin (%, line)' in css
    assert '<text class="pph-tx pph-ax-l"' in css and '<text class="pph-tx pph-ax-r"' in css
    assert '[data-theme="dark"] .pph-ax-l{fill:#6e9bd1;}' in css      # = dark .pph-bar
    assert '[data-theme="dark"] .pph-ax-r{fill:#f0a35e;}' in css      # = dark .pph-line
