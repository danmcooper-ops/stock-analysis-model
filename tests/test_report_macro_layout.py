# tests/test_report_macro_layout.py
"""Guard: the Macro Outlook tab's place in the nav and its phone layout.

Two things about this tab are invisible on a desktop browser and so are easy
to undo by accident:

  * it leads the nav menu (leftmost column on desktop, first block in the
    stacked phone menu) because it is the top-down context for everything the
    other views say about individual names. It is the only group appended
    conditionally, so the natural edit — `groups.push` — silently sends it
    back to last place;
  * its Overview is a story beside a rail of glance tiles and nothing else —
    the full histories are drawn once, on the section sub-tabs, never
    repeated here, and the per-sector read is the narrative's Key sector
    influences — not a grid trailing the tiles, and not repeated on the
    Sector Analysis tabs;
  * its charts are `<svg viewBox>` with no fixed height, so their height is a
    function of the CARD's width. That makes the card/tile grid floors, not
    any height rule, what decides whether a chart fits on an iPhone screen.
    At the desktop 460px floor a landscape phone got ONE full-bleed card per
    row whose chart drew 295px tall — taller than the 393px viewport.

Both are one-line rules in a 9k-line template; pin them.
"""
import os
import re

_TEMPLATE = os.path.join(os.path.dirname(__file__), '..', 'templates',
                         'report.html')


def _css():
    return open(_TEMPLATE, encoding='utf-8').read()


def _media_block(header):
    """Every block written under `header`, joined. The sheet opens the same
    two phone queries several times over (the popup, the chrome compaction,
    the macro tab), so a single match would read the wrong one; braces are
    matched by hand so nested blocks come back intact."""
    text, out, at = _css(), [], 0
    while True:
        at = text.find(header, at)
        if at < 0:
            return '\n'.join(out)
        start = at + len(header)
        depth, i = 1, start
        while depth:
            depth += {'{': 1, '}': -1}.get(text[i], 0)
            i += 1
        out.append(text[start:i - 1])
        at = i


def _floors(block, selector):
    """Every grid-template-columns minmax() floor written for `selector`."""
    out = []
    for rule in re.findall(re.escape(selector) + r'\{([^}]*)\}', block):
        out += [int(px) for px in
                re.findall(r'minmax\((\d+)px', rule)]
    return out


_PORTRAIT = '@media(max-width:900px) and (orientation:portrait){'
_LANDSCAPE = '@media(orientation:landscape) and (max-height:500px){'


def test_macro_outlook_leads_the_nav_menu():
    css = _css()
    assert "groups.unshift({v:'macro'" in css, \
        'Macro Outlook must lead the nav groups, not trail them'
    assert "groups.push({v:'macro'" not in css


def test_macro_cards_drop_their_grid_floor_on_a_landscape_phone():
    desktop = _floors(_css(), '.mac-cards')
    assert 460 in desktop, 'expected the desktop .mac-cards floor to survive'
    landscape = _floors(_media_block(_LANDSCAPE), '.mac-cards')
    assert landscape, 'landscape phones need their own .mac-cards floor'
    assert max(landscape) <= 320, \
        'a floor above ~320px puts one full-width chart per landscape row'


def test_macro_tiles_go_two_column_on_a_portrait_phone():
    floors = _floors(_media_block(_PORTRAIT), '.mac-tiles')
    assert floors and max(floors) <= 190, \
        'the Overview tiles need a floor that fits two columns at 393px'


def test_every_macro_chart_is_scrubbable():
    """All three chart shapes — the section/overview line charts, the tile
    sparklines and the yield curve — hand the pointer to the same handlers.
    A new chart added without them would read as broken next to the others."""
    css = _css()
    for fn in ('_macChartSVG', '_mSparkSVG', '_macCurveCard'):
        body = re.search(r'function ' + fn + r'\(.*?\n\}\n', css, re.S)
        assert body, 'could not find %s' % fn
        for handler in ('onmousemove="macHovMove', 'onmouseleave="macHovLeave',
                        'ontouchmove="macHovTouch', 'ontouchend="macHovLeave'):
            assert handler in body.group(0), '%s is missing %s' % (fn, handler)


def test_scrub_registry_is_cleared_per_render():
    """renderMacro replaces the whole subtree; keeping the previous render's
    entries would leak a chart per visit and let a stale id win a lookup."""
    body = re.search(r'function renderMacro\(\).*?\n\}\n', _css(), re.S)
    assert body and '_MAC_HOV={}' in body.group(0)
    assert '_MAC_READ={}' in body.group(0)


def test_touch_scrub_does_not_swallow_the_gesture():
    """The macro tab stacks charts the full height of a phone screen, so the
    touch handler must not preventDefault — that would trap the page scroll
    (the sector chart can afford it; one chart does not fill the view)."""
    body = re.search(r'function macHovTouch\(.*?\n\}\n', _css(), re.S)
    assert body and 'preventDefault' not in body.group(0)


def test_out_of_grid_macro_cards_are_capped_in_landscape():
    """The yield-curve and OAS cards are direct children of #macro-view, so
    the grid floor above does not reach them — they need their own cap or the
    curve alone fills a landscape viewport."""
    rules = re.findall(r'#macro-view>\.mac-card\{([^}]*)\}',
                       _media_block(_LANDSCAPE))
    assert rules and any('max-width' in r for r in rules)


def test_macro_overview_reads_top_down():
    """The Overview reads as one document: the regime card carries the
    bottom line, then the key numbers, then the narrative (the economy, what
    to watch, the sectors). The tiles come first so the story's figures are
    seen before they are argued, and the bottom line renders only on the
    Overview, inside the regime card."""
    css = _css()
    body = re.search(r'function _macOverviewHTML\(\).*?\n\}\n', css, re.S).group(0)
    tiles_at, nar_at = body.find('mac-tiles'), body.find('_macNarrativeHTML()')
    assert 0 <= tiles_at < nar_at, 'key numbers lead, the narrative follows'
    assert 'Key numbers' in body
    reg = re.search(r'function _macRegimeHTML\(full\).*?\n\}\n', css, re.S).group(0)
    assert "if(full&&_nar&&_nar.bottom_line)" in reg
    assert '_esc(_nar.bottom_line)' in reg
    nar = re.search(r'function _macNarrativeHTML\(\).*?\n\}\n', css, re.S).group(0)
    order = [nar.find(x) for x in ('>The economy<', '>What to watch<', '>Sectors<', 'min read')]
    assert all(i >= 0 for i in order) and order == sorted(order)


def test_macro_narrative_escapes_every_model_string():
    """Narrative strings are Claude output, not repo-authored markup: every
    interpolation of them must pass through _esc(), and the stance value may
    only reach a class attribute through the whitelist lookup."""
    css = _css()
    m = re.search(r'function _macNarrativeHTML\(\).*?\n\}\n', css, re.S)
    assert m, 'could not find _macNarrativeHTML'
    body = m.group(0)
    # every read of a narrative field that lands in HTML is wrapped in _esc(
    for field in ('row.headline', 'row.outlook', 'row.influence',
                  'nar.model'):
        for at in [m.start() for m in re.finditer(re.escape(field), body)]:
            if body[at - 1] in '!(' and \
                    body[at + len(field)] in '?)&|':
                continue  # a truthiness guard, not an interpolation
            prefix = body[max(0, at - 6):at]
            assert '_esc(' in prefix, \
                '%s is interpolated without _esc()' % field
    assert "STANCE={tailwind:'up',headwind:'down'}" in body, \
        'stance must map to CSS classes only through the whitelist'
    assert "STANCE[row.stance]||''" in body
    assert '[data-theme="dark"] #macro-view{' in css, \
        'the macro tokens need a dark-mode redefinition'


def test_macro_narrative_is_lead_paragraphs_with_bullets():
    """The narrative reads part paragraph, part list: each kickered section
    opens with its lead sentence as a paragraph and lists the supporting
    sentences as bullets, and the tailwinds and headwinds are short lists.
    The split is done client-side from the model's paragraphs, so cached
    narratives render the same way."""
    css = _css()
    nar = re.search(r'function _macNarrativeHTML\(\).*?\n\}\n', css, re.S)
    assert nar, 'could not find _macNarrativeHTML'
    nar = nar.group(0)
    assert '_macSentences(p)' in nar
    assert '<p class="mac-nar-p">\'+_esc(ss[0])' in nar, \
        'the lead sentence is the paragraph'
    assert "ss.slice(1).forEach" in nar and 'mac-nar-ul' in nar, \
        'the rest of the section renders as bullets'
    assert 's.outlook' not in nar and 'row.outlook' not in nar, \
        'the Overview prints the influence, not the one-line outlook'
    assert re.search(r'\.mac-narrative\{[^}]*max-width:\d+ch', css), \
        'prose needs a reading measure'
    assert re.search(r'\.mac-nar-p\{[^}]*line-height:1\.[6-9]', css), \
        'paragraphs need a generous leading'
    assert '.mac-nar-cols' not in css, 'the two-column bullet grid is gone'


def test_macro_sentence_split_keeps_decimals_and_abbreviations():
    """The splitter must never break a sentence on a decimal (4.96%) or on
    an abbreviation (U.S., vs.) — pin the guards in the regexes."""
    fn = re.search(r'function _macSentences\(text\)\{.*?\n\}\n', _css(), re.S)
    assert fn, 'could not find _macSentences'
    fn = fn.group(0)
    assert r'\s+(?=' in fn, 'a split needs whitespace after the stop'
    assert r'\bvs' in fn and r'\b[A-Za-z]' in fn


def test_macro_narrative_sectors_open_in_place():
    """The stance chips are the sector index: grouped through the whitelist,
    addressed by index (never by a model string in an attribute), and a chip
    opens that sector's kicker and influence in place with the way into its
    page. A pre-v3 narrative (no influence) links straight to the page. An
    older narrative without `watch` keeps The balance."""
    css = _css()
    nar = re.search(r'function _macNarrativeHTML\(\).*?\n\}\n', css, re.S).group(0)
    assert "STANCE={tailwind:'up',headwind:'down'}" in nar
    assert "grp[STANCE[row.stance]||'']" in nar
    assert "_macSecPick(this,'+it.i+')" in nar, 'chips address sectors by index'
    assert "_macGoSector('+it.i+')" in nar, 'pre-v3 chips link to the page'
    assert '_esc(it.row.sector)' in nar
    assert "watch.forEach(function(x){h+='<li>'+_esc(x)+'</li>';});" in nar
    assert 'The balance' in nar, 'a pre-v7 narrative keeps its winds'
    pick = re.search(r'function _macSecPick\(btn,i\)\{.*?\n\}\n', css, re.S).group(0)
    assert '_esc(row.influence)' in pick and '_esc(row.headline)' in pick
    assert "_macGoSector('+i+')" in pick
    go = re.search(r'function _macGoSector\(i\)\{.*?\n\}\n', css, re.S)
    # A drill like the chart drill-downs, so the way back returns to the
    # Macro sub-tab and scroll position (tests/test_report_drill_back.py).
    assert go and "_drillPush('Macro Outlook')" in go.group(0)
    assert '_poolTab=row.sector' in go.group(0) and '_drillSeal();' in go.group(0)
    assert "navGo('pool'" not in go.group(0), 'navGo would spend the drill at once'


def test_macro_overview_is_the_landing_view():
    """The report opens on the Macro Overview when the snapshot has macro
    data, and on Sector Analysis otherwise. The hash omits `v` only for the
    default view, so both must read the same variable."""
    css = _css()
    assert "var _DEFAULT_VIEW=_MACRO_AVAILABLE?'macro':'pool';" in css
    assert 'var curView=_DEFAULT_VIEW,' in css
    assert 'if(curView!==_DEFAULT_VIEW)p.v=curView;' in css
    assert "if(curView!=='pool')p.v=curView;" not in css
    # the markup ships #view-pool active; init hands the class over
    init = css[css.rindex('moveFilterBar(curView);') - 300:
               css.rindex('moveFilterBar(curView);')]
    assert "classList.toggle('active',el.id==='view-'+curView)" in init


def test_sector_tabs_carry_no_macro_narrative():
    """The Claude narrative's per-sector read renders once, on the Macro
    Outlook Overview under Key sector influences. The Sector Analysis tabs
    used to repeat it in a Macro Outlook section above Sector Headwinds &
    Tailwinds; that block is now the sector's only headwinds/tailwinds read,
    so the macro section, its renderer and its styles are gone, and the
    Overview does not grow a sector grid back either."""
    css = _css()
    for dead in ('renderPoolMacroOutlook', '_macSecFigs', 'pp-macro',
                 '_macSectorsHTML', 'mac-sectors', 'mac-nar-sec'):
        assert dead not in css, '%s is dead, drop it' % dead
    pool = re.search(r'var primerHtml=renderPoolPrimer\(sec\);'
                     r'.*?pp-section pp-liquidity', css, re.S)
    assert pool, 'could not find the per-sector section assembly'
    pool = pool.group(0)
    assert 'Macro Outlook<' not in pool
    assert pool.find('pp-section pp-primer') < pool.find('pp-section pp-signals') \
        < pool.find('pp-section pp-structure'), \
        'Sector Headwinds & Tailwinds follows the primer directly'
    ov = re.search(r'function _macOverviewHTML\(\).*?\n\}\n', css, re.S)
    assert ov, 'could not find _macOverviewHTML'
    body = re.sub(r'/\*.*?\*/', '', ov.group(0), flags=re.S)
    assert 'sector' not in body.lower(), \
        'the Overview is the story and the tiles, nothing else'


def test_macro_overview_does_not_repeat_section_charts():
    """Every full history is drawn on one of the section sub-tabs; the
    Overview keeps the sparkline tiles, each section's figures end in a Full
    charts link, and a tile opens its series' full chart in a dialog."""
    css = _css()
    body = re.search(r'function _macOverviewHTML\(\).*?\n\}\n', css, re.S).group(0)
    assert '_macCurveCard' not in body, 'the yield curve lives on Rates & Curve'
    assert '_macCardHTML' not in body, 'full histories live on the section tabs'
    assert '_mSparkSVG' in body, 'the glance tiles keep their sparklines'
    assert '_macZoomAttrs(' in body, 'a tile opens its larger chart'
    card = re.search(r'function _macCardHTML\(id,s\).*?\n\}\n', css, re.S).group(0)
    assert '_macZoomAttrs(id,s.l)' in card and '_MAC_EXP' in card, 'so does every section card'
    fig = re.search(r'function _macNarFigure\(.*?\n\}\n', css, re.S).group(0)
    assert 'Full charts' in fig and "navGo(\\'macro\\'" in fig
    sec = re.search(r'function _macSectionHTML\(k\).*?\n\}\n', css, re.S)
    assert sec and "k==='rates'&&MACRO.curve" in sec.group(0)


def test_macro_overview_tile_band_spans_the_page():
    """The key numbers are a band across the page (four columns on a
    desktop) rather than a rail beside the story. No max-width rule may set
    the band's columns, or it would outrank the phone tile floors."""
    css = _css()
    wide = _media_block('@media(min-width:1001px){')
    assert re.search(r'\.mac-overview \.mac-tiles\{[^}]*repeat\(4', wide)
    assert not re.search(r'@media\(max-width:\d+px\)\{[^@]*\.mac-overview \.mac-tiles\{[^}]*grid-template-columns', css), \
        'a max-width rule on .mac-overview .mac-tiles would beat the phone floors'


def test_every_macro_subtab_opens_with_an_explainer():
    """Each Macro Outlook section sub-tab leads with one paragraph saying
    what its graphs show and how to read them. The text lives in one map
    keyed by sub-tab; a section added to MACRO_TABS without an entry would
    render bare, so the two lists must agree. The Overview carries no
    explainer — its narrative speaks for itself."""
    css = _css()
    tabs = re.search(r'var MACRO_TABS=\[(.*?)\];', css, re.S)
    assert tabs, 'could not find MACRO_TABS'
    keys = re.findall(r"k:'(\w+)'", tabs.group(1))
    intro = re.search(r'var _MAC_SEC_INTRO=\{(.*?)\n\};', css, re.S)
    assert intro, 'could not find _MAC_SEC_INTRO'
    have = dict(re.findall(r"\n  (\w+):'(.*)'", intro.group(1)))
    assert 'overview' in keys and 'overview' not in have
    for k in keys:
        if k == 'overview':
            continue
        assert k in have, f'sub-tab {k!r} has no explainer paragraph'
        assert len(have[k]) > 200, f'{k!r} explainer is not a paragraph'
    ov = re.search(r'function _macOverviewHTML\(\).*?\n\}\n', css, re.S)
    assert ov and '_macIntroHTML' not in ov.group(0)
    sec = re.search(r'function _macSectionHTML\(k\).*?\n\}\n', css, re.S)
    assert sec and '_macIntroHTML(k)' in sec.group(0)
    # the explainer precedes the data note, so it is the first thing read
    body = sec.group(0)
    assert body.find('_macIntroHTML(k)') < body.find('mac-sec-note')


def test_macro_narrative_sections_carry_figures():
    """Each kickered narrative section is followed by a figure of the
    indicators its paragraph cites (found by repo-authored patterns, in the
    order cited), and the sector influences open with a stance strip. The
    model text is searched, never echoed into the figure."""
    css = _css()
    nar = re.search(r'function _macNarrativeHTML\(\).*?\n\}\n', css, re.S)
    assert nar, 'could not find _macNarrativeHTML'
    nar = nar.group(0)
    assert 'h+=_macNarFigure(i,p)' in nar, \
        'each narrative section needs its figure'
    assert '_macStanceStrip(grp,secLink)' in nar, \
        'the sector influences open with the stance strip'
    fig = re.search(r'function _macNarFigure\(i,text\)\{.*?\n\}\n', css,
                    re.S)
    assert fig, 'could not find _macNarFigure'
    body = fig.group(0)
    assert 'text' not in body.replace('_macNarIds(i,text)', '') \
        .replace('function _macNarFigure(i,text)', ''), \
        'the paragraph only selects series; it is never written out'
    assert '_macRankWords(' in body, 'the rank is stated in words'
    defaults = re.search(r'var _MAC_NAR_DEFAULT=\[(.*?)\];', css, re.S)
    assert defaults and defaults.group(1).count('[') == 3, \
        'one default list per narrative paragraph'


def test_regime_note_matches_the_model_weights_and_cutoffs():
    """The Market Regime box explains its meters in words, quoting the
    composite's weights and regime cut-offs; keep them in step with
    models/macro.py so the explanation never describes another model."""
    from models.macro import MACRO_WEIGHTS, assess_macro_regime
    src = _css()
    m = re.search(r"var _MAC_REGIME_NOTE='([^']*)'", src)
    assert m, 'regime explainer missing'
    note = m.group(1)
    assert MACRO_WEIGHTS == {'vix': 0.25, 'yield_curve': 0.25,
                             'credit_spread': 0.20, 'spy_momentum': 0.15,
                             'industrial_rs': 0.15}
    assert '25% each' in note and '20%' in note and '15% each' in note
    for phrase in ('+0.40 Expansion', '+0.10 Mid Cycle', '−0.10 Neutral',
                   '−0.40 Late Cycle'):
        assert phrase in note
    import inspect
    body = inspect.getsource(assess_macro_regime)
    for cut in ('> 0.40', '> 0.10', '> -0.10', '> -0.40'):
        assert cut in body
    assert '_MAC_REGIME_NOTE' in src.split('function _macRegimeHTML', 1)[1]
