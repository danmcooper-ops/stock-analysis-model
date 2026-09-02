# tests/test_report_macro_layout.py
"""Guard: the Macro Outlook tab's place in the nav and its phone layout.

Two things about this tab are invisible on a desktop browser and so are easy
to undo by accident:

  * it leads the nav menu (leftmost column on desktop, first block in the
    stacked phone menu) because it is the top-down context for everything the
    other views say about individual names. It is the only group appended
    conditionally, so the natural edit — `groups.push` — silently sends it
    back to last place;
  * its Overview is a story beside a rail of glance tiles — the full
    histories are drawn once, on the section sub-tabs, never repeated here;
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


def test_macro_narrative_leads_the_overview_subtab():
    """The Claude narrative is the tab's headline read; it must render before
    the tile grid, and only from the narrative renderer so the block cannot
    silently drift below the tiles."""
    body = re.search(r'function _macOverviewHTML\(\).*?\n\}\n', _css(), re.S)
    assert body, 'could not find _macOverviewHTML'
    body = body.group(0)
    nar_at = body.find('_macNarrativeHTML()')
    tiles_at = body.find('mac-tiles')
    assert nar_at >= 0, '_macOverviewHTML no longer renders the narrative'
    assert tiles_at > nar_at, 'the narrative must precede the tile grid'


def test_macro_narrative_escapes_every_model_string():
    """Narrative strings are Claude output, not repo-authored markup: every
    interpolation of them must pass through _esc(), and the stance value may
    only reach a class attribute through the whitelist lookup."""
    css = _css()
    body = ''
    for fn in ('_macNarrativeHTML', '_macSectorsHTML'):
        m = re.search(r'function ' + fn + r'\(\).*?\n\}\n', css, re.S)
        assert m, 'could not find %s' % fn
        body += m.group(0)
    # every read of a narrative field that lands in HTML is wrapped in _esc(
    for field in ('s.sector', 's.headline', 's.outlook', 'nar.model'):
        for at in [m.start() for m in re.finditer(re.escape(field), body)]:
            if (field == 's.sector'
                    and body[at - 3:at + len(field) + 1] == 'sd[s.sector]'):
                # data lookup keyed by the schema-pinned GICS enum; the value
                # never lands in HTML (it feeds _macSecFigs, which formats)
                continue
            prefix = body[max(0, at - 6):at]
            assert '_esc(' in prefix, \
                '%s is interpolated without _esc()' % field
    assert "STANCE={tailwind:'up',headwind:'down'}" in body, \
        'stance must map to CSS classes only through the whitelist'
    assert "STANCE[s.stance]||''" in body
    # trend reaches HTML only through the glyph whitelist in _macSecFigs
    figs = re.search(r'function _macSecFigs\(.*?\n\}\n', css, re.S)
    assert figs and 'TRENDG={improving:' in figs.group(0), \
        'trend must map to glyphs only through the whitelist'


def test_macro_narrative_sector_rows_carry_metric_figs():
    """Each sector entry carries the sector's hard ETF numbers from
    sector_data, sourced inline (MACRO_SUM) so they paint at first render,
    and degrading to prose-only when a sector has no metrics or the
    snapshot predates sector_data. They are set in the muted token so they
    read as a footnote to the sentence, not a second column."""
    css = _css()
    figs = re.search(r'function _macSecFigs\(d\)\{.*?\n\}\n', css, re.S)
    assert figs, 'could not find _macSecFigs'
    figs = figs.group(0)
    assert "if(!d)return ''" in figs, 'figs must degrade to nothing'
    # local parquet RS is fresher than the yfinance fallback — keep the order
    assert figs.find('d.rs_3m!=null') < figs.find('rel_strength_3m'), \
        'rs_3m must be preferred over rel_strength_3m'
    sec = re.search(r'function _macSectorsHTML\(\).*?\n\}\n', css, re.S)
    assert sec, 'could not find _macSectorsHTML'
    sec = sec.group(0)
    assert '(MACRO_SUM&&MACRO_SUM.sector_data)||(MACRO&&MACRO.sector_data)' \
        in sec, 'sector_data must come from the inline summary first'
    assert '_macSecFigs(sd[s.sector])' in sec
    assert re.search(r'\.mac-nar-figs\{[^}]*tabular-nums', css), \
        'metric figs need tabular numerals'
    assert re.search(r'\.mac-nar-figs\{[^}]*color:var\(--mac-', css), \
        'figs take their colour from the macro tokens (dark mode swaps them)'
    assert '[data-theme="dark"] #macro-view{' in css, \
        'the macro tokens need a dark-mode redefinition'


def test_macro_narrative_is_prose_not_lists():
    """The narrative reads top to bottom as a story: kickered paragraphs at
    a book measure, the tailwinds and headwinds folded into one sentence
    each, and the sector outlooks grouped by stance (through the same
    whitelist that picks their class) rather than laid out as a grid of
    rows with bullet lists beside it."""
    css = _css()
    nar = re.search(r'function _macNarrativeHTML\(\).*?\n\}\n', css, re.S)
    assert nar, 'could not find _macNarrativeHTML'
    nar = nar.group(0)
    assert '<ul' not in nar and '<li' not in nar, \
        'tailwinds/headwinds are sentences now, not bullet lists'
    assert '_macJoinClauses(tw)' in nar and '_macJoinClauses(hw)' in nar
    assert 's.outlook' not in nar, \
        'the sector outlooks are their own section, not part of the story'
    assert re.search(r'\.mac-narrative\{[^}]*max-width:\d+ch', css), \
        'prose needs a reading measure'
    assert re.search(r'\.mac-nar-p\{[^}]*line-height:1\.[6-9]', css), \
        'paragraphs need a generous leading'
    assert '.mac-nar-cols' not in css, 'the two-column bullet grid is gone'


def test_macro_sector_implications_are_their_own_section():
    """The eleven sector outlooks render as a full-width section beneath the
    story and the tiles — three stance columns, collapsing to one on a
    phone — rather than as a tail on the narrative card."""
    css = _css()
    sec = re.search(r'function _macSectorsHTML\(\).*?\n\}\n', css, re.S)
    assert sec, 'could not find _macSectorsHTML'
    sec = sec.group(0)
    assert "GROUPS=[['tailwind'," in sec and "['neutral'," in sec \
        and "['headwind'," in sec, 'sectors are grouped by stance'
    assert 'mac-sec-cols' in sec
    ov = re.search(r'function _macOverviewHTML\(\).*?\n\}\n', css, re.S)
    assert ov, 'could not find _macOverviewHTML'
    ov = ov.group(0)
    assert ov.find('_macSectorsHTML()') > ov.find('mac-tiles'), \
        'the sector section follows the story + tiles grid'
    assert re.search(r'\.mac-sec-cols\{[^}]*repeat\(3,', css)
    assert ('@media(max-width:900px){.mac-sec-cols{grid-template-columns:1fr;}}'
            in css), 'the sector columns need a narrow-screen collapse'


def test_macro_overview_does_not_repeat_section_charts():
    """Every full history is drawn on one of the five section sub-tabs; the
    Overview used to redraw the yield curve and the 10Y–2Y history under
    the tiles. It keeps the sparkline tiles and links to the sections."""
    css = _css()
    body = re.search(r'function _macOverviewHTML\(\).*?\n\}\n', css, re.S)
    assert body, 'could not find _macOverviewHTML'
    body = body.group(0)
    assert '_macCurveCard' not in body, 'the yield curve lives on Rates & Curve'
    assert '_macCardHTML' not in body, 'full histories live on the section tabs'
    assert '_mSparkSVG' in body, 'the glance tiles keep their sparklines'
    assert "navGo('macro'" in body.replace('\\', ''), \
        'the overview must point at the section sub-tabs'
    # the curve is still drawn where it belongs
    sec = re.search(r'function _macSectionHTML\(k\).*?\n\}\n', css, re.S)
    assert sec and "k==='rates'&&MACRO.curve" in sec.group(0)


def test_macro_overview_rail_collapses_below_desktop():
    """Story left, tiles right on a desktop; one column below ~1000px. The
    two-column rule must be the one behind a min-width query so the phone
    tile floors (single-class rules further down) are not outranked."""
    css = _css()
    assert re.search(r'\.mac-overview\{[^}]*grid-template-columns:1fr;', css)
    wide = _media_block('@media(min-width:1001px){')
    assert re.search(r'\.mac-overview\{[^}]*grid-template-columns:minmax', wide)
    assert re.search(r'\.mac-overview \.mac-tiles\{[^}]*1fr 1fr', wide)
    assert not re.search(r'@media\(max-width:\d+px\)\{[^@]*\.mac-overview \.mac-tiles\{[^}]*grid-template-columns', css), \
        'a max-width rule on .mac-overview .mac-tiles would beat the phone floors'


def test_every_macro_subtab_opens_with_an_explainer():
    """Each Macro Outlook sub-tab leads with one paragraph saying what its
    graphs show and how to read them. The text lives in one map keyed by
    sub-tab; a sub-tab added to MACRO_TABS without an entry would render
    bare, so the two lists must agree, and both renderers must draw it."""
    css = _css()
    tabs = re.search(r'var MACRO_TABS=\[(.*?)\];', css, re.S)
    assert tabs, 'could not find MACRO_TABS'
    keys = re.findall(r"k:'(\w+)'", tabs.group(1))
    intro = re.search(r'var _MAC_SEC_INTRO=\{(.*?)\n\};', css, re.S)
    assert intro, 'could not find _MAC_SEC_INTRO'
    have = dict(re.findall(r"\n  (\w+):'(.*)'", intro.group(1)))
    for k in keys:
        assert k in have, f'sub-tab {k!r} has no explainer paragraph'
        assert len(have[k]) > 200, f'{k!r} explainer is not a paragraph'
    ov = re.search(r'function _macOverviewHTML\(\).*?\n\}\n', css, re.S)
    assert ov and "_macIntroHTML('overview')" in ov.group(0)
    sec = re.search(r'function _macSectionHTML\(k\).*?\n\}\n', css, re.S)
    assert sec and '_macIntroHTML(k)' in sec.group(0)
    # the explainer precedes the data note, so it is the first thing read
    body = sec.group(0)
    assert body.find('_macIntroHTML(k)') < body.find('mac-sec-note')
