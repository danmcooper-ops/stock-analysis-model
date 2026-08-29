# tests/test_report_macro_layout.py
"""Guard: the Macro Outlook tab's place in the nav and its phone layout.

Two things about this tab are invisible on a desktop browser and so are easy
to undo by accident:

  * it leads the nav menu (leftmost column on desktop, first block in the
    stacked phone menu) because it is the top-down context for everything the
    other views say about individual names. It is the only group appended
    conditionally, so the natural edit — `groups.push` — silently sends it
    back to last place;
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
    body = re.search(r'function _macNarrativeHTML\(\).*?\n\}\n', _css(), re.S)
    assert body, 'could not find _macNarrativeHTML'
    body = body.group(0)
    # every read of a narrative field that lands in HTML is wrapped in _esc(
    for field in ('s.sector', 's.outlook', 'nar.model'):
        for at in [m.start() for m in re.finditer(re.escape(field), body)]:
            prefix = body[max(0, at - 6):at]
            assert '_esc(' in prefix, \
                '%s is interpolated without _esc()' % field
    assert "STANCE={tailwind:'up',headwind:'down'}" in body, \
        'stance must map to CSS classes only through the whitelist'
    assert "STANCE[s.stance]||''" in body


def test_macro_narrative_columns_collapse_on_narrow_screens():
    css = _css()
    # two-column on desktop…
    assert re.search(r'\.mac-nar-cols\{[^}]*grid-template-columns:1fr 1fr',
                     css)
    assert re.search(r'\.mac-nar-sectors\{[^}]*grid-template-columns:1fr 1fr',
                     css)
    # …one column on a phone
    assert ('@media(max-width:700px){.mac-nar-cols,.mac-nar-sectors'
            '{grid-template-columns:1fr;}}') in css, \
        'the narrative grids need a narrow-screen collapse'
