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


def test_out_of_grid_macro_cards_are_capped_in_landscape():
    """The yield-curve and OAS cards are direct children of #macro-view, so
    the grid floor above does not reach them — they need their own cap or the
    curve alone fills a landscape viewport."""
    rules = re.findall(r'#macro-view>\.mac-card\{([^}]*)\}',
                       _media_block(_LANDSCAPE))
    assert rules and any('max-width' in r for r in rules)
