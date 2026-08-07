# tests/test_report_popup_viewport.py
"""Guard: the detail popup can never be sized shorter than the screen.

The popup has ended mid-screen — with the bare table showing below it and the
rest of the popup unreachable — every time some single measurement of "how tall
is the viewport" resolved short: first `--vvh` alone, then `100svh`, then
`100dvh`. The box model that survives is:

  * the overlay is a box constraint (`inset:0`), never a measurement, and
    never given a minimum height (a fixed box taller than the viewport is
    clipped by the viewport with nowhere to scroll, which would strand its
    own bottom);
  * the shell takes the LARGER of two independent readings, so one of them
    resolving short cannot shorten the popup, and too-tall degrades to a
    scroll in the overlay rather than amputation.

These are one-line CSS properties in a 9k-line template, easy to "simplify"
back into a single unit by someone who doesn't know the history, so pin them.
"""
import os
import re

_TEMPLATE = os.path.join(os.path.dirname(__file__), '..', 'templates', 'report.html')


def _css():
    return open(_TEMPLATE, encoding='utf-8').read()


def _rules(selector):
    """Every declaration block written for `selector` (base rule + the media
    query overrides), so a fix applied to only one of them fails here."""
    return re.findall(re.escape(selector) + r'\{([^}]*)\}', _css())


def test_overlay_is_a_box_constraint_not_a_measurement():
    base = [r for r in _rules('.detail-modal') if 'position:fixed' in r]
    assert len(base) == 1, 'expected exactly one positioned .detail-modal rule'
    rule = base[0]
    assert 'inset:0' in rule, '.detail-modal must be sized by inset:0'
    assert not re.search(r'(?<!min-)height:\s*(100|var|calc)', rule), \
        '.detail-modal must not measure its own height (regression: 100svh)'


def test_overlay_has_no_minimum_height():
    for rule in _rules('.detail-modal'):
        assert 'min-height' not in rule, \
            'a min-height taller than the viewport strands the overlay bottom'


def test_overlay_scrolls_rather_than_clipping():
    rule = [r for r in _rules('.detail-modal') if 'position:fixed' in r][0]
    assert 'overflow-y:auto' in rule, \
        'the overlay is the safety valve for a stale-tall shell'
    assert 'overscroll-behavior:contain' in rule


def test_shell_height_takes_the_larger_of_two_readings():
    heights = [re.search(r'(?<!max-)(?<!min-)height:([^;]*)', r)
               for r in _rules('.detail-content')]
    heights = [h.group(1) for h in heights if h]
    assert heights, 'no .detail-content height rule found'
    for h in heights:
        if 'dvh' in h:
            assert 'max(' in h and '--vvh' in h, \
                'dvh alone has no floor; a short reading amputates the popup: ' + h
        else:
            # the pre-dvh @supports arm, where --vvh is the only reading
            assert '--vvh' in h, h


def test_vvh_is_measured_from_the_layout_viewport_top():
    m = re.search(r"setProperty\('--vvh',([^)]*\)?[^)]*)\)", _css())
    assert m, '--vvh is no longer set from visualViewport'
    assert 'offsetTop' in m.group(1), \
        '--vvh must add offsetTop, else pinch-zoom reads a fraction of the screen'


def test_pinch_zoom_out_has_a_floor():
    """The table views pan as page content, so the document is as wide as the
    matrix (~2700px on a phone) and without a minimum-scale Safari lets a
    pinch zoom out until the whole document fits — scale ~0.15, an unreadable
    sliver. The floor must ride in the viewport meta; no CSS or JS can cap
    pinch zoom."""
    meta = re.search(r'<meta name="viewport" content="([^"]*)"', _css())
    assert meta, 'viewport meta missing'
    content = meta.group(1)
    m = re.search(r'minimum-scale=([0-9.]+)', content)
    assert m, 'viewport meta lost its minimum-scale: ' + content
    assert 0.3 <= float(m.group(1)) <= 1.0, content
    # iOS refuses to cap zooming IN (accessibility) — do not ship the flags
    # it would ignore anyway, they only mislead readers of the template.
    assert 'user-scalable=no' not in content and 'maximum-scale' not in content, content


def test_popup_remeasures_the_viewport_before_it_opens():
    body = re.search(r'function openDet\(tk\)\{.*?\n\}', _css(), re.S)
    assert body, 'openDet not found'
    assert '_syncVv' in body.group(0), \
        'the popup must re-measure before it is sized against the reading'
