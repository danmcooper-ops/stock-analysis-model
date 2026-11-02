# tests/test_report_zoom_reset.py
"""iOS Safari zooms in on a focused small-text field and stays zoomed after
it blurs. The template pins the viewport to scale 1 for a moment on focusout
and then restores the meta tag's own content, so pinch zoom keeps working."""
import os
import re

_TEMPLATE = os.path.join(os.path.dirname(__file__), '..', 'templates', 'report.html')


def _src():
    return open(_TEMPLATE, encoding='utf-8').read()


def test_focusout_resets_zoom_and_restores_viewport():
    s = _src()
    i = s.index("document.addEventListener('focusout'")
    block = s[i:i + 900]
    assert '_isIos()' in block
    assert 'maximum-scale=1.0' in block and 'minimum-scale=1.0' in block
    # the original content comes back, so the zoom-in feature survives
    assert "m.setAttribute('content',cur)" in block
    # moving focus between fields keeps the zoom
    assert '_isField(document.activeElement)' in block


def test_viewport_meta_still_allows_zooming_in():
    meta = re.search(r'<meta name="viewport" content="([^"]+)"', _src()).group(1)
    assert 'maximum-scale' not in meta and 'user-scalable' not in meta
