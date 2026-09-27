# tests/test_report_appearance.py
"""Appearance: Light / Auto / Dark.

stock_theme_v1 holds the viewer's choice ('light' | 'dark' | 'auto',
missing = auto). A pre-paint script resolves it before first paint, and
data-theme="dark" on <html> stays the only thing the CSS and JS read. The
new chrome survives the blanket `[data-theme="dark"] body *{color:#fff
!important}` rule through its --sac colour variable, and every chrome token
pair clears WCAG AA in both themes.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

TEMPLATE = Path(__file__).resolve().parents[1] / 'templates' / 'report.html'


def _tpl():
    return TEMPLATE.read_text(encoding='utf-8')


def _fn(src, name):
    start = src.index('function ' + name + '(')
    return src[start:src.index('\n}\n', start) + 2]


def _prepaint(src):
    m = re.search(r"<script>(\(function\(\)\{var p='auto';.*?\}\)\(\);)</script>", src)
    assert m, 'pre-paint appearance script not found'
    return m.group(1)


@pytest.mark.skipif(shutil.which('node') is None, reason='node not installed')
@pytest.mark.parametrize('stored,os_dark,want_dark,want_pref', [
    ('light', False, False, 'light'),
    ('light', True, False, 'light'),
    ('dark', False, True, 'dark'),
    ('dark', True, True, 'dark'),
    ('auto', False, False, 'auto'),
    ('auto', True, True, 'auto'),
    (None, True, True, 'auto'),       # first visit follows the device
    (None, False, False, 'auto'),
    ('bogus', True, True, 'auto'),    # unknown value = auto
    ('THROW', True, True, 'auto'),    # blocked storage still resolves
])
def test_prepaint_resolves_the_choice(tmp_path, stored, os_dark, want_dark, want_pref):
    stub = (
        'var attrs={};var document={documentElement:{setAttribute:function(k,v){attrs[k]=v;}}};\n'
        'var localStorage={getItem:function(k){if(%s==="THROW")throw new Error("x");return %s;}};\n'
        'var window={matchMedia:function(q){return{matches:%s&&q.indexOf("dark")>=0};}};\n'
    ) % (json.dumps(stored), json.dumps(stored), 'true' if os_dark else 'false')
    js = stub + _prepaint(_tpl()) + '\nconsole.log(JSON.stringify(attrs));\n'
    script = tmp_path / 'prepaint.js'
    script.write_text(js, encoding='utf-8')
    r = subprocess.run(['node', str(script)], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    attrs = json.loads(r.stdout)
    assert attrs.get('data-theme-pref') == want_pref
    assert (attrs.get('data-theme') == 'dark') is want_dark


def test_set_theme_takes_three_choices_and_follows_the_device_live():
    src = _tpl()
    body = _fn(src, 'setTheme')
    assert "p!=='light'&&p!=='dark'" in body and "p='auto'" in body
    assert "localStorage.setItem(_THEME_KEY,p)" in body
    assert "matchMedia('(prefers-color-scheme: dark)')" in src
    assert "_THEME_MQ.addEventListener('change'" in src
    # Only Auto follows the device; an explicit choice ignores it.
    assert "_themePref()==='auto'" in src


def test_every_control_is_a_data_theme_ctl():
    src = _tpl()
    for pref in ('light', 'auto', 'dark'):
        # sidebar radiogroup + the floating menu
        assert src.count(f'data-theme-ctl="{pref}"') >= 2, pref
    assert 'tt-knob' not in src and 'id="theme-toggle"' not in src
    assert '<meta name="theme-color" id="meta-theme-color"' in src


def test_chrome_colour_survives_the_blanket_dark_rule():
    src = _tpl()
    blanket = src.index('[data-theme="dark"] body,[data-theme="dark"] body *{color:#fff !important;}')
    rescue = src.index('html body .sa-chrome,html body .sa-chrome *{color:var(--sac) !important;}')
    # Same !important, higher specificity ((0,1,2) vs (0,1,1)) and later.
    assert rescue > blanket
    for chrome in ('id="sidebar" class="sa-chrome"', 'id="tabbar" class="sa-chrome"',
                   'id="subnav" class="sa-chrome"', 'id="theme-menu" class="sa-chrome"'):
        assert chrome in src, chrome


def _chrome_tokens():
    src = _tpl()
    block = src[src.index('/* === SA CHROME'):]
    root = re.search(r'\n:root\{(.*?)\n\}', block, re.S).group(1)
    dark = re.search(r'\n\[data-theme="dark"\]\{(.*?)\n\}', block, re.S).group(1)

    def parse(css):
        return dict(re.findall(r'(--[\w-]+):\s*([^;]+);', css))
    return parse(root), parse(dark)


def test_chrome_tokens_are_declared_on_bare_root():
    """A token defined only under [data-theme] would be undefined in the
    other theme."""
    light, dark = _chrome_tokens()
    assert set(dark) <= set(light), sorted(set(dark) - set(light))


def _lum(hexv):
    h = hexv.strip().lstrip('#')
    if len(h) == 3:
        h = ''.join(c * 2 for c in h)
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def _ratio(a, b):
    la, lb = _lum(a), _lum(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


# (text, background) — every text colour the chrome paints, on the surface
# it is painted on.
PAIRS = [
    ('--sb-text', '--sb-bg'), ('--sb-muted', '--sb-bg'), ('--chrome-accent', '--sb-bg'),
    ('#ffffff', '--chrome-pill-pos'), ('#ffffff', '--chrome-pill-neg'),
    ('--chip-on-fg', '--chip-on-bg'), ('--chip-fg', '--chip-bg'),
    ('--search-fg', '--search-bg'), ('--search-muted', '--search-bg'),
    ('--link', '--chip-bg'), ('--chrome-pos', '--chip-bg'), ('--chrome-neg', '--chip-bg'),
    ('--chrome-heading', '--chip-bg'), ('--chrome-muted', '--chip-bg'),
    ('--rt-buy-fg', '--rt-buy-bg'), ('--rt-lean-fg', '--rt-lean-bg'),
    ('--rt-hold-fg', '--rt-hold-bg'), ('--rt-sell-fg', '--rt-sell-bg'),
    ('--tabbar-fg', '--tabbar-bg'),
]


@pytest.mark.parametrize('theme', ['light', 'dark'])
def test_chrome_tokens_clear_wcag_aa(theme):
    light, dark = _chrome_tokens()
    toks = dict(light)
    if theme == 'dark':
        toks.update(dark)

    def val(k):
        v = k if k.startswith('#') else toks[k].strip()
        assert v.startswith('#'), (k, v)
        return v
    bad = [(t, b, round(_ratio(val(t), val(b)), 2)) for t, b in PAIRS
           if _ratio(val(t), val(b)) < 4.5]
    assert not bad, bad
