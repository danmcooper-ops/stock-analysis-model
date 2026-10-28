"""Port a built report into the Artifact-wrapper HTML shape.

    python scripts/make_artifact.py output/currency_screener_<date>.html out.html

Claude Artifacts wrap the supplied file in their own
`<!doctype html><head></head><body>` skeleton, so the page content must be
written without those tags. This script performs that transform on a normal
build rather than maintaining a second template — the report stays one
source of truth, and the artifact is always a faithful port of what CI ships.

Two substantive changes beyond unwrapping:

**Theme tokens gain the two missing cases.** The standalone report defines
light tokens on `:root` and dark under `:root[data-theme="dark"]`, because its
own bootstrap script always stamps the attribute. Inside the artifact viewer
the attribute may be absent (follow the OS) or stamped either way by the
viewer's own toggle, so the palette is emitted four times: light on `:root`,
dark under `prefers-color-scheme`, then both explicit attribute selectors so
the viewer's toggle overrides the media query in both directions.

**The bootstrap stamps light explicitly.** Previously a stored 'light'
preference left the attribute unset, which was correct only because there was
no media query to fall through to. With one present, an unset attribute on a
dark-mode OS would silently contradict the stored choice.
"""

import argparse
import os
import re
import sys

BOOTSTRAP_OLD = """(function(){try{var t=localStorage.getItem('ccy_theme_v1');
if(t==='dark')document.documentElement.setAttribute('data-theme','dark');
else if(!t&&window.matchMedia&&matchMedia('(prefers-color-scheme:dark)').matches)
document.documentElement.setAttribute('data-theme','dark');}catch(e){}})();"""

BOOTSTRAP_NEW = """(function(){try{var t=localStorage.getItem('ccy_theme_v1');
if(t==='dark'||t==='light')document.documentElement.setAttribute('data-theme',t);
}catch(e){}})();"""


def _block(css, selector):
    """Return the declarations inside `selector{...}`."""
    i = css.index(selector + '{')
    j = css.index('}', i)
    return css[i + len(selector) + 1:j].strip()


def transform(html):
    # --- split the document -------------------------------------------------
    head = re.search(r'<head>(.*?)</head>', html, re.S).group(1)
    body = re.search(r'<body>(.*?)</body>', html, re.S).group(1)

    style = re.search(r'<style>(.*?)</style>', head, re.S).group(1)
    scripts = re.findall(r'<script>(.*?)</script>', head, re.S)
    bootstrap = next((s for s in scripts if 'ccy_theme_v1' in s), '')

    # --- rebuild the theme tokens ------------------------------------------
    light = _block(style, ':root')
    dark = _block(style, ':root[data-theme="dark"]')

    tokens = (
        ':root{\n%s\n}\n'
        '@media (prefers-color-scheme: dark){:root{\n%s\n}}\n'
        ':root[data-theme="dark"]{\n%s\n}\n'
        ':root[data-theme="light"]{\n%s\n}\n'
        % (light, dark, dark, light)
    )

    # Drop the two original blocks and prepend the four-way token set.
    rest = style
    for sel in (':root[data-theme="dark"]', ':root'):
        i = rest.index(sel + '{')
        j = rest.index('}', i)
        rest = rest[:i] + rest[j + 1:]
    style = tokens + rest.lstrip()

    bootstrap = bootstrap.replace(BOOTSTRAP_OLD, BOOTSTRAP_NEW)
    if BOOTSTRAP_NEW not in bootstrap:
        raise SystemExit('bootstrap script changed shape; update make_artifact.py')

    return '<style>\n%s\n</style>\n<script>%s</script>\n%s' % (
        style, bootstrap, body.strip())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source')
    ap.add_argument('dest')
    args = ap.parse_args(argv)

    with open(args.source, encoding='utf-8') as f:
        html = f.read()

    out = transform(html)

    # Word-boundary match: a bare substring check would flag the page's own
    # <header> element as a surviving <head>.
    stray = re.search(r'<!doctype|</?(html|head|body)\s*>', out, re.I)
    if stray:
        raise SystemExit('document tag %r survived the transform' % stray.group(0))

    os.makedirs(os.path.dirname(os.path.abspath(args.dest)) or '.', exist_ok=True)
    with open(args.dest, 'w', encoding='utf-8') as f:
        f.write(out)
    print('wrote %s (%.0f KB)' % (args.dest, os.path.getsize(args.dest) / 1024))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
