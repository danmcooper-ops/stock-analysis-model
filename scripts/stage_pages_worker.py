#!/usr/bin/env python3
"""Put the login-checking Worker into a Cloudflare Pages deploy directory.

The report on Cloudflare Pages sits behind Cloudflare Access (allowlisted
emails, one-time-code login). ``cloudflare/pages/_worker.js`` re-verifies the
Access JWT on every request before serving a file, so the site fails closed
if Access is ever detached from it. Pages runs a ``_worker.js`` found at the
root of the deploy directory ("advanced mode"); this copies it there with the
team domain and application audience tag filled in.

Both values come from the environment (``CF_ACCESS_TEAM_DOMAIN``,
``CF_ACCESS_AUD``) and neither is a secret: the team domain is public and the
AUD tag only names the Access application. They are validated here, because
a typo would lock everyone out rather than let anyone in, and the nightly log
is the place to catch that.

Usage:
    python scripts/stage_pages_worker.py <deploy dir>

Exit code: 0 staged, 1 missing or malformed configuration.
"""
import argparse
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(REPO, 'cloudflare', 'pages', '_worker.js')
TEAM_PLACEHOLDER = '__ACCESS_TEAM_DOMAIN__'
AUD_PLACEHOLDER = '__ACCESS_AUD__'
TEAM_RE = re.compile(r'^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.cloudflareaccess\.com$')
AUD_RE = re.compile(r'^[0-9a-f]{64}$')


def render(template, team, aud):
    """Return the Worker source with the Access settings filled in."""
    team = (team or '').strip().lower()
    aud = (aud or '').strip().lower()
    if not TEAM_RE.match(team):
        raise ValueError(f'CF_ACCESS_TEAM_DOMAIN must look like <team>.cloudflareaccess.com, got {team!r}')
    if not AUD_RE.match(aud):
        raise ValueError(f'CF_ACCESS_AUD must be the 64-hex-character Application Audience tag, got {aud!r}')
    for placeholder in (TEAM_PLACEHOLDER, AUD_PLACEHOLDER):
        if f"'{placeholder}'" not in template:
            raise ValueError(f'{placeholder} not found in the Worker template')
    return (template.replace(f"'{TEAM_PLACEHOLDER}'", f"'{team}'", 1)
                    .replace(f"'{AUD_PLACEHOLDER}'", f"'{aud}'", 1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('deploy_dir', help='the directory wrangler will deploy')
    args = ap.parse_args(argv)

    if not os.path.isdir(args.deploy_dir):
        print(f'stage_pages_worker: {args.deploy_dir} is not a directory', file=sys.stderr)
        return 1
    with open(TEMPLATE, encoding='utf-8') as fh:
        template = fh.read()
    try:
        source = render(template, os.environ.get('CF_ACCESS_TEAM_DOMAIN'), os.environ.get('CF_ACCESS_AUD'))
    except ValueError as e:
        print(f'stage_pages_worker: {e}', file=sys.stderr)
        return 1
    out = os.path.join(args.deploy_dir, '_worker.js')
    with open(out, 'w', encoding='utf-8') as fh:
        fh.write(source)
    print(f'stage_pages_worker: wrote {out} (Access team {os.environ["CF_ACCESS_TEAM_DOMAIN"].strip().lower()})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
