#!/usr/bin/env python3
"""Render and deliver the portfolio-alerts digest.

The nightly run writes ``portfolio_alerts.json`` (``scripts/portfolios.py
alerts --json``) and archives it on the ``data/snapshots`` branch. This module
turns that JSON into the nightly text file and a GitHub-issue body, and posts
the issue. It imports nothing outside the standard library, so the digest
workflow (``.github/workflows/portfolio-alerts.yml``) runs it on a bare
runner without installing the project.

    python scripts/portfolio_digest.py text alerts.json
    python scripts/portfolio_digest.py md alerts.json [--pages-url URL]
    python scripts/portfolio_digest.py post alerts.json --repo OWNER/NAME [--pages-url URL] [--dry-run]

``post`` is idempotent per date: it creates ``Portfolio alerts — <date>``
only when no issue with that title exists and the digest has at least one
Action/Watch alert or a systemic-day banner; it then closes older open
digests. Quiet days post nothing.
"""
import argparse
import json
import subprocess
import sys

LABEL = 'portfolio-alerts'
# GitHub rejects issue bodies over 65,536 characters; a flood day's FYI list
# alone can run to hundreds of lines.
MAX_FYI_LINES = 60
MAX_BODY = 60000
LEVEL_NAMES = {'action': 'Action', 'watch': 'Watch', 'fyi': 'FYI'}
RATINGS = ('BUY', 'LEAN BUY', 'HOLD', 'PASS')


def _pct(v, sign=True):
    if not isinstance(v, (int, float)):
        return '—'
    return f"{v:+.0%}" if sign else f"{v:.0%}"


def stats_line(st):
    """One line of portfolio stats (rating mix, medians, concentration)."""
    mix = ' '.join(f"{k}:{v}" for k, v in (st.get('ratings') or {}).items() if v) or 'no ratings'
    score = st.get('median_score')
    conc = ''
    if st.get('top_sector'):
        conc = f"  top sector {st['top_sector']} {_pct(st.get('top_sector_weight'), False)}"
        if st.get('concentrated'):
            conc += ' (concentrated)'
    return (f"{mix}  median MoS {_pct(st.get('median_mos'))}  "
            f"score {'—' if score is None else f'{score:.0f}'}  "
            f"spread {_pct(st.get('median_spread'))}{conc}")


def counts(pf):
    c = {'action': 0, 'watch': 0, 'fyi': 0}
    for a in pf.get('alerts') or ():
        c[a['level']] = c.get(a['level'], 0) + 1
    return c


def totals(digest):
    t = {'action': 0, 'watch': 0, 'fyi': 0}
    for pf in digest.get('portfolios') or ():
        for k, v in counts(pf).items():
            t[k] += v
    return t


def should_post(digest):
    """Post only when there is something to act on or a systemic banner."""
    t = totals(digest)
    return bool(t['action'] or t['watch'] or (digest.get('systemic') or {}).get('flood'))


def title(digest):
    t = totals(digest)
    parts = [f"{t[k]} {LEVEL_NAMES[k].lower()}" for k in ('action', 'watch') if t[k]]
    flag = ' — systemic day' if (digest.get('systemic') or {}).get('flood') else ''
    return f"Portfolio alerts — {digest['date']}" + (f" ({', '.join(parts)})" if parts else '') + flag


def render_text(digest):
    """The nightly output/portfolio_alerts_<date>.txt."""
    lines = [f"Portfolio alerts {digest['date']} (vs {digest.get('prev_date') or 'no prior snapshot'})", '']
    msg = (digest.get('systemic') or {}).get('message')
    if msg:
        lines += [f"!! {msg}", '']
    pfs = digest.get('portfolios') or []
    if not pfs:
        return '\n'.join(lines + ['No portfolios defined.']) + '\n'
    for pf in pfs:
        st = pf.get('stats') or {}
        chg = (f"  today: +{st.get('upgrades', 0)} up / -{st.get('downgrades', 0)} down"
               if digest.get('prev_date') else '')
        mode = '' if pf.get('mode', 'buy_line') == 'buy_line' else f"  [alerts: {pf['mode']}]"
        lines.append(f"{pf['name']} [{pf['id']}]: {st.get('n', 0)} stocks{chg}{mode}")
        lines.append('  ' + stats_line(st))
        fyi = 0
        for a in pf.get('alerts') or ():
            if a['level'] == 'fyi':
                fyi += 1
                continue
            lines.append(f"  {LEVEL_NAMES[a['level']].upper():<6} {a['message']}")
            for w in a.get('why') or ():
                lines.append(f"           · {w}")
        if fyi:
            lines.append(f"  ({fyi} quieter change{'s' if fyi != 1 else ''}: moves within a side or on missing data)")
        lines.append('')
    t = totals(digest)
    note = '' if digest.get('prev_date') else ' — no prior snapshot, so no change alerts'
    lines.append(f"{t['action']} action, {t['watch']} watch, {t['fyi']} FYI{note}")
    return '\n'.join(lines) + '\n'


def render_markdown(digest, pages_url=None):
    """The GitHub issue body."""
    out = [f"## Portfolio alerts — {digest['date']}"]
    sub = f"_vs {digest.get('prev_date') or 'no prior snapshot'}"
    if pages_url:
        sub += f" · [open the Portfolios view]({pages_url.rstrip('/')}/#s=%7B%22v%22%3A%22pf%22%7D)"
    out += [sub + '_', '']
    msg = (digest.get('systemic') or {}).get('message')
    if msg:
        out += [f"> ⚠️ **{msg.split(':', 1)[0]}:**{msg.split(':', 1)[1] if ':' in msg else ''}", '']
    for pf in digest.get('portfolios') or ():
        st = pf.get('stats') or {}
        mix = ' · '.join(f"{k} {v}" for k, v in (st.get('ratings') or {}).items())
        out += [f"### {pf['name']} — {st.get('n', 0)} stocks", f"{mix} · median MoS {_pct(st.get('median_mos'))}", '']
        fyi = [a for a in pf.get('alerts') or () if a['level'] == 'fyi']
        for lv in ('action', 'watch'):
            items = [a for a in pf.get('alerts') or () if a['level'] == lv]
            if not items:
                continue
            out.append(f"**{LEVEL_NAMES[lv]}**")
            for a in items:
                msg_ = a['message']
                t = a['ticker']
                if msg_.startswith(t + ' '):
                    msg_ = msg_[len(t) + 1:]
                out.append(f"- **{t}** {msg_}")
                for w in a.get('why') or ():
                    out.append(f"  - {w}")
            out.append('')
        if fyi:
            out.append(f"<details><summary>{len(fyi)} quieter change{'s' if len(fyi) != 1 else ''}</summary>\n")
            out += [f"- {a['message']}" for a in fyi[:MAX_FYI_LINES]]
            if len(fyi) > MAX_FYI_LINES:
                out.append(f"- … and {len(fyi) - MAX_FYI_LINES} more (see the report)")
            out += ['', '</details>', '']
        if not (pf.get('alerts') or ()):
            out += ['_No changes._', '']
    out.append('<sub>Action = crossed the buy line (into or out of BUY/LEAN BUY). '
               'Watch = reversals of a recent crossing, score drops ≥10 pts, fair-value jumps ≥50%, '
               'portfolio joins/leaves, earnings within 7 days. FYI = moves within a side, or on missing data.</sub>')
    body = '\n'.join(out) + '\n'
    if len(body) > MAX_BODY:
        body = body[:MAX_BODY].rsplit('\n', 1)[0] + '\n\n_… truncated; the full list is in the report._\n'
    return body


# ---------------------------------------------------------------- posting

def _gh(args, run=subprocess.run):
    return run(['gh', *args], check=True, capture_output=True, text=True).stdout


def post(digest, repo, pages_url=None, dry_run=False, run=subprocess.run):
    """Create today's digest issue if warranted and absent; close older ones.
    Returns a short status string."""
    if not should_post(digest):
        return 'quiet day: nothing to post'
    t = title(digest)
    prefix = f"Portfolio alerts — {digest['date']}"
    listing = json.loads(_gh(['issue', 'list', '--repo', repo, '--label', LABEL, '--state', 'all',
                              '--limit', '100', '--json', 'number,title,state'], run) or '[]')
    if any(i['title'].startswith(prefix) for i in listing):
        return f'already posted for {digest["date"]}'
    body = render_markdown(digest, pages_url)
    if dry_run:
        return f'dry run: would create "{t}" ({len(body)} chars)'
    try:
        _gh(['label', 'create', LABEL, '--repo', repo, '--color', '2a78d6',
             '--description', 'Nightly portfolio alerts digest'], run)
    except subprocess.CalledProcessError:
        pass                                   # already exists
    url = _gh(['issue', 'create', '--repo', repo, '--title', t, '--label', LABEL,
               '--body', body], run).strip()
    for i in listing:
        if i.get('state', '').upper() == 'OPEN':
            _gh(['issue', 'close', str(i['number']), '--repo', repo,
                 '--comment', f'Superseded by {url}'], run)
    return f'created {url}'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=('text', 'md', 'post'))
    ap.add_argument('digest', help='portfolio_alerts.json')
    ap.add_argument('--pages-url')
    ap.add_argument('--repo', help='OWNER/NAME (post)')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args(argv)
    with open(a.digest, encoding='utf-8') as f:
        digest = json.load(f)
    if a.cmd == 'text':
        sys.stdout.write(render_text(digest))
    elif a.cmd == 'md':
        sys.stdout.write(render_markdown(digest, a.pages_url))
    else:
        if not a.repo:
            ap.error('post needs --repo')
        print(post(digest, a.repo, a.pages_url, a.dry_run))
    return 0


if __name__ == '__main__':
    sys.exit(main())
