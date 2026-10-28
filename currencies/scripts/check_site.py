"""Verify the published screener landed, and report what changed.

    python scripts/check_site.py [--wait-min 110] [--home DIR]

GitHub Actions builds and deploys the site (`build.yml`). This script is the
check that runs afterwards, on the Mac mini (scheduled-tasks/daily-check/):
it waits for today's build to reach the live site, archives the day's
results, diffs them against the previous archived day, and writes
`report.md` and `status.json` for the scheduled task to summarise.

It replaced a cloud routine that broke in two ways this script is built
around:

**The site URL is derived, not written down.** The routine's prompt named the
repository's old name, and GitHub redirects a renamed repo's git URLs but not
its Pages site — so it read a 404 and reported a healthy site as down. The
base URL comes from `git remote get-url origin` (`--site` overrides).

**It waits instead of assuming.** GitHub starts scheduled runs hours late
(2h45m on 2026-10-15), so a check at a fixed time saw yesterday's site and
called the build failed. This polls `manifest.json` until its `run_date` is
today or `--wait-min` runs out. If it never lands it says why: the ECB
published nothing today (a correct skip — `should_build.ecb_freshness`, the
same question the build gate asks), today's scheduled run failed, or it is
still queued/running (late).

Statuses, most severe first: SITE-WRONG (404, or a ~16 KB README instead of
the ~680 KB report), JPY-INVERTED (the REER sign convention has flipped —
see CLAUDE.md, "The sanity check"), FAILED, LATE, SKIPPED, OK. Exit 0 for
OK and SKIPPED, 1 otherwise. Never rebuilds or publishes: the github-pages
environment only accepts deployments from Actions on the default branch.

Stdlib only, like the rest of the project; network failures degrade to a
status, never a traceback.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

TZ = ZoneInfo('America/New_York')
DEFAULT_HOME = os.path.expanduser('~/CurrencyScreener')
FALLBACK_REMOTE = 'https://github.com/danmcooper-ops/Currency-Screener-'

# The healthy page is ~650-700 KB. Pages falling back to serving the README
# from a branch gives ~16 KB, which is the failure this band exists to name.
PAGE_MIN_BYTES = 600_000
PAGE_MAX_BYTES = 750_000
README_MAX_BYTES = 50_000

# JPY's Valuation pillar sits ~90 (BIS REER 164 in 1994 -> ~65). Below 50 it
# no longer screens cheap, which means the heaviest pillar runs backwards.
JPY_VALUATION_FLOOR = 50.0

HISTORY_KEEP_DAYS = 400

STATUS_ORDER = ('SITE-WRONG', 'JPY-INVERTED', 'FAILED', 'LATE', 'SKIPPED', 'OK')
OK_STATUSES = ('OK', 'SKIPPED')


# ---------------------------------------------------------------- pure parts

def parse_remote(url):
    """'owner/repo' from an HTTPS or SSH GitHub remote, or None."""
    m = re.search(r'github\.com[:/]+([^/]+)/([^/]+?)(?:\.git)?/?$', url or '')
    return '%s/%s' % (m.group(1), m.group(2)) if m else None


def site_url(slug):
    """Project Pages URL for 'owner/repo'. Pages lower-cases the owner."""
    owner, repo = slug.split('/', 1)
    return 'https://%s.github.io/%s/' % (owner.lower(), repo)


def classify_page(status, size):
    """'ok', 'readme', 'missing' or 'odd' for the deployed index.html."""
    if status != 200:
        return 'missing'
    if size < README_MAX_BYTES:
        return 'readme'
    if PAGE_MIN_BYTES <= size <= PAGE_MAX_BYTES:
        return 'ok'
    return 'odd'


def previous_day(history_dir, today):
    """Newest archived run_date strictly before `today`, or None."""
    try:
        names = os.listdir(history_dir)
    except OSError:
        return None
    days = sorted(n for n in names
                  if re.fullmatch(r'\d{4}-\d{2}-\d{2}', n) and n < today
                  and os.path.isfile(os.path.join(history_dir, n,
                                                  'results.json')))
    return days[-1] if days else None


def _rows(payload):
    return {r['code']: r for r in (payload or {}).get('results') or []
            if isinstance(r, dict) and r.get('code')}


def diff_results(prev, cur):
    """Rating changes, plus currencies that appeared or disappeared."""
    p, c = _rows(prev), _rows(cur)
    changes = []
    for code in sorted(c):
        if code in p and p[code].get('rating') != c[code].get('rating'):
            changes.append({'code': code,
                            'old': p[code].get('rating'),
                            'new': c[code].get('rating'),
                            'old_score': p[code].get('_composite_score'),
                            'new_score': c[code].get('_composite_score')})
    return {'changes': changes,
            'added': sorted(set(c) - set(p)),
            'removed': sorted(set(p) - set(c))}


def top_n(payload, n=5):
    rows = [r for r in _rows(payload).values()
            if isinstance(r.get('_composite_score'), (int, float))]
    rows.sort(key=lambda r: -r['_composite_score'])
    return [(r['code'], r['_composite_score'], r.get('rating')) for r in rows[:n]]


def jpy_valuation(payload):
    """(score, inverted). A missing score is reported, not called inverted."""
    score = (_rows(payload).get('JPY') or {}).get('_score_pillar_valuation')
    if not isinstance(score, (int, float)):
        return None, False
    return score, score < JPY_VALUATION_FLOOR


def data_quality(manifest):
    """Provenance problems: stale cache slots and recorded events."""
    prov = (manifest or {}).get('provenance') or {}
    issues = []
    stale = prov.get('stale_slots')
    if isinstance(stale, (int, float)) and stale > 0:
        issues.append('%d stale cache slot(s): an upstream source failed and '
                      'an expired cached response was served' % stale)
    events = prov.get('event_counts') or {}
    if events:
        issues.append('provenance events: %s' % ', '.join(
            '%s=%s' % kv for kv in sorted(events.items())))
    if not prov:
        issues.append('manifest carries no provenance block')
    return issues


def runs_today(runs, today):
    """Scheduled workflow runs created on `today` (New York date)."""
    out = []
    for r in runs or []:
        try:
            created = datetime.fromisoformat(
                r['created_at'].replace('Z', '+00:00')).astimezone(TZ)
        except (KeyError, ValueError, AttributeError):
            continue
        if created.date().isoformat() == today:
            out.append(r)
    return out


def not_landed_status(ecb_proceed, todays_runs):
    """Why today's build is not on the site: SKIPPED, FAILED or LATE.

    `ecb_proceed` False means the ECB published nothing today, so the build
    gate skipped correctly. Otherwise a finished non-success run with nothing
    still going is FAILED; anything else (no run yet, queued, running, or a
    green run whose gate skipped for another reason) is LATE.
    """
    if ecb_proceed is False:
        return 'SKIPPED'
    pending = [r for r in todays_runs if r.get('status') != 'completed']
    failed = [r for r in todays_runs if r.get('status') == 'completed'
              and r.get('conclusion') not in ('success', 'skipped', None)]
    if failed and not pending:
        return 'FAILED'
    return 'LATE'


def worst(statuses):
    return min(statuses, key=STATUS_ORDER.index) if statuses else 'OK'


def render_report(ctx):
    """Markdown for the scheduled task. Leads with what changed."""
    L = ['# Currency screener check — %s' % ctx['today'], '',
         '**Result: %s** — %s' % (ctx['status'], ctx['reason']), '']
    d = ctx.get('diff')
    if d is not None:
        L.append('## Rating changes vs %s' % ctx['prev_date'])
        if d['changes']:
            for c in d['changes']:
                L.append('- %s: %s -> %s (composite %s -> %s)'
                         % (c['code'], c['old'], c['new'],
                            c['old_score'], c['new_score']))
        else:
            L.append('- none')
        if d['added']:
            L.append('- new in universe: %s' % ', '.join(d['added']))
        if d['removed']:
            L.append('- dropped from universe: %s' % ', '.join(d['removed']))
        L.append('')
    elif ctx.get('results') is not None:
        L += ['## Rating changes', '- comparison unavailable: no earlier run '
              'archived on this machine', '']
    if ctx.get('top'):
        L.append('## Top 5 by composite')
        L += ['%d. %s %s (%s)' % (i + 1, c, s, r)
              for i, (c, s, r) in enumerate(ctx['top'])]
        L.append('')
    if ctx.get('results') is not None:
        score, inverted = ctx['jpy']
        L.append('## JPY Valuation')
        if score is None:
            L.append('- not reported in results.json')
        elif inverted:
            L.append('- **%.1f — BELOW %.0f. JPY no longer screens cheap: the '
                     'REER sign convention has likely inverted and the '
                     'Valuation pillar runs backwards.**'
                     % (score, JPY_VALUATION_FLOOR))
        else:
            L.append('- %.1f (screens cheap, as it should)' % score)
        L.append('')
    if ctx.get('results') is not None or ctx.get('issues'):
        L.append('## Data quality')
        L += ['- %s' % i for i in ctx.get('issues') or []] or ['- clean']
        L.append('')
    L.append('## Site')
    L.append('- %s — index.html HTTP %s, %s bytes (%s); manifest run_date %s'
             % (ctx['site'], ctx.get('page_status'), ctx.get('page_size'),
                ctx.get('page_kind'), ctx.get('site_run_date')))
    for r in ctx.get('runs') or []:
        L.append('- workflow run %s: %s/%s %s' % (
            r.get('created_at'), r.get('status'), r.get('conclusion'),
            r.get('html_url')))
    return '\n'.join(L) + '\n'


# ---------------------------------------------------------------- I/O parts

def http_get(url, timeout=30):
    """(status, body). Never raises; status is None on a network failure."""
    req = urllib.request.Request(url, headers={
        'User-Agent': 'currency-screener-check',
        'Cache-Control': 'no-cache'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, b''
    except (urllib.error.URLError, OSError, ValueError):
        return None, b''


def get_json(url):
    status, body = http_get(url)
    if status != 200:
        return status, None
    try:
        return status, json.loads(body)
    except ValueError:
        return status, None


def git_remote():
    try:
        return subprocess.run(
            ['git', '-C', REPO_ROOT, 'remote', 'get-url', 'origin'],
            capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ''


def scheduled_runs(slug):
    """Recent scheduled runs of build.yml via the public Actions API."""
    _, data = get_json('https://api.github.com/repos/%s/actions/workflows/'
                       'build.yml/runs?event=schedule&per_page=5' % slug)
    return (data or {}).get('workflow_runs') or []


def ecb_check(today):
    """should_build's freshness question. None if it could not be asked."""
    try:
        from scripts.should_build import ecb_freshness
        return ecb_freshness(today)
    except Exception as e:  # never let the diagnosis crash the report
        return None, 'ECB check failed: %s' % e


def seed_from_pages_live(history_dir, today):
    """First run only: archive the pages-live mirror's results if older."""
    try:
        out = subprocess.run(
            ['git', '-C', REPO_ROOT, 'show',
             'origin/pages-live:docs/results.json'],
            capture_output=True, text=True, timeout=30)
        payload = json.loads(out.stdout) if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, ValueError):
        payload = None
    day = (payload or {}).get('date')
    if not day or day >= today:
        return None
    archive(history_dir, day, payload, None)
    return day


def archive(history_dir, day, results, manifest):
    d = os.path.join(history_dir, day)
    os.makedirs(d, exist_ok=True)
    for name, obj in (('results.json', results), ('manifest.json', manifest)):
        if obj is None:
            continue
        tmp = os.path.join(d, name + '.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(obj, f, indent=1)
        os.replace(tmp, os.path.join(d, name))


def prune(history_dir, today, keep_days=HISTORY_KEEP_DAYS):
    cutoff = (date.fromisoformat(today) - timedelta(days=keep_days)).isoformat()
    for n in os.listdir(history_dir):
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}', n) and n < cutoff:
            shutil.rmtree(os.path.join(history_dir, n), ignore_errors=True)


def load_json(path):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--site', help='Pages base URL (default: from git origin)')
    ap.add_argument('--home', default=os.environ.get('CURRENCY_CHECK_HOME',
                                                     DEFAULT_HOME),
                    help='work directory: history/, report.md, status.json')
    ap.add_argument('--wait-min', type=float, default=110,
                    help='minutes to wait for today\'s build to land')
    ap.add_argument('--poll-min', type=float, default=10)
    ap.add_argument('--today', help='YYYY-MM-DD (default: today in New York)')
    args = ap.parse_args(argv)

    today = args.today or datetime.now(TZ).date().isoformat()
    slug = parse_remote(git_remote()) or parse_remote(FALLBACK_REMOTE)
    site = (args.site or site_url(slug)).rstrip('/') + '/'
    history = os.path.join(args.home, 'history')
    os.makedirs(history, exist_ok=True)

    deadline = time.monotonic() + args.wait_min * 60
    ecb = None
    while True:
        m_status, manifest = get_json(site + 'manifest.json')
        landed = (manifest or {}).get('run_date') == today
        if landed or m_status == 404:
            break
        if ecb is None:
            ecb = ecb_check(today)
            if ecb[0] is False:
                break  # nothing new to publish today; no point waiting
        if time.monotonic() >= deadline:
            break
        print('check_site: site shows %s, waiting for %s'
              % ((manifest or {}).get('run_date'), today), flush=True)
        time.sleep(args.poll_min * 60)

    page_status, body = http_get(site)
    page_kind = classify_page(page_status, len(body))
    runs = runs_today(scheduled_runs(slug), today)

    ctx = {'today': today, 'site': site, 'page_status': page_status,
           'page_size': len(body), 'page_kind': page_kind, 'runs': runs,
           'site_run_date': (manifest or {}).get('run_date'),
           'results': None, 'issues': []}
    statuses, reasons = [], []

    if page_kind in ('missing', 'readme') or m_status == 404:
        statuses.append('SITE-WRONG')
        reasons.append('index.html is %s (HTTP %s, %d bytes)%s' % (
            'a README-sized page — Pages is serving a branch, not the '
            'Actions artifact' if page_kind == 'readme' else 'not served',
            page_status, len(body),
            '; manifest.json 404' if m_status == 404 else ''))
    elif page_kind == 'odd':
        ctx['issues'].append('index.html is %d bytes, outside the usual '
                             '%d-%d' % (len(body), PAGE_MIN_BYTES,
                                        PAGE_MAX_BYTES))

    if landed:
        _, results = get_json(site + 'results.json')
        if results is None:
            statuses.append('SITE-WRONG')
            reasons.append('manifest is current but results.json is unreadable')
        else:
            if previous_day(history, today) is None:
                seed_from_pages_live(history, today)
            prev_date = previous_day(history, today)
            archive(history, today, results, manifest)
            prune(history, today)
            ctx.update(results=results, top=top_n(results),
                       jpy=jpy_valuation(results), prev_date=prev_date,
                       diff=(diff_results(load_json(os.path.join(
                           history, prev_date, 'results.json')), results)
                           if prev_date else None))
            ctx['issues'] += data_quality(manifest)
            if ctx['jpy'][1]:
                statuses.append('JPY-INVERTED')
                reasons.append('JPY Valuation %.1f' % ctx['jpy'][0])
            statuses.append('OK')
            reasons.append('today\'s build is live')
    elif m_status != 404:
        if ecb is None:
            ecb = ecb_check(today)
        st = not_landed_status(ecb[0], runs)
        statuses.append(st)
        reasons.append({
            'SKIPPED': 'no build due: %s' % ecb[1],
            'FAILED': 'today\'s scheduled build failed and nothing is '
                      'still running',
            'LATE': 'site still shows %s after waiting %g min (%s)' % (
                ctx['site_run_date'], args.wait_min,
                'no scheduled run started today yet' if not runs
                else 'run queued, running, or skipped by the gate'),
        }[st])

    ctx['status'] = worst(statuses)
    ctx['reason'] = '; '.join(reasons)
    report = render_report(ctx)
    with open(os.path.join(args.home, 'report.md'), 'w', encoding='utf-8') as f:
        f.write(report)
    with open(os.path.join(args.home, 'status.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'date': today, 'result': ctx['status'],
                   'reason': ctx['reason'], 'site': site,
                   'checked_at': datetime.now(TZ).isoformat(timespec='seconds')},
                  f, indent=1)
    print(report)
    return 0 if ctx['status'] in OK_STATUSES else 1


if __name__ == '__main__':
    raise SystemExit(main())
