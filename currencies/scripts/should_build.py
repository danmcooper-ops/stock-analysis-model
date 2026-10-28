"""Decide whether a scheduled build should proceed.

    python scripts/should_build.py --tz America/New_York --hour 16

Writes `proceed=true|false` and a `reason=` line in GitHub Actions output
format, so the workflow can gate its steps with an `if:` rather than failing
the job — a deliberate skip is not an error and should not show a red X.

Three independent checks.

**Local hour.** GitHub cron is fixed-UTC, but 16:15 America/New_York is
20:15 UTC under EDT and 21:15 UTC under EST. The workflow therefore registers
both crons and this check discards any run that starts before 16:00 local —
under EST that is the 20:15 UTC cron, which lands at 15:15.

The hour is a floor, not a window. GitHub routinely starts scheduled runs
late, and on this repository the delay grew from ~30 minutes to two or more
hours. An earlier version required the run to land *inside* the 16:00 hour,
so from 2026-08-26 every run was discarded as late and the site silently
stopped updating while every run showed green.

**Already built today.** Once both crons can pass the hour check (always,
under EDT), both would build. `--last-built` takes the Unix timestamp of the
last publish — the workflow reads it from the `pages-live` commit — and the
run is skipped if that falls on today's local date. A missing or unreadable
timestamp does not gate: a duplicate build republishes identical output,
while a wrong skip loses the day.

**Fresh ECB data.** The model's spine is the ECB daily reference rate series,
which is not published on TARGET holidays. Rather than maintain a holiday
calendar — which would need updating annually and would encode the wrong
country's calendar anyway — this asks the question that actually matters: did
today's rates publish? If the newest observation is not today's date, there is
nothing new to score, and the build is skipped.

That also catches outages, not just holidays: an ECB endpoint failure looks
identical to a holiday here, and in both cases rebuilding would republish
yesterday's numbers under today's date.
"""

import argparse
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import ecb_client


def emit(proceed, reason):
    """Write GitHub Actions outputs, and echo for the run log."""
    line_p = 'proceed=%s' % ('true' if proceed else 'false')
    line_r = 'reason=%s' % reason
    out = os.environ.get('GITHUB_OUTPUT')
    if out:
        with open(out, 'a', encoding='utf-8') as f:
            f.write(line_p + '\n' + line_r + '\n')
    print('%s  (%s)' % (line_p, reason))
    return 0


def _parse_timestamp(value, tz):
    """Unix seconds -> aware datetime in tz, or None if absent or unparseable."""
    try:
        return datetime.fromtimestamp(int(value), tz)
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tz', default='America/New_York',
                    help='IANA timezone the --hour check is evaluated in')
    ap.add_argument('--hour', type=int, default=None,
                    help='only proceed at or after this local hour (0-23); '
                         'omit to skip the time check entirely')
    ap.add_argument('--last-built', default=None,
                    help='Unix timestamp of the last publish; skip if it is '
                         'already today in --tz. Empty or invalid is ignored')
    ap.add_argument('--skip-freshness', action='store_true',
                    help='do not require a same-day ECB observation')
    args = ap.parse_args(argv)

    try:
        tz = ZoneInfo(args.tz)
    except Exception:
        return emit(True, 'unknown timezone %r, not gating on time' % args.tz)

    now = datetime.now(tz)

    if args.hour is not None and now.hour < args.hour:
        return emit(False, 'local time is %s in %s, before %02d:00'
                    % (now.strftime('%H:%M'), args.tz, args.hour))

    last = _parse_timestamp(args.last_built, tz)
    if last is not None and last.date() == now.date():
        return emit(False, 'already built today at %s'
                    % last.strftime('%H:%M %Z'))

    if args.skip_freshness:
        return emit(True, 'freshness check disabled')

    proceed, reason = ecb_freshness(now.date().isoformat())
    return emit(proceed, reason)


def ecb_freshness(today):
    """(proceed, reason): did the ECB publish reference rates dated `today`?

    Shared with scripts/check_site.py, which asks the same question to tell a
    correctly skipped day from a build that should have landed and did not.
    """
    per_eur = ecb_client.fetch_reference_rates(ttl_days=0.0)
    if not per_eur:
        # Distinct from a holiday: the source is unreachable. Skipping is
        # still right — there is nothing new to publish either way.
        return False, 'ECB reference rates unavailable'

    usd = per_eur.get('USD') or {}
    if not usd:
        return False, 'ECB responded without a USD series'

    latest = max(usd)
    if latest != today:
        return False, ('newest ECB observation is %s, not %s '
                       '(TARGET holiday, weekend, or not yet published)'
                       % (latest, today))

    return True, 'ECB published %s; building' % latest


if __name__ == '__main__':
    raise SystemExit(main())
