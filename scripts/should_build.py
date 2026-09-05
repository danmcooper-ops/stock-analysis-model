# scripts/should_build.py
"""Decide whether this scheduled run is the one that lands at a local hour.

GitHub Actions cron is UTC-only and has no timezone field, so a single
schedule drifts by an hour twice a year: `0 20 * * 1-5` is 16:00 in New York
from March to November and 15:00 for the rest of the year. The fix is to
register one cron per UTC offset the zone uses, let all of them fire, and
have the job's first step ask this script which firing is the real one for
today. The others exit early.

    on:
      schedule:
        - cron: '0 20 * * 1-5'   # 16:00 America/New_York while EDT
        - cron: '0 21 * * 1-5'   # 16:00 America/New_York while EST

Usage:
    python scripts/should_build.py --tz America/New_York --hour 16 --cron "0 20 * * 1-5"

Prints a one-line verdict and, when $GITHUB_OUTPUT is set, writes
`build=true` or `build=false` for later steps to gate on:

    - id: gate
      run: python scripts/should_build.py --tz America/New_York --hour 16 --cron "${{ github.event.schedule }}"
    - if: steps.gate.outputs.build == 'true'
      run: ...

An empty --cron means the run was not schedule-triggered (workflow_dispatch,
push), and those always build. Exit status is 0 for both verdicts — it is
non-zero only when the arguments are unusable, so a typo in a cron or a zone
name fails loudly instead of silently skipping every run forever.

The verdict asks what UTC time the target hour falls on today and compares
that to the cron, rather than converting the cron into local time. Both
directions agree for an afternoon target; they part company when the target
hour is itself near a transition, because a local wall time can be skipped
(spring forward) or happen twice (fall back), and `.replace()` on such a
time resolves through `fold` rather than failing. For a target hour within
an hour of a zone's transition point — 00:00 in America/Havana, say — read
the verdict on the transition day before trusting it.

Only the minute and hour fields are read. Day-of-week is left to Actions,
which evaluates it in UTC: that matches the local day only while the UTC
instant and the local one fall on the same date, which holds for a
late-afternoon US Eastern target but not for every zone and hour.
"""
import argparse
import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def parse_cron_hm(cron):
    """Return (minute, hour) as UTC ints from a 5-field cron expression."""
    fields = cron.split()
    if len(fields) != 5:
        raise ValueError(f'expected a 5-field cron expression, got {len(fields)} fields: {cron!r}')
    for name, field in (('minute', fields[0]), ('hour', fields[1])):
        if not field.isdigit():
            raise ValueError(
                f'{name} field {field!r} is not a plain number. This gate resolves one firing to '
                f'one wall-clock time, and a wildcard, list, range or step has no single time to '
                f'convert — give each firing its own cron line instead.')
    minute, hour = int(fields[0]), int(fields[1])
    if not 0 <= minute <= 59:
        raise ValueError(f'minute {minute} out of range 0-59: {cron!r}')
    if not 0 <= hour <= 23:
        raise ValueError(f'hour {hour} out of range 0-23: {cron!r}')
    return minute, hour


def wanted_utc(tz_name, target_hour, now=None):
    """The UTC time the target local hour falls on today — the cron that should win.

    `now` is converted into the target zone before the hour is substituted:
    the substitution has to happen on a datetime that already carries the
    zone, or ZoneInfo derives the offset from the wrong wall time and the
    result is just the target hour back again.
    """
    now = now or datetime.now(timezone.utc)
    local_today = (now.astimezone(ZoneInfo(tz_name))
                      .replace(hour=target_hour, minute=0, second=0, microsecond=0))
    return local_today.astimezone(timezone.utc)


def should_build(cron, tz_name, target_hour, now=None):
    """Return (build, reason) for a cron expression against a target local hour."""
    now = now or datetime.now(timezone.utc)
    zone = ZoneInfo(tz_name)

    if not cron.strip():
        return True, 'no schedule in the event context — not a scheduled run, building'

    minute, hour = parse_cron_hm(cron)
    want = wanted_utc(tz_name, target_hour, now)
    abbrev = now.astimezone(zone).strftime('%Z')

    if (hour, minute) == (want.hour, want.minute):
        return True, f'{cron!r} is {target_hour:02d}:00 {abbrev} in {tz_name} today, building'
    return False, (f'{cron!r} is the off-DST twin — {target_hour:02d}:00 {abbrev} in {tz_name} '
                   f'is {want:%H:%M} UTC today, skipping')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--tz', required=True,
                    help='IANA zone the target hour is expressed in (e.g. America/New_York)')
    ap.add_argument('--hour', type=int, required=True, help='target local hour, 0-23')
    ap.add_argument('--cron', default='',
                    help='the schedule that triggered this run (${{ github.event.schedule }}); '
                         'empty on workflow_dispatch and push, which always build')
    args = ap.parse_args()

    if not 0 <= args.hour <= 23:
        raise SystemExit(f'should_build: --hour must be 0-23, got {args.hour}')
    try:
        build, reason = should_build(args.cron, args.tz, args.hour)
    except ZoneInfoNotFoundError:
        raise SystemExit(f'should_build: unknown timezone {args.tz!r} '
                         f'(on a slim image the zone database may be missing: pip install tzdata)') from None
    except ValueError as e:
        raise SystemExit(f'should_build: {e}') from None

    print(reason)
    github_output = os.environ.get('GITHUB_OUTPUT')
    if github_output:
        with open(github_output, 'a', encoding='utf-8') as f:
            f.write(f'build={"true" if build else "false"}\n')


if __name__ == '__main__':
    main()
