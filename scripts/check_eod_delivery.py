# scripts/check_eod_delivery.py
"""Alert when the nightly EOD run stops landing snapshots on data/snapshots.

The daily routine runs on one machine and pushes `results_<date>.json.gz` to
the `data/snapshots` branch. When it does not run, nothing anywhere says so:
the branch simply gains no commit, the report keeps serving the previous day,
and the gap is only noticed later as a hole in the backtest corpus. Over the
21 days to 2026-09-05 five of fifteen weekdays were missing this way, and the
2026-08-11 snapshot was lost outright without an alert.

This reads the branch listing from the GitHub API and fails when the newest
snapshot is staler than --max-age-days. Staleness is deliberately used
instead of "every weekday must be present": market holidays would otherwise
raise a false alarm every few weeks, and a run that skips one day but keeps
going is a different (smaller) problem than a run that has stopped.

`--kind weekly` applies the same check to the Sunday backtest: it reads
`backtest_summary_<date>.json` instead and allows 8 days, so one Sunday that
produced nothing goes red on the Monday after it. The missing-weekday table
does not apply to a weekly job and is skipped.

Usage:
    python scripts/check_eod_delivery.py --repo owner/name
    python scripts/check_eod_delivery.py --repo owner/name --max-age-days 4
    python scripts/check_eod_delivery.py --repo owner/name --kind weekly

A GitHub token is read from $GITHUB_TOKEN when present; the endpoint is
public, so a token only raises the rate limit. Exit status is 0 when the
archive is fresh, 1 when it is stale, and 2 when the branch could not be
read at all -- a network or permissions failure must not read as "the
pipeline is fine".
"""
import argparse
import datetime
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

# Both forms are archived: plain .json predates the gzip change, .json.gz is
# what the routine writes now. A mixed archive is expected and supported.
_SNAPSHOT_RE = re.compile(r'^results_(\d{4}-\d{2}-\d{2})\.json(\.gz)?$')

# A Friday run seen on the Tuesday after a Monday holiday is 4 days old and
# healthy. Anything past that means at least one ordinary weekday vanished.
_DEFAULT_MAX_AGE_DAYS = 4

# The weekly backtest commits `backtest_summary_<date>.json` each Sunday.
_WEEKLY_RE = re.compile(r'^backtest_summary_(\d{4}-\d{2}-\d{2})\.json$')

# kind -> (filename pattern, default max age, what a file is, its plural, what stale means)
KINDS = {
    'snapshot': (_SNAPSHOT_RE, _DEFAULT_MAX_AGE_DAYS, 'snapshot', 'snapshots',
                 'the nightly run has stopped landing snapshots'),
    # A Sunday summary seen on the Monday after the NEXT Sunday is 8 days old:
    # that Sunday has produced nothing, which is what this is meant to catch.
    'weekly': (_WEEKLY_RE, 8, 'backtest summary', 'backtest summaries',
               'the weekly backtest has stopped landing summaries'),
}


def snapshot_dates(names, pattern=_SNAPSHOT_RE):
    """Parse `results_<date>.json[.gz]` filenames (or *pattern*'s) into a sorted date list."""
    dates = set()
    for name in names:
        m = pattern.match(name)
        if m:
            try:
                dates.add(datetime.date.fromisoformat(m.group(1)))
            except ValueError:
                continue  # results_2026-13-45.json and friends
    return sorted(dates)


def missing_weekdays(dates, today, lookback_days):
    """Weekdays in the lookback window with no snapshot. Informational only.

    Market holidays land here too, which is precisely why this does not gate
    the exit status -- it is a table for a human to read, not a test.
    """
    have = set(dates)
    out = []
    for i in range(lookback_days, 0, -1):
        day = today - datetime.timedelta(days=i)
        if day.weekday() < 5 and day not in have:
            out.append(day)
    return out


def verdict(dates, today, max_age_days=_DEFAULT_MAX_AGE_DAYS, kind='snapshot'):
    """Return (ok, age_days, reason) for the archive's freshness."""
    _, _, noun, nouns, stopped = KINDS[kind]
    if not dates:
        return False, None, f'no {nouns} found on the branch at all'
    newest = dates[-1]
    age = (today - newest).days
    if age > max_age_days:
        return False, age, (f'newest {noun} is {newest} ({age} days old, limit {max_age_days}) '
                            f'-- {stopped}')
    return True, age, f'newest {noun} is {newest} ({age} days old)'


def fetch_branch_filenames(repo, branch, token=None, timeout=30):
    """Root-level filenames on `branch` via the GitHub contents API."""
    url = f'https://api.github.com/repos/{repo}/contents/?ref={urllib.parse.quote(branch, safe="")}'
    req = urllib.request.Request(url, headers={
        'Accept': 'application/vnd.github+json',
        'User-Agent': 'check-eod-delivery',
    })
    if token:
        req.add_header('Authorization', f'Bearer {token}')
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.load(resp)
    if not isinstance(payload, list):
        raise ValueError(f'unexpected API response for {repo}@{branch}: not a directory listing')
    return [entry.get('name', '') for entry in payload]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--repo', required=True, help='owner/name of the GitHub repository')
    ap.add_argument('--branch', default='data/snapshots', help='branch holding the archive')
    ap.add_argument('--kind', choices=sorted(KINDS), default='snapshot',
                    help='nightly snapshots (default) or weekly backtest summaries')
    ap.add_argument('--max-age-days', type=int, default=None,
                    help=f'fail when the newest file is older than this (default {_DEFAULT_MAX_AGE_DAYS} '
                         f'for snapshot, {KINDS["weekly"][1]} for weekly)')
    ap.add_argument('--lookback-days', type=int, default=21,
                    help='window for the informational missing-weekday table (default 21)')
    args = ap.parse_args(argv)

    try:
        names = fetch_branch_filenames(args.repo, args.branch, os.environ.get('GITHUB_TOKEN'))
    except (urllib.error.URLError, ValueError, OSError) as e:
        # Exit 2, never 0 and never 1: an unreadable branch is an unknown, and
        # it must not be mistaken for the stale verdict that exit 1 carries.
        print(f'check_eod_delivery: could not read {args.repo}@{args.branch}: {e}', file=sys.stderr)
        return 2

    pattern, default_age, _, nouns, _ = KINDS[args.kind]
    max_age = default_age if args.max_age_days is None else args.max_age_days
    today = datetime.date.today()
    dates = snapshot_dates(names, pattern)
    ok, _, reason = verdict(dates, today, max_age, args.kind)

    print(f'{"OK" if ok else "STALE"}: {reason}')
    print(f'archive holds {len(dates)} {nouns}')

    gaps = missing_weekdays(dates, today, args.lookback_days) if args.kind == 'snapshot' else []
    if gaps:
        print(f'\nweekdays with no snapshot in the last {args.lookback_days} days '
              f'({len(gaps)}, market holidays included):')
        for day in gaps:
            print(f'  {day:%a %Y-%m-%d}')
        # An isolated missed day does not trip the staleness gate -- the run
        # recovers the next morning and the newest snapshot looks fine. It is
        # still a permanent hole in the backtest corpus, so raise it where a
        # green check cannot hide it.
        if os.environ.get('GITHUB_ACTIONS'):
            days = ', '.join(f'{d:%Y-%m-%d}' for d in gaps)
            print(f'::warning title=EOD snapshots missing::{len(gaps)} weekday(s) '
                  f'with no snapshot in the last {args.lookback_days} days: {days}')

    github_output = os.environ.get('GITHUB_OUTPUT')
    if github_output:
        with open(github_output, 'a', encoding='utf-8') as f:
            f.write(f'fresh={"true" if ok else "false"}\n')
            f.write(f'newest={dates[-1] if dates else ""}\n')
            f.write(f'missing_weekdays={len(gaps)}\n')

    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
