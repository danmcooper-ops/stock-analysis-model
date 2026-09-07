# scripts/market_open.py
"""Decide whether the US equity market traded on a given day.

The daily analysis routine costs 3-6 hours and writes an ~85 MiB snapshot to
the `data/snapshots` archive. On a day the exchanges never opened there are no
new bars to fetch, so all of that work re-publishes the previous session's data
under a new date and puts a duplicate-content day into the corpus the weekly
backtest calibrates on. This gate is the cheap check that runs first.

    python scripts/market_open.py            # today, in America/New_York
    python scripts/market_open.py --date 2026-09-07

Exit status is the verdict, and the three outcomes are deliberately distinct:

    0   open — the market traded, run the pipeline
    10  closed — weekend, holiday or a known ad-hoc closure, skip the run
    1   the gate itself failed (bad argument, unusable date)

A caller must treat only 10 as "skip". Failing open on status 1 is the
important half: a bug in this file should cost one wasted run, not silently
stop the daily corpus from growing for weeks, which is the harm the whole
project is built to avoid.

The calendar is computed rather than tabulated, so it needs no data file and no
dependency beyond the standard library — it still answers correctly in a year
nobody has updated it for. It implements the modern NYSE schedule (Rule 7.2)
and is accurate from 1998, when MLK Day was first observed; earlier years are
rejected rather than answered wrongly.

Two rules in here look like bugs and are not:

* A holiday on a **Saturday** is observed the preceding Friday, but New Year's
  Day is exempt — when January 1 falls on a Saturday the exchange does not
  close on December 31. (Verified against 2021-12-31 and 2010-12-31, both
  ordinary sessions.) A holiday on a **Sunday** is observed the following
  Monday, New Year's Day included.
* **Juneteenth** is a federal holiday from 2021 but the NYSE first closed for
  it in 2022, so it is gated on the year.

Unscheduled closures — national days of mourning, hurricanes — cannot be
derived from a rule and live in AD_HOC_CLOSURES below. That table is the one
part of this file that goes stale: a future closure of that kind is not known
until it is announced, and the gate will wave the day through until someone
adds the date.
"""
import argparse
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

EXCHANGE_TZ = 'America/New_York'

#: Verdict exit codes. Anything else out of main() means the gate broke.
EXIT_OPEN = 0
EXIT_CLOSED = 10

#: The first year this calendar is valid for — MLK Day's first NYSE observance.
FIRST_SUPPORTED_YEAR = 1998

MONDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY = 0, 3, 4, 5, 6

#: Closures that no rule produces. Keys are the dates the exchange was shut.
AD_HOC_CLOSURES = {
    date(2012, 10, 29): 'Hurricane Sandy',
    date(2012, 10, 30): 'Hurricane Sandy',
    date(2018, 12, 5): 'national day of mourning (George H. W. Bush)',
    date(2025, 1, 9): 'national day of mourning (Jimmy Carter)',
}


def easter(year):
    """Easter Sunday in the Gregorian calendar (Meeus/Jones/Butcher)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    g = (8 * b + 13) // 25
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    lam = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 19 * lam) // 433
    month = (h + lam - 7 * m + 90) // 25
    day = (h + lam - 7 * m + 33 * month + 19) % 32
    return date(year, month, day)


def nth_weekday(year, month, weekday, n):
    """The nth `weekday` of a month, e.g. nth_weekday(2026, 9, MONDAY, 1)."""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def last_weekday(year, month, weekday):
    """The final `weekday` of a month — Memorial Day's rule."""
    if month == 12:
        last = date(year, 12, 31)
    else:
        last = date(year, month + 1, 1) - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def observed(day, shift_saturday=True):
    """Move a fixed-date holiday off a weekend the way the NYSE does.

    Saturday shifts back to Friday and Sunday forward to Monday, except for
    New Year's Day on a Saturday, which is not observed at all — the caller
    passes shift_saturday=False and gets None.
    """
    if day.weekday() == SATURDAY:
        return day - timedelta(days=1) if shift_saturday else None
    if day.weekday() == SUNDAY:
        return day + timedelta(days=1)
    return day


def nyse_holidays(year):
    """Every full-day NYSE closure in `year`, as {date: name}.

    Ad-hoc closures are included so a single call answers the whole question.
    Early-close sessions (the day after Thanksgiving, Christmas Eve) are not
    here: the market trades, bars are produced, and the run should happen.
    """
    if year < FIRST_SUPPORTED_YEAR:
        raise ValueError(
            f'{year} predates this calendar — the NYSE holiday schedule before '
            f'{FIRST_SUPPORTED_YEAR} differs (MLK Day was not observed) and is not modelled here')

    holidays = {}

    def add(day, name):
        if day is not None and day.year == year:
            holidays[day] = name

    add(observed(date(year, 1, 1), shift_saturday=False), "New Year's Day")
    add(nth_weekday(year, 1, MONDAY, 3), 'Martin Luther King Jr. Day')
    add(nth_weekday(year, 2, MONDAY, 3), "Washington's Birthday")
    add(easter(year) - timedelta(days=2), 'Good Friday')
    add(last_weekday(year, 5, MONDAY), 'Memorial Day')
    if year >= 2022:
        add(observed(date(year, 6, 19)), 'Juneteenth')
    add(observed(date(year, 7, 4)), 'Independence Day')
    add(nth_weekday(year, 9, MONDAY, 1), 'Labor Day')
    add(nth_weekday(year, 11, THURSDAY, 4), 'Thanksgiving Day')
    add(observed(date(year, 12, 25)), 'Christmas Day')

    for day, reason in AD_HOC_CLOSURES.items():
        add(day, reason)

    return holidays


def market_status(day):
    """Return (is_open, reason) for a single calendar date."""
    if day.weekday() >= SATURDAY:
        return False, f'{day:%Y-%m-%d} is a {day:%A} — the market is closed at weekends'

    holiday = nyse_holidays(day.year).get(day)
    if holiday:
        return False, f'{day:%Y-%m-%d} is {holiday} — the market is closed'

    return True, f'{day:%Y-%m-%d} is a {day:%A} trading session — the market was open'


def today_in_exchange_tz():
    """Today's date where the exchange is, not where this machine is."""
    return datetime.now(ZoneInfo(EXCHANGE_TZ)).date()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--date', dest='day',
                    help=f'date to test as YYYY-MM-DD (default: today in {EXCHANGE_TZ})')
    args = ap.parse_args()

    if args.day:
        try:
            day = date.fromisoformat(args.day)
        except ValueError:
            raise SystemExit(f'market_open: --date must be YYYY-MM-DD, got {args.day!r}') from None
    else:
        day = today_in_exchange_tz()

    try:
        is_open, reason = market_status(day)
    except ValueError as e:
        raise SystemExit(f'market_open: {e}') from None

    print(reason)
    raise SystemExit(EXIT_OPEN if is_open else EXIT_CLOSED)


if __name__ == '__main__':
    main()
