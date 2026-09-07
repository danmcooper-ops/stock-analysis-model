"""Tests for scripts/market_open.py — the trading-day gate on the daily run.

The gate's job is to stop a 3-6 hour pipeline from running on a day that
produced no bars, so the cases worth pinning are the ones where a plain
"is it a weekday" check gets it wrong: the moving holidays, the weekend
observance shifts, and the two exceptions (New Year's Day on a Saturday,
Juneteenth before 2022).

The holiday dates below are the published NYSE calendars, not this module's
output, so a refactor that breaks the arithmetic fails here.
"""

from datetime import date, timedelta

import pytest

from scripts.market_open import (
    AD_HOC_CLOSURES,
    EXIT_CLOSED,
    EXIT_OPEN,
    easter,
    last_weekday,
    market_status,
    nth_weekday,
    nyse_holidays,
    observed,
)

MONDAY, THURSDAY, SATURDAY = 0, 3, 5

# Published NYSE full-day closures, straight from the exchange calendars.
NYSE_2025 = {
    date(2025, 1, 1): "New Year's Day",
    date(2025, 1, 9): 'national day of mourning (Jimmy Carter)',
    date(2025, 1, 20): 'Martin Luther King Jr. Day',
    date(2025, 2, 17): "Washington's Birthday",
    date(2025, 4, 18): 'Good Friday',
    date(2025, 5, 26): 'Memorial Day',
    date(2025, 6, 19): 'Juneteenth',
    date(2025, 7, 4): 'Independence Day',
    date(2025, 9, 1): 'Labor Day',
    date(2025, 11, 27): 'Thanksgiving Day',
    date(2025, 12, 25): 'Christmas Day',
}

NYSE_2026 = {
    date(2026, 1, 1): "New Year's Day",
    date(2026, 1, 19): 'Martin Luther King Jr. Day',
    date(2026, 2, 16): "Washington's Birthday",
    date(2026, 4, 3): 'Good Friday',
    date(2026, 5, 25): 'Memorial Day',
    date(2026, 6, 19): 'Juneteenth',
    date(2026, 7, 3): 'Independence Day',      # Jul 4 is a Saturday
    date(2026, 9, 7): 'Labor Day',
    date(2026, 11, 26): 'Thanksgiving Day',
    date(2026, 12, 25): 'Christmas Day',
}

NYSE_2027 = {
    date(2027, 1, 1): "New Year's Day",
    date(2027, 1, 18): 'Martin Luther King Jr. Day',
    date(2027, 2, 15): "Washington's Birthday",
    date(2027, 3, 26): 'Good Friday',
    date(2027, 5, 31): 'Memorial Day',
    date(2027, 6, 18): 'Juneteenth',           # Jun 19 is a Saturday
    date(2027, 7, 5): 'Independence Day',      # Jul 4 is a Sunday
    date(2027, 9, 6): 'Labor Day',
    date(2027, 11, 25): 'Thanksgiving Day',
    date(2027, 12, 24): 'Christmas Day',       # Dec 25 is a Saturday
}


class TestPublishedCalendars:
    @pytest.mark.parametrize('year, expected', [(2025, NYSE_2025), (2026, NYSE_2026), (2027, NYSE_2027)])
    def test_matches_the_exchange_calendar_exactly(self, year, expected):
        assert nyse_holidays(year) == expected

    def test_no_holiday_ever_lands_on_a_weekend(self):
        # An observed closure on a Saturday would mean the shift rule is wrong.
        for year in range(1998, 2041):
            for day in nyse_holidays(year):
                assert day.weekday() < SATURDAY, f'{day:%Y-%m-%d} is a {day:%A}'

    def test_holiday_count_matches_the_rules_every_year(self):
        # Nine fixed + moving holidays before Juneteenth, ten from 2022 — less
        # one in a year New Year's Day falls on a Saturday and is not observed
        # (2000, 2005, 2011, 2022), plus any ad-hoc closure.
        for year in range(1998, 2041):
            expected = 10 if year >= 2022 else 9
            if date(year, 1, 1).weekday() == SATURDAY:
                expected -= 1
            expected += sum(1 for d in AD_HOC_CLOSURES if d.year == year)
            assert len(nyse_holidays(year)) == expected, year

    def test_the_new_years_saturday_years_really_are_short(self):
        # Guards the test above from passing by adjusting to a broken calendar:
        # 2000 had eight NYSE closures, and that is the published number.
        assert len(nyse_holidays(2000)) == 8
        assert "New Year's Day" not in nyse_holidays(2000).values()


class TestWeekendObservance:
    def test_saturday_holiday_moves_back_to_friday(self):
        assert observed(date(2026, 7, 4)) == date(2026, 7, 3)

    def test_sunday_holiday_moves_forward_to_monday(self):
        assert observed(date(2027, 7, 4)) == date(2027, 7, 5)

    def test_weekday_holiday_is_untouched(self):
        assert observed(date(2025, 7, 4)) == date(2025, 7, 4)

    def test_new_years_day_on_a_saturday_is_not_observed(self):
        # NYSE Rule 7.2's exception: the exchange does not close on Dec 31.
        assert observed(date(2022, 1, 1), shift_saturday=False) is None
        assert date(2021, 12, 31) not in nyse_holidays(2021)
        assert market_status(date(2021, 12, 31))[0] is True

    def test_new_years_day_on_a_sunday_still_moves_to_monday(self):
        assert date(2023, 1, 2) in nyse_holidays(2023)

    def test_observance_never_leaks_into_an_adjacent_year(self):
        # Jan 1 on a Sunday observes into Jan 2, which is still in-year; the
        # guard matters for a Dec 25 Saturday, whose Friday shift stays in-year.
        for year in range(1998, 2041):
            for day in nyse_holidays(year):
                assert day.year == year


class TestJuneteenth:
    def test_absent_before_the_first_nyse_observance(self):
        # A federal holiday from 2021, but the exchange first closed in 2022.
        assert 'Juneteenth' not in nyse_holidays(2021).values()
        assert market_status(date(2021, 6, 18))[0] is True

    def test_present_from_2022(self):
        assert nyse_holidays(2022)[date(2022, 6, 20)] == 'Juneteenth'  # Jun 19 was a Sunday


class TestEaster:
    @pytest.mark.parametrize('year, expected', [
        (2024, date(2024, 3, 31)),
        (2025, date(2025, 4, 20)),
        (2026, date(2026, 4, 5)),
        (2027, date(2027, 3, 28)),
        (2038, date(2038, 4, 25)),   # latest possible date
    ])
    def test_known_easter_sundays(self, year, expected):
        assert easter(year) == expected

    def test_good_friday_is_always_two_days_before_easter(self):
        for year in range(1998, 2041):
            good_friday = easter(year) - timedelta(days=2)
            assert good_friday.weekday() == 4
            assert nyse_holidays(year)[good_friday] == 'Good Friday'


class TestWeekdayHelpers:
    def test_nth_weekday_when_the_month_starts_on_the_target(self):
        # Sep 2026 starts on a Tuesday; Labor Day is the 7th.
        assert nth_weekday(2026, 9, MONDAY, 1) == date(2026, 9, 7)
        # Jun 2025 starts on a Sunday, so the first Monday is the 2nd.
        assert nth_weekday(2025, 6, MONDAY, 1) == date(2025, 6, 2)

    def test_nth_weekday_fourth_thursday(self):
        assert nth_weekday(2026, 11, THURSDAY, 4) == date(2026, 11, 26)

    def test_last_weekday_of_may(self):
        assert last_weekday(2026, 5, MONDAY) == date(2026, 5, 25)
        assert last_weekday(2027, 5, MONDAY) == date(2027, 5, 31)

    def test_last_weekday_of_december_does_not_overflow_the_year(self):
        # The December branch exists because month + 1 is not a valid date.
        assert last_weekday(2026, 12, MONDAY) == date(2026, 12, 28)


class TestMarketStatus:
    @pytest.mark.parametrize('day, reason_fragment', [
        (date(2026, 9, 7), 'Labor Day'),
        (date(2026, 7, 3), 'Independence Day'),
        (date(2026, 11, 26), 'Thanksgiving Day'),
        (date(2026, 4, 3), 'Good Friday'),
        (date(2025, 1, 9), 'Jimmy Carter'),
    ])
    def test_holidays_are_closed_and_named(self, day, reason_fragment):
        is_open, reason = market_status(day)
        assert is_open is False
        assert reason_fragment in reason

    @pytest.mark.parametrize('day', [date(2026, 9, 5), date(2026, 9, 6)])
    def test_weekends_are_closed(self, day):
        is_open, reason = market_status(day)
        assert is_open is False
        assert 'weekend' in reason

    @pytest.mark.parametrize('day', [
        date(2026, 9, 8),    # ordinary Tuesday
        date(2026, 9, 4),    # Friday before Labor Day
        date(2026, 11, 27),  # day after Thanksgiving — early close, still trades
        date(2026, 12, 24),  # Christmas Eve — early close, still trades
    ])
    def test_trading_sessions_are_open(self, day):
        assert market_status(day)[0] is True

    def test_year_before_the_supported_range_is_rejected(self):
        # Loud failure beats a wrong answer for a year the rules differ in.
        with pytest.raises(ValueError, match='predates this calendar'):
            market_status(date(1990, 5, 1))


class TestMain:
    def _run(self, monkeypatch, capsys, argv):
        monkeypatch.setattr('sys.argv', ['market_open.py'] + argv)
        from scripts.market_open import main
        with pytest.raises(SystemExit) as e:
            main()
        return e.value.code, capsys.readouterr().out

    def test_open_day_exits_zero(self, monkeypatch, capsys):
        code, out = self._run(monkeypatch, capsys, ['--date', '2026-09-08'])
        assert code == EXIT_OPEN
        assert 'market was open' in out

    def test_closed_day_exits_with_the_skip_code(self, monkeypatch, capsys):
        code, out = self._run(monkeypatch, capsys, ['--date', '2026-09-07'])
        assert code == EXIT_CLOSED
        assert 'Labor Day' in out

    def test_skip_code_is_distinct_from_a_gate_failure(self, monkeypatch, capsys):
        # The caller keys off 10 exactly; if a crash also meant "skip", one bug
        # would silently stop the daily corpus growing.
        assert EXIT_CLOSED not in (0, 1, 2)

    def test_unparseable_date_is_a_gate_failure_not_a_skip(self, monkeypatch, capsys):
        code, _ = self._run(monkeypatch, capsys, ['--date', 'notadate'])
        assert code != EXIT_CLOSED
        assert 'YYYY-MM-DD' in str(code)

    def test_unsupported_year_is_a_gate_failure_not_a_skip(self, monkeypatch, capsys):
        code, _ = self._run(monkeypatch, capsys, ['--date', '1990-05-01'])
        assert code != EXIT_CLOSED
        assert 'predates this calendar' in str(code)

    def test_no_date_argument_uses_the_exchange_timezone(self, monkeypatch, capsys):
        monkeypatch.setattr('scripts.market_open.today_in_exchange_tz', lambda: date(2026, 9, 7))
        code, out = self._run(monkeypatch, capsys, [])
        assert code == EXIT_CLOSED
        assert '2026-09-07' in out
