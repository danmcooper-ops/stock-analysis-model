"""Tests for scripts/should_build.py — the DST gate on scheduled workflows.

The gate exists because Actions cron is UTC-only, so the interesting cases
are the two weekends a year when the correct cron swaps. These pin both
sides of each 2026 transition (spring forward March 8, fall back November 1)
and assert the property that actually matters: across a whole year, exactly
one of the registered crons builds on any given day.
"""

from datetime import datetime, timedelta, timezone

import pytest

from scripts.should_build import parse_cron_hm, should_build, wanted_utc

# 16:00 America/New_York, one cron per offset the zone uses.
EDT_CRON = '0 20 * * 1-5'
EST_CRON = '0 21 * * 1-5'
TZ = 'America/New_York'


def fired_at(cron, day):
    """The UTC instant `cron` nominally fires on `day`."""
    minute, hour = parse_cron_hm(cron)
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=timezone.utc)


class TestParseCronHm:
    def test_reads_minute_and_hour(self):
        assert parse_cron_hm('0 20 * * 1-5') == (0, 20)
        assert parse_cron_hm('30 7 * * *') == (30, 7)

    def test_wrong_field_count_rejected(self):
        with pytest.raises(ValueError, match='5-field cron'):
            parse_cron_hm('0 20 * *')
        with pytest.raises(ValueError, match='5-field cron'):
            parse_cron_hm('')

    def test_non_literal_minute_or_hour_rejected(self):
        with pytest.raises(ValueError, match='minute field'):
            parse_cron_hm('*/15 20 * * 1-5')
        with pytest.raises(ValueError, match='hour field'):
            parse_cron_hm('0 * * * 1-5')
        with pytest.raises(ValueError, match='hour field'):
            parse_cron_hm('0 20,21 * * 1-5')

    def test_out_of_range_rejected(self):
        with pytest.raises(ValueError, match='hour 25 out of range'):
            parse_cron_hm('0 25 * * *')
        with pytest.raises(ValueError, match='minute 61 out of range'):
            parse_cron_hm('61 5 * * *')


class TestWantedUtc:
    """The substitution must happen in the target zone, not in UTC.

    Doing it on the UTC-aware datetime the caller passes in returns the
    target hour unchanged, which silently makes every cron look wrong.
    """

    def test_edt_target_resolves_to_20_utc(self):
        now = datetime(2026, 6, 15, 20, 0, tzinfo=timezone.utc)
        assert wanted_utc(TZ, 16, now).hour == 20

    def test_est_target_resolves_to_21_utc(self):
        now = datetime(2026, 1, 12, 21, 0, tzinfo=timezone.utc)
        assert wanted_utc(TZ, 16, now).hour == 21

    def test_answer_does_not_depend_on_which_twin_asked(self):
        # Both firings on the same day must agree on what the right one is.
        day = datetime(2026, 6, 15)
        assert wanted_utc(TZ, 16, fired_at(EDT_CRON, day)) == wanted_utc(TZ, 16, fired_at(EST_CRON, day))

    def test_utc_input_is_not_returned_unchanged(self):
        # Regression: .replace(hour=16) on a UTC datetime yields 16:00 UTC.
        now = datetime(2026, 6, 15, 20, 0, tzinfo=timezone.utc)
        assert wanted_utc(TZ, 16, now).hour != 16

    def test_half_hour_zone_keeps_the_minute(self):
        now = datetime(2026, 6, 15, 10, 30, tzinfo=timezone.utc)
        want = wanted_utc('Asia/Kolkata', 16, now)
        assert (want.hour, want.minute) == (10, 30)


class TestShouldBuildAcrossDst:
    @pytest.mark.parametrize('day, builds, skips', [
        # Friday before spring forward (2026-03-08) — still EST.
        (datetime(2026, 3, 6), EST_CRON, EDT_CRON),
        # Monday after — EDT.
        (datetime(2026, 3, 9), EDT_CRON, EST_CRON),
        # Friday before fall back (2026-11-01) — still EDT.
        (datetime(2026, 10, 30), EDT_CRON, EST_CRON),
        # Monday after — EST.
        (datetime(2026, 11, 2), EST_CRON, EDT_CRON),
        # Deep summer and deep winter.
        (datetime(2026, 6, 15), EDT_CRON, EST_CRON),
        (datetime(2026, 1, 12), EST_CRON, EDT_CRON),
    ])
    def test_correct_cron_wins_on_each_side_of_a_transition(self, day, builds, skips):
        assert should_build(builds, TZ, 16, now=fired_at(builds, day))[0] is True
        assert should_build(skips, TZ, 16, now=fired_at(skips, day))[0] is False

    def test_exactly_one_cron_builds_every_weekday_of_2026(self):
        day = datetime(2026, 1, 1)
        while day.year == 2026:
            if day.weekday() < 5:
                built = [c for c in (EDT_CRON, EST_CRON)
                         if should_build(c, TZ, 16, now=fired_at(c, day))[0]]
                assert built == [EDT_CRON] or built == [EST_CRON], f'{day:%Y-%m-%d} built {built}'
            day += timedelta(days=1)

    def test_reason_names_the_local_time_and_abbreviation(self):
        _, reason = should_build(EDT_CRON, TZ, 16, now=fired_at(EDT_CRON, datetime(2026, 6, 15)))
        assert '16:00 EDT' in reason
        _, reason = should_build(EST_CRON, TZ, 16, now=fired_at(EST_CRON, datetime(2026, 1, 12)))
        assert '16:00 EST' in reason

    def test_skip_reason_names_the_twin_and_the_right_utc_time(self):
        _, reason = should_build(EST_CRON, TZ, 16, now=fired_at(EST_CRON, datetime(2026, 6, 15)))
        assert 'off-DST twin' in reason
        assert '20:00 UTC' in reason


class TestShouldBuildOtherZones:
    def test_southern_hemisphere_dst_runs_the_other_way(self):
        # Sydney is UTC+11 in January and UTC+10 in July, so the cron that
        # hits 09:00 local swaps in the opposite season to New York's.
        jan = should_build('0 22 * * *', 'Australia/Sydney', 9,
                           now=datetime(2026, 1, 14, 22, 0, tzinfo=timezone.utc))
        jul = should_build('0 23 * * *', 'Australia/Sydney', 9,
                           now=datetime(2026, 7, 15, 23, 0, tzinfo=timezone.utc))
        assert jan[0] is True
        assert jul[0] is True

    def test_half_hour_offset_zone(self):
        # Kolkata is UTC+5:30 year-round; 10:30 UTC is 16:00 local.
        build, _ = should_build('30 10 * * *', 'Asia/Kolkata', 16,
                                now=datetime(2026, 6, 15, 10, 30, tzinfo=timezone.utc))
        assert build is True

    def test_half_hour_zone_rejects_the_right_hour_at_the_wrong_minute(self):
        # 10:00 UTC is 15:30 in Kolkata, not 16:00 — an hour-only comparison
        # would wave this through.
        build, _ = should_build('0 10 * * *', 'Asia/Kolkata', 16,
                                now=datetime(2026, 6, 15, 10, 0, tzinfo=timezone.utc))
        assert build is False


class TestNonScheduledRuns:
    @pytest.mark.parametrize('cron', ['', '   '])
    def test_empty_cron_always_builds(self, cron):
        build, reason = should_build(cron, TZ, 16, now=datetime(2026, 6, 15, 3, 0, tzinfo=timezone.utc))
        assert build is True
        assert 'not a scheduled run' in reason


class TestMain:
    def _run(self, monkeypatch, tmp_path, capsys, argv, with_output=True):
        out = tmp_path / 'gh_output'
        monkeypatch.setattr('sys.argv', ['should_build.py'] + argv)
        if with_output:
            monkeypatch.setenv('GITHUB_OUTPUT', str(out))
        else:
            monkeypatch.delenv('GITHUB_OUTPUT', raising=False)
        from scripts.should_build import main
        main()
        return capsys.readouterr().out, (out.read_text(encoding='utf-8') if out.exists() else '')

    def test_writes_build_flag_to_github_output(self, monkeypatch, tmp_path, capsys):
        # Pin "now" so the verdict does not depend on when the suite runs.
        monkeypatch.setattr('scripts.should_build.should_build',
                            lambda *a, **k: (True, 'pinned'))
        stdout, written = self._run(monkeypatch, tmp_path, capsys,
                                    ['--tz', TZ, '--hour', '16', '--cron', EDT_CRON])
        assert 'pinned' in stdout
        assert written == 'build=true\n'

    def test_writes_false_when_skipping(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr('scripts.should_build.should_build',
                            lambda *a, **k: (False, 'pinned'))
        _, written = self._run(monkeypatch, tmp_path, capsys,
                               ['--tz', TZ, '--hour', '16', '--cron', EST_CRON])
        assert written == 'build=false\n'

    def test_no_github_output_env_is_not_an_error(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr('scripts.should_build.should_build',
                            lambda *a, **k: (True, 'pinned'))
        stdout, _ = self._run(monkeypatch, tmp_path, capsys,
                              ['--tz', TZ, '--hour', '16', '--cron', EDT_CRON], with_output=False)
        assert 'pinned' in stdout

    def test_bad_cron_exits_non_zero(self, monkeypatch, tmp_path, capsys):
        with pytest.raises(SystemExit) as e:
            self._run(monkeypatch, tmp_path, capsys,
                      ['--tz', TZ, '--hour', '16', '--cron', '0 * * * *'])
        assert 'hour field' in str(e.value)

    def test_unknown_timezone_exits_non_zero(self, monkeypatch, tmp_path, capsys):
        with pytest.raises(SystemExit) as e:
            self._run(monkeypatch, tmp_path, capsys,
                      ['--tz', 'Mars/Olympus_Mons', '--hour', '16', '--cron', EDT_CRON])
        assert 'unknown timezone' in str(e.value)

    def test_out_of_range_hour_exits_non_zero(self, monkeypatch, tmp_path, capsys):
        with pytest.raises(SystemExit) as e:
            self._run(monkeypatch, tmp_path, capsys,
                      ['--tz', TZ, '--hour', '24', '--cron', EDT_CRON])
        assert '--hour must be 0-23' in str(e.value)
