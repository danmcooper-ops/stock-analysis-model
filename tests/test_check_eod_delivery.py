"""Tests for scripts/check_eod_delivery.py — the alarm on a stopped nightly.

The cases are drawn from the real archive as of 2026-09-05, where five of
fifteen weekdays were missing and the newest snapshot was 2026-09-03. The
important properties are that a holiday-length gap stays quiet, an actual
stoppage does not, and an unreadable branch never reads as healthy.
"""

import datetime

from scripts.check_eod_delivery import (missing_weekdays, snapshot_dates, verdict)

D = datetime.date


class TestSnapshotDates:
    def test_reads_both_archive_forms(self):
        names = ['results_2026-09-03.json', 'results_2026-09-02.json.gz']
        assert snapshot_dates(names) == [D(2026, 9, 2), D(2026, 9, 3)]

    def test_ignores_everything_else(self):
        names = ['README.md', 'portfolio_report_2026-09-03.txt', 'results_2026-09-03.html',
                 'stock_analysis_results_2026-09-03.json', 'results_2026-09-03.json.bak']
        assert snapshot_dates(names) == []

    def test_returns_sorted_and_deduped(self):
        # A date present as both .json and .json.gz is one snapshot, not two.
        names = ['results_2026-09-03.json', 'results_2026-09-01.json', 'results_2026-09-03.json.gz']
        assert snapshot_dates(names) == [D(2026, 9, 1), D(2026, 9, 3)]

    def test_impossible_date_does_not_crash(self):
        assert snapshot_dates(['results_2026-13-45.json']) == []

    def test_empty_input(self):
        assert snapshot_dates([]) == []


class TestVerdict:
    def test_fresh_archive_passes(self):
        ok, age, _ = verdict([D(2026, 9, 3)], D(2026, 9, 4))
        assert ok is True
        assert age == 1

    def test_friday_snapshot_seen_on_monday_is_fine(self):
        ok, age, _ = verdict([D(2026, 9, 4)], D(2026, 9, 7))
        assert ok is True
        assert age == 3

    def test_holiday_length_gap_stays_quiet(self):
        # Fri snapshot, checked Tue after a Monday market holiday: 4 days.
        ok, age, _ = verdict([D(2026, 9, 4)], D(2026, 9, 8))
        assert ok is True
        assert age == 4

    def test_one_day_past_the_holiday_allowance_alerts(self):
        ok, age, reason = verdict([D(2026, 9, 3)], D(2026, 9, 8))
        assert ok is False
        assert age == 5
        assert 'has stopped landing snapshots' in reason

    def test_the_real_2026_09_05_state(self):
        # Newest was 2026-09-03 with Friday the 4th missing. On the Saturday
        # that is only 2 days old, so the alarm correctly had not fired yet.
        ok, age, _ = verdict([D(2026, 9, 3)], D(2026, 9, 5))
        assert ok is True
        assert age == 2

    def test_empty_archive_is_a_failure_not_a_pass(self):
        ok, age, reason = verdict([], D(2026, 9, 5))
        assert ok is False
        assert age is None
        assert 'no snapshots found' in reason

    def test_threshold_is_configurable(self):
        # 2026-08-31 -> 2026-09-05 is 5 days: past a limit of 4, inside 10.
        assert verdict([D(2026, 8, 31)], D(2026, 9, 5), max_age_days=4)[0] is False
        assert verdict([D(2026, 8, 31)], D(2026, 9, 5), max_age_days=10)[0] is True

    def test_boundary_is_inclusive(self):
        # Exactly at the limit passes; one day past it fails.
        assert verdict([D(2026, 9, 1)], D(2026, 9, 5), max_age_days=4)[0] is True
        assert verdict([D(2026, 8, 31)], D(2026, 9, 5), max_age_days=4)[0] is False


class TestMissingWeekdays:
    def test_finds_the_real_gaps(self):
        # The archive as it actually stood on 2026-09-05.
        present = [D(2026, 8, 17), D(2026, 8, 18), D(2026, 8, 19), D(2026, 8, 21),
                   D(2026, 8, 25), D(2026, 8, 26), D(2026, 8, 31),
                   D(2026, 9, 1), D(2026, 9, 2), D(2026, 9, 3)]
        gaps = missing_weekdays(present, D(2026, 9, 5), 21)
        assert gaps == [D(2026, 8, 20), D(2026, 8, 24), D(2026, 8, 27),
                        D(2026, 8, 28), D(2026, 9, 4)]

    def test_weekends_are_never_reported(self):
        gaps = missing_weekdays([], D(2026, 9, 7), 7)
        assert all(g.weekday() < 5 for g in gaps)
        assert D(2026, 9, 5) not in gaps  # Saturday
        assert D(2026, 9, 6) not in gaps  # Sunday

    def test_today_is_excluded(self):
        # The EOD run for today has not happened at 16:00, so today is never a gap.
        gaps = missing_weekdays([], D(2026, 9, 4), 3)
        assert D(2026, 9, 4) not in gaps

    def test_complete_archive_reports_nothing(self):
        present = [D(2026, 9, 1), D(2026, 9, 2), D(2026, 9, 3), D(2026, 9, 4)]
        assert missing_weekdays(present, D(2026, 9, 5), 4) == []


class TestFetchFailureIsNotSuccess:
    def test_unreadable_branch_exits_two(self, monkeypatch, capsys):
        import scripts.check_eod_delivery as mod

        def boom(*a, **k):
            raise OSError('connection reset')

        monkeypatch.setattr(mod, 'fetch_branch_filenames', boom)
        monkeypatch.setattr('sys.argv', ['x', '--repo', 'owner/name'])
        # Exit 2, distinct from the 1 that means "stale". A network blip must
        # not be reportable as either a healthy or a stopped pipeline.
        assert mod.main() == 2
        err = capsys.readouterr().err
        assert 'could not read' in err
        assert 'connection reset' in err

    def test_stale_archive_exits_one(self, monkeypatch, capsys):
        import scripts.check_eod_delivery as mod

        monkeypatch.setattr(mod, 'fetch_branch_filenames',
                            lambda *a, **k: ['results_2020-01-01.json'])
        monkeypatch.setattr('sys.argv', ['x', '--repo', 'owner/name'])
        assert mod.main() == 1
        assert 'STALE' in capsys.readouterr().out

    def test_fresh_archive_exits_zero(self, monkeypatch, capsys):
        import datetime as _dt
        import scripts.check_eod_delivery as mod

        today = _dt.date.today()
        monkeypatch.setattr(mod, 'fetch_branch_filenames',
                            lambda *a, **k: [f'results_{today:%Y-%m-%d}.json.gz'])
        monkeypatch.setattr('sys.argv', ['x', '--repo', 'owner/name'])
        assert mod.main() == 0
        assert 'OK' in capsys.readouterr().out
