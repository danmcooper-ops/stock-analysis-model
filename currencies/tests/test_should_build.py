"""The scheduled-build gate.

Offline: the ECB fetch is stubbed. The cases that matter are the ones that
decide whether a weekday run happens at all, and the failure modes are
asymmetric — wrongly proceeding republishes yesterday's numbers under today's
date, wrongly skipping just loses a day that the next run reproduces exactly.
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from scripts import should_build

NY = ZoneInfo('America/New_York')


@pytest.fixture
def run(monkeypatch, capsys):
    """Invoke main() with a pinned clock and a stubbed ECB response."""
    def _run(local_hour=16, ecb_offset_days=0, ecb=True, argv=None):
        now = datetime.now(NY).replace(hour=local_hour, minute=15)

        class FrozenDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                return now

        monkeypatch.setattr(should_build, 'datetime', FrozenDateTime)

        def fake_fetch(*a, **kw):
            if not ecb:
                return None
            day = (now.date() + timedelta(days=ecb_offset_days)).isoformat()
            return {'USD': {day: 1.15}}

        monkeypatch.setattr(should_build.ecb_client,
                            'fetch_reference_rates', fake_fetch)
        should_build.main(argv if argv is not None
                          else ['--tz', 'America/New_York', '--hour', '16'])
        return capsys.readouterr().out
    return _run


def test_proceeds_at_the_target_hour_with_todays_rates(run):
    assert 'proceed=true' in run(local_hour=16, ecb_offset_days=0)


def test_skips_before_the_target_hour(run):
    """The EST-side cron lands at 15:15 under EDT."""
    out = run(local_hour=15)
    assert 'proceed=false' in out
    assert 'before 16:00' in out


@pytest.mark.parametrize('local_hour', [17, 18, 19, 23])
def test_proceeds_when_the_cron_starts_late(run, local_hour):
    """GitHub delays scheduled runs by hours; a late run must still build.

    Requiring the exact 16:00 hour dropped every run from 2026-08-26 on.
    """
    assert 'proceed=true' in run(local_hour=local_hour, ecb_offset_days=0)


def _hour_argv(last_built):
    return ['--tz', 'America/New_York', '--hour', '16',
            '--last-built', last_built]


def test_skips_when_already_built_today(run):
    """Under EDT both crons pass the hour floor; only the first may build."""
    today_1615 = datetime.now(NY).replace(hour=16, minute=15)
    out = run(local_hour=18, argv=_hour_argv(str(int(today_1615.timestamp()))))
    assert 'proceed=false' in out
    assert 'already built today' in out


def test_proceeds_when_last_build_was_yesterday(run):
    yesterday = datetime.now(NY).replace(hour=16, minute=15) - timedelta(days=1)
    out = run(local_hour=16, argv=_hour_argv(str(int(yesterday.timestamp()))))
    assert 'proceed=true' in out


@pytest.mark.parametrize('last_built', ['', 'not-a-number'])
def test_unreadable_last_built_does_not_gate(run, last_built):
    """No pages-live branch yet, or a failed fetch: build rather than skip."""
    assert 'proceed=true' in run(local_hour=16, argv=_hour_argv(last_built))


def test_skips_when_ecb_published_nothing_today(run):
    """A TARGET holiday: newest observation is an earlier date.

    Rebuilding would republish yesterday's rates stamped with today's date.
    """
    out = run(local_hour=16, ecb_offset_days=-1)
    assert 'proceed=false' in out
    assert 'newest ECB observation' in out


def test_skips_when_ecb_is_unreachable(run):
    out = run(local_hour=16, ecb=False)
    assert 'proceed=false' in out
    assert 'unavailable' in out


def test_skip_freshness_flag_bypasses_the_data_check(run):
    out = run(local_hour=16, ecb_offset_days=-30,
              argv=['--hour', '16', '--skip-freshness'])
    assert 'proceed=true' in out


def test_omitting_hour_disables_the_time_check(run):
    out = run(local_hour=3, ecb_offset_days=0, argv=[])
    assert 'proceed=true' in out


def test_unknown_timezone_fails_open(run):
    """A bad tz must not silently suppress every build."""
    out = run(argv=['--tz', 'Not/AZone', '--hour', '16'])
    assert 'proceed=true' in out


def test_writes_github_output_file(monkeypatch, tmp_path, run):
    out_file = tmp_path / 'gho.txt'
    monkeypatch.setenv('GITHUB_OUTPUT', str(out_file))
    run(local_hour=16, ecb_offset_days=0)
    written = out_file.read_text()
    assert 'proceed=true' in written
    assert written.count('\n') == 2      # exactly proceed= and reason=
    assert 'reason=' in written
