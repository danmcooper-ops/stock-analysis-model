"""data/screen_skip_cache.py: TTL expiry, jitter, skip reasons, persistence."""

import json
from datetime import date, timedelta

import pytest

from data import screen_skip_cache as ssc
from data.screen_skip_cache import (DEAD_TTL_DAYS, MCAP_SKIP_FRACTION, MCAP_TTL_DAYS,
                                    ScreenSkipCache, _jitter)

TODAY = date(2026, 9, 16)
FLOOR = 300e6


@pytest.fixture
def path(tmp_path):
    return str(tmp_path / 'cache' / 'screen_skip.json')


def _cache(path, entries=None, today=TODAY):
    if entries is not None:
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(entries, f)
    return ScreenSkipCache(path, today=today)


# --- jitter ------------------------------------------------------------------

def test_jitter_is_deterministic_and_bounded():
    for tk in ('AAPL', 'MNSKY', 'X', 'BRK-B'):
        j = _jitter(tk, MCAP_TTL_DAYS)
        assert j == _jitter(tk, MCAP_TTL_DAYS)
        assert 0 <= j < MCAP_TTL_DAYS
    assert _jitter('AAPL', 0) == 0
    # crc32-based, so it does not depend on PYTHONHASHSEED
    import zlib
    assert _jitter('AAPL', 7) == zlib.crc32(b'AAPL') % 7


def test_jitter_staggers_tickers():
    assert len({_jitter(f'T{i}', MCAP_TTL_DAYS) for i in range(200)}) == MCAP_TTL_DAYS


# --- TTL expiry --------------------------------------------------------------

def _last_valid_day(ticker, base):
    return base + _jitter(ticker, base) - 1


@pytest.mark.parametrize('ticker', ['TINY', 'MICRO', 'ZZZZ'])
def test_mcap_entry_expires_after_jittered_ttl(path, ticker):
    last = _last_valid_day(ticker, MCAP_TTL_DAYS)
    seen = TODAY - timedelta(days=last)
    c = _cache(path, {ticker: {'kind': 'mcap', 'mcap': 10e6, 'date': seen.isoformat()}})
    assert c.skip_reason(ticker, FLOOR) == f'mcap $10M on {seen.isoformat()} (cached)'
    c.today = TODAY + timedelta(days=1)
    assert c.skip_reason(ticker, FLOOR) is None


@pytest.mark.parametrize('ticker', ['DEAD', 'GONE'])
def test_dead_entry_expires_after_jittered_ttl(path, ticker):
    last = _last_valid_day(ticker, DEAD_TTL_DAYS)
    seen = TODAY - timedelta(days=last)
    c = _cache(path, {ticker: {'kind': 'dead', 'date': seen.isoformat()}})
    assert c.skip_reason(ticker, FLOOR) == f'no data from any source since {seen.isoformat()}'
    c.today = TODAY + timedelta(days=1)
    assert c.skip_reason(ticker, FLOOR) is None


def test_future_dated_and_malformed_entries_do_not_skip(path):
    c = _cache(path, {
        'FUT': {'kind': 'dead', 'date': (TODAY + timedelta(days=1)).isoformat()},
        'BAD': {'kind': 'dead', 'date': 'yesterday'},
        'NODATE': {'kind': 'mcap', 'mcap': 1.0},
        'ODD': {'kind': 'other', 'date': TODAY.isoformat()},
    })
    for tk in ('FUT', 'BAD', 'NODATE', 'ODD', 'UNKNOWN'):
        assert c.skip_reason(tk, FLOOR) is None


# --- skip_reason: mcap vs dead ----------------------------------------------

def test_mcap_skip_needs_to_be_far_below_floor(path):
    c = _cache(path)
    c.record_mcap('FAR', FLOOR * MCAP_SKIP_FRACTION * 0.99)
    c.record_mcap('NEAR', FLOOR * MCAP_SKIP_FRACTION)
    c.record_mcap('NONE', None)
    assert c.skip_reason('FAR', FLOOR).startswith('mcap $')
    assert c.skip_reason('NEAR', FLOOR) is None
    assert c.skip_reason('NONE', FLOOR) == f'mcap $0M on {TODAY.isoformat()} (cached)'


def test_mcap_skip_disabled_without_floor_but_dead_still_skips(path):
    c = _cache(path)
    c.record_mcap('SMALL', 1e6)
    c.record_dead('DEAD')
    assert c.skip_reason('SMALL', 0) is None and c.skip_reason('SMALL', None) is None
    assert c.skip_reason('DEAD', None).startswith('no data from any source')


def test_record_overwrites_kind_and_forget_drops(path):
    c = _cache(path)
    c.record_dead('X')
    c.record_mcap('X', 1e6)
    assert c.skip_reason('X', FLOOR).startswith('mcap $1M')
    c.forget('X')
    c.forget('never-there')
    assert c.skip_reason('X', FLOOR) is None and len(c) == 0


# --- persistence -------------------------------------------------------------

def test_save_round_trips_and_is_noop_when_clean(path):
    c = _cache(path)
    c.save()                                   # nothing recorded: no file
    import os
    assert not os.path.exists(path)
    c.record_mcap('A', 5e6)
    c.record_dead('B')
    c.save()
    with open(path, encoding='utf-8') as f:
        on_disk = json.load(f)
    assert on_disk == {'A': {'kind': 'mcap', 'mcap': 5e6, 'date': TODAY.isoformat()},
                       'B': {'kind': 'dead', 'date': TODAY.isoformat()}}
    assert not os.path.exists(path + '.tmp')
    again = ScreenSkipCache(path, today=TODAY)
    assert len(again) == 2 and again.skip_reason('B', FLOOR)
    # forget on a reloaded cache marks it dirty and persists the drop
    again.forget('A')
    again.save()
    assert len(ScreenSkipCache(path, today=TODAY)) == 1


def test_unreadable_file_starts_empty(path, caplog):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('{not json')
    with caplog.at_level('WARNING', logger=ssc.__name__):
        c = ScreenSkipCache(path, today=TODAY)
    assert len(c) == 0
    assert 'unreadable' in caplog.text


def test_missing_file_starts_empty_silently(path, caplog):
    with caplog.at_level('WARNING', logger=ssc.__name__):
        c = ScreenSkipCache(path, today=TODAY)
    assert len(c) == 0 and caplog.text == ''


def test_save_failure_is_logged_not_raised(tmp_path, caplog):
    blocker = tmp_path / 'file'
    blocker.write_text('x', encoding='utf-8')
    c = ScreenSkipCache(str(blocker / 'sub' / 'screen_skip.json'), today=TODAY)
    c.record_dead('A')
    with caplog.at_level('WARNING', logger=ssc.__name__):
        c.save()
    assert 'save failed' in caplog.text


def test_defaults_to_today(path):
    assert ScreenSkipCache(path).today == date.today()
