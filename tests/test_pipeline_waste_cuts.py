"""Guards for the nightly-run waste cuts: no retrying dead symbols, and a
Glassdoor circuit breaker so one dead endpoint cannot cost every ticker."""

from unittest import mock

import pytest

from data import culture_client as cc
from data.yfinance_client import YFinanceClient, _is_not_found


def _client():
    return YFinanceClient(request_delay=0, fetch_timeout=None)


def test_retry_gives_up_immediately_on_404(monkeypatch):
    monkeypatch.setattr('time.sleep', lambda s: None)
    calls = []

    def dead():
        calls.append(1)
        raise RuntimeError('HTTP Error 404: Quote not found for symbol: ZZZZ')

    with pytest.raises(RuntimeError):
        _client()._retry(dead)
    assert len(calls) == 1


def test_retry_still_retries_transient_errors(monkeypatch):
    monkeypatch.setattr('time.sleep', lambda s: None)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError('reset by peer')
        return 'ok'

    assert _client()._retry(flaky) == 'ok'
    assert len(calls) == 3


def test_is_not_found_does_not_match_throttle():
    assert _is_not_found(RuntimeError('No fundamentals data found for symbol: X'))
    assert not _is_not_found(RuntimeError('Too Many Requests'))


def test_glassdoor_breaker_disables_after_consecutive_failures(monkeypatch):
    monkeypatch.setattr(cc, '_glassdoor_cache', {})
    client = cc.CultureClient(request_delay=0)
    with mock.patch('urllib.request.urlopen', side_effect=OSError('404')) as m:
        for i in range(cc.GLASSDOOR_MAX_CONSECUTIVE_FAILURES + 5):
            client.fetch_glassdoor('Acme Corp', f'T{i}')
    assert m.call_count == cc.GLASSDOOR_MAX_CONSECUTIVE_FAILURES


# ---------------------------------------------------------------------------
# Screen skip cache
# ---------------------------------------------------------------------------

from datetime import date, timedelta  # noqa: E402

from data.screen_skip_cache import (DEAD_TTL_DAYS, MCAP_TTL_DAYS,  # noqa: E402
                                    ScreenSkipCache)

D0 = date(2026, 9, 9)


def _cache(tmp_path, today=D0):
    return ScreenSkipCache(path=str(tmp_path / 'skip.json'), today=today)


def test_mcap_far_below_floor_is_skipped_next_run(tmp_path):
    c = _cache(tmp_path)
    c.record_mcap('TINY', 20e6)
    c.save()
    nxt = _cache(tmp_path, D0 + timedelta(days=1))
    assert nxt.skip_reason('TINY', 300e6)
    # The floor was lowered: no longer far below it.
    assert nxt.skip_reason('TINY', 30e6) is None
    # No floor at all: never skip on mcap.
    assert nxt.skip_reason('TINY', 0) is None


def test_mcap_near_floor_is_not_skipped(tmp_path):
    c = _cache(tmp_path)
    c.record_mcap('NEAR', 250e6)
    assert c.skip_reason('NEAR', 300e6) is None


def test_entries_expire_within_twice_the_base_ttl(tmp_path):
    c = _cache(tmp_path)
    c.record_mcap('TINY', 1e6)
    c.record_dead('GONE')
    c.save()
    late = _cache(tmp_path, D0 + timedelta(days=2 * MCAP_TTL_DAYS))
    assert late.skip_reason('TINY', 300e6) is None
    late = _cache(tmp_path, D0 + timedelta(days=2 * DEAD_TTL_DAYS))
    assert late.skip_reason('GONE', 300e6) is None


def test_forget_and_corrupt_file(tmp_path):
    c = _cache(tmp_path)
    c.record_dead('GONE')
    c.forget('GONE')
    assert c.skip_reason('GONE', 0) is None
    (tmp_path / 'skip.json').write_text('{not json', encoding='utf-8')
    assert len(_cache(tmp_path)) == 0


# --- Staged across cloud runs (scheduled-tasks/cloud-daily-stock-analysis) ---
# The cache is committed to the data/snapshots branch AFTER the analysis, so a
# staged copy is always at least a day old by the time the next run reads it.

def test_staged_cache_still_skips_when_a_day_or_more_stale(tmp_path):
    c = _cache(tmp_path)
    c.record_mcap('TINY', 20e6)
    c.record_dead('GONE')
    c.save()
    # A normal night (staged from yesterday) and a few missed nights.
    for lag in (1, 2, 5):
        nxt = _cache(tmp_path, D0 + timedelta(days=lag))
        assert nxt.skip_reason('TINY', 300e6), f"mcap entry lost at lag {lag}"
        assert nxt.skip_reason('GONE', 300e6), f"dead entry lost at lag {lag}"


def test_future_dated_entries_are_refetched_not_skipped(tmp_path):
    """A --run-date re-run of a past session stages a NEWER cache.

    Those entries describe a day that hasn't happened yet for this run, so
    they must not screen anything out: _age_ok's `0 <= days` rejects them.
    """
    c = _cache(tmp_path)
    c.record_mcap('TINY', 1e6)
    c.record_dead('GONE')
    c.save()
    past = _cache(tmp_path, D0 - timedelta(days=3))
    assert past.skip_reason('TINY', 300e6) is None
    assert past.skip_reason('GONE', 300e6) is None


def test_save_is_idempotent_while_not_dirty(tmp_path):
    """The periodic in-loop flush must not rewrite the file every 500 tickers."""
    c = _cache(tmp_path)
    c.record_mcap('TINY', 1e6)
    c.save()
    path = tmp_path / 'skip.json'
    before = path.stat().st_mtime_ns
    c.save()   # nothing recorded since — must be a no-op
    assert path.stat().st_mtime_ns == before
    c.record_dead('GONE')
    c.save()
    assert 'GONE' in path.read_text(encoding='utf-8')
