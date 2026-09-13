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
