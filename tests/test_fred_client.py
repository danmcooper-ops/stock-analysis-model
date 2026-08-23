"""Tests for data/fred_client.py — offline, no network."""

import json
import os
import sys
import time
from datetime import date

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.fred_client import FREDClient


@pytest.fixture(autouse=True)
def _no_env_key(monkeypatch):
    """A developer's real FRED_API_KEY must not leak into these tests."""
    monkeypatch.delenv('FRED_API_KEY', raising=False)


def make_client(tmp_path, api_key=None):
    return FREDClient(api_key=api_key, cache_dir=str(tmp_path),
                      request_delay=0)


KEYED_PAYLOAD = json.dumps({
    'observations': [
        {'date': '2026-08-19', 'value': '4.28'},
        {'date': '2026-08-20', 'value': '.'},          # holiday marker
        {'date': '2026-08-21', 'value': '4.31'},
        {'date': '2026-08-22', 'value': ''},           # blank
    ]
})

KEYLESS_CSV = (
    'observation_date,CPIAUCSL\n'
    '2026-06-01,313.5\n'
    '2026-07-01,314.2\n'
    '2026-08-01,.\n'
)


class TestKeyedParse:
    def test_parses_raw_values_and_skips_holidays(self, tmp_path, monkeypatch):
        c = make_client(tmp_path, api_key='k' * 32)
        monkeypatch.setattr(c, '_get', lambda url, params: KEYED_PAYLOAD)
        obs = c._fetch_keyed('DGS10')
        assert obs == {date(2026, 8, 19): 4.28, date(2026, 8, 21): 4.31}

    def test_malformed_json_returns_none(self, tmp_path, monkeypatch):
        c = make_client(tmp_path, api_key='k' * 32)
        monkeypatch.setattr(c, '_get', lambda url, params: '<html>oops')
        assert c._fetch_keyed('DGS10') is None

    def test_network_failure_returns_none(self, tmp_path, monkeypatch):
        c = make_client(tmp_path, api_key='k' * 32)
        monkeypatch.setattr(c, '_get', lambda url, params: None)
        assert c._fetch_keyed('DGS10') is None


class TestKeylessParse:
    def test_parses_csv(self, tmp_path, monkeypatch):
        c = make_client(tmp_path)
        monkeypatch.setattr(c, '_get', lambda url, params: KEYLESS_CSV)
        obs = c._fetch_keyless('CPIAUCSL')
        assert obs == {date(2026, 6, 1): 313.5, date(2026, 7, 1): 314.2}

    def test_non_csv_body_returns_none(self, tmp_path, monkeypatch):
        c = make_client(tmp_path)
        monkeypatch.setattr(c, '_get', lambda url, params: '<html>error')
        assert c._fetch_keyless('CPIAUCSL') is None


class TestUnitFidelity:
    """The bond repo's client divides every value by 100 (its universe is
    all-percent). This client must store index levels and counts raw —
    a CPI of 314.2 coming back as 3.142 is the regression to guard."""

    def test_index_level_survives_untouched(self, tmp_path, monkeypatch):
        c = make_client(tmp_path)
        monkeypatch.setattr(c, '_get', lambda url, params: KEYLESS_CSV)
        obs = c.fetch_series('CPIAUCSL')
        assert obs[date(2026, 7, 1)] == 314.2


class TestFetchSeries:
    def test_keyed_falls_back_to_keyless(self, tmp_path, monkeypatch):
        c = make_client(tmp_path, api_key='k' * 32)
        monkeypatch.setattr(c, '_fetch_keyed', lambda *a, **k: None)
        monkeypatch.setattr(c, '_fetch_keyless',
                            lambda *a, **k: {date(2026, 8, 1): 1.0})
        assert c.fetch_series('X') == {date(2026, 8, 1): 1.0}

    def test_total_failure_returns_empty_dict(self, tmp_path, monkeypatch):
        c = make_client(tmp_path)
        monkeypatch.setattr(c, '_get', lambda url, params: None)
        assert c.fetch_series('X') == {}
        # ... and memoizes the failure instead of refetching
        monkeypatch.setattr(c, '_get', lambda url, params: KEYLESS_CSV)
        assert c.fetch_series('X') == {}

    def test_memoizes_within_session(self, tmp_path, monkeypatch):
        c = make_client(tmp_path)
        calls = []
        def _get(url, params):
            calls.append(url)
            return KEYLESS_CSV.replace('CPIAUCSL', 'X')
        monkeypatch.setattr(c, '_get', _get)
        c.fetch_series('X')
        c.fetch_series('X')
        assert len(calls) == 1


class TestDiskCache:
    def test_round_trip(self, tmp_path, monkeypatch):
        c1 = make_client(tmp_path)
        monkeypatch.setattr(c1, '_get', lambda url, params: KEYLESS_CSV)
        first = c1.fetch_series('CPIAUCSL')

        c2 = make_client(tmp_path)
        monkeypatch.setattr(c2, '_get',
                            lambda url, params: pytest.fail('hit network'))
        assert c2.fetch_series('CPIAUCSL') == first

    def test_expired_cache_refetches(self, tmp_path, monkeypatch):
        c1 = make_client(tmp_path)
        monkeypatch.setattr(c1, '_get', lambda url, params: KEYLESS_CSV)
        c1.fetch_series('CPIAUCSL')
        # Backdate the cache file two days
        path = c1._cache_path('CPIAUCSL')
        old = time.time() - 2 * 86400
        os.utime(path, (old, old))

        c2 = make_client(tmp_path)
        assert c2._load_cache('CPIAUCSL') is None

    def test_corrupt_cache_refetches(self, tmp_path, monkeypatch):
        c = make_client(tmp_path)
        with open(c._cache_path('X'), 'w') as fh:
            fh.write('{not json')
        assert c._load_cache('X') is None


class TestAsOfValue:
    OBS = {date(2026, 8, 15): 4.1, date(2026, 8, 18): 4.2,
           date(2026, 8, 21): 4.3}

    def test_exact_date(self):
        d, v = FREDClient._as_of_value(self.OBS, date(2026, 8, 18))
        assert (d, v) == (date(2026, 8, 18), 4.2)

    def test_nearest_before_never_forward(self):
        d, v = FREDClient._as_of_value(self.OBS, date(2026, 8, 20))
        assert (d, v) == (date(2026, 8, 18), 4.2)

    def test_lookback_limit(self):
        d, v = FREDClient._as_of_value(self.OBS, date(2026, 9, 15))
        assert (d, v) == (None, None)

    def test_empty_obs(self):
        assert FREDClient._as_of_value({}, date(2026, 8, 18)) == (None, None)


class TestAvailability:
    def test_keyless_client_reports_unavailable_key(self, tmp_path):
        c = make_client(tmp_path)
        assert c.available is False
        assert c.history_source == 'keyless'

    def test_env_key_picked_up(self, tmp_path, monkeypatch):
        monkeypatch.setenv('FRED_API_KEY', 'e' * 32)
        c = make_client(tmp_path)
        assert c.available is True
        assert c.history_source == 'keyed'
