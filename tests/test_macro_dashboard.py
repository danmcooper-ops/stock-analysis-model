"""Tests for scripts/macro_dashboard.py — pure transforms and payload
assembly with a dict-backed stub client. No network."""

import os
import sys
from datetime import date, timedelta

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.macro_dashboard import (
    yoy, mom_diff, percentile_rank, zscore, downsample, recession_bands,
    build_macro_payload, MACRO_SERIES, OVERVIEW_IDS,
)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

class TestYoY:
    def test_exact_year_apart(self):
        obs = {date(2025, 7, 1): 100.0, date(2026, 7, 1): 103.0}
        out = yoy(obs)
        assert out[date(2026, 7, 1)] == pytest.approx(3.0)

    def test_day_of_month_drift_within_tolerance(self):
        # Jan-15 matches against the nearest obs ~a year back (Jan-1)
        obs = {date(2025, 1, 1): 200.0, date(2026, 1, 15): 210.0}
        out = yoy(obs)
        assert out[date(2026, 1, 15)] == pytest.approx(5.0)

    def test_gap_beyond_tolerance_is_dropped(self):
        obs = {date(2024, 1, 1): 100.0, date(2026, 1, 1): 120.0}
        out = yoy(obs)
        assert date(2026, 1, 1) not in out

    def test_empty(self):
        assert yoy({}) == {}


class TestMoMDiff:
    def test_consecutive_diffs(self):
        obs = {date(2026, 5, 1): 100.0, date(2026, 6, 1): 110.0,
               date(2026, 7, 1): 95.0}
        out = mom_diff(obs)
        assert out[date(2026, 6, 1)] == pytest.approx(10.0)
        assert out[date(2026, 7, 1)] == pytest.approx(-15.0)
        assert date(2026, 5, 1) not in out

    def test_single_point(self):
        assert mom_diff({date(2026, 5, 1): 1.0}) == {}


class TestPercentileRank:
    def test_latest_is_max(self):
        assert percentile_rank([1, 2, 3], 3) == 1.0

    def test_latest_is_min(self):
        assert percentile_rank([1, 2, 3], 1) == pytest.approx(1 / 3)

    def test_empty_or_none(self):
        assert percentile_rank([], 1) is None
        assert percentile_rank([1, 2], None) is None


class TestZscore:
    def test_centered(self):
        assert zscore([1.0, 2.0, 3.0], 2.0) == pytest.approx(0.0)

    def test_needs_min_history(self):
        assert zscore([1.0, 2.0], 2.0) is None

    def test_zero_variance(self):
        assert zscore([2.0, 2.0, 2.0], 2.0) == 0.0


class TestDownsample:
    def test_trailing_year_kept_dense_older_thinned(self):
        last = date(2026, 8, 21)
        obs = {last - timedelta(days=i): float(i) for i in range(0, 900)}
        out = downsample(obs)
        recent = [d for d in out if (last - d).days <= 365]
        older = [d for d in out if (last - d).days > 365]
        assert len(recent) == 366           # full daily density
        assert len(older) < 900 - 366       # thinned
        assert min(obs) in out              # first point kept
        assert last in out                  # last point kept

    def test_empty(self):
        assert downsample({}) == {}


class TestRecessionBands:
    def test_two_episodes(self):
        obs = {date(2020, 1, 1): 0, date(2020, 2, 1): 1, date(2020, 3, 1): 1,
               date(2020, 4, 1): 0, date(2021, 1, 1): 1, date(2021, 2, 1): 0}
        bands = recession_bands(obs)
        assert bands == [['2020-02-01', '2020-04-01'],
                         ['2021-01-01', '2021-02-01']]

    def test_open_episode_ends_at_last_obs(self):
        obs = {date(2026, 6, 1): 0, date(2026, 7, 1): 1, date(2026, 8, 1): 1}
        assert recession_bands(obs) == [['2026-07-01', '2026-08-01']]

    def test_empty(self):
        assert recession_bands({}) == []


# ---------------------------------------------------------------------------
# Payload assembly
# ---------------------------------------------------------------------------

class StubFRED:
    """Dict-backed stand-in for FREDClient."""

    available = True

    def __init__(self, table=None):
        self.table = table or {}

    def fetch_series(self, series_id, start=None, end=None, force=False):
        return dict(self.table.get(series_id, {}))

    def fetch_cmt_curve(self, as_of=None, with_dates=False):
        return {}

    def fetch_bucket_oas(self, as_of=None):
        return {}


def _monthly(start_year, n, base=100.0, step=1.0):
    out = {}
    y, m = start_year, 1
    for i in range(n):
        out[date(y, m, 1)] = base + i * step
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def full_stub():
    as_of = date(2026, 8, 22)
    table = {}
    for meta in MACRO_SERIES:
        if meta['freq'] in ('d', 'w'):
            table[meta['id']] = {as_of - timedelta(days=i * 7): 4.0 + i * 0.01
                                 for i in range(120)}
        else:
            table[meta['id']] = _monthly(2016, 128)
    table['USREC'] = {date(2020, 1, 1): 0, date(2020, 2, 1): 1,
                      date(2020, 4, 1): 0}
    return StubFRED(table), as_of


class TestBuildMacroPayload:
    def test_schema_and_summary(self):
        fred, as_of = full_stub()
        p = build_macro_payload(fred, as_of=as_of)
        assert p is not None
        sc = p['sidecar']
        assert sc['as_of'] == '2026-08-22'
        assert sc['keyed'] is True
        assert sc['recessions'] == [['2020-02-01', '2020-04-01']]
        assert {s['k'] for s in sc['sections']} == \
            {'rates', 'inflation', 'growth', 'credit', 'housing'}
        # every declared series made it in with the stub's full data
        assert set(sc['series']) == {m['id'] for m in MACRO_SERIES}
        entry = sc['series']['UNRATE']
        for key in ('l', 'latest', 'prior', 'chg_1m', 'chg_1y', 'pctile',
                    'pct_win', 'hist'):
            assert key in entry
        assert entry['latest']['v'] is not None
        # summary tiles cover the overview ids, with small sparklines
        tiles = p['summary']['tiles']
        assert [t['id'] for t in tiles] == OVERVIEW_IDS
        assert all(len(t['spark']) <= 60 for t in tiles)

    def test_offline_returns_none(self):
        assert build_macro_payload(StubFRED()) is None

    def test_partial_failures_are_skipped(self):
        fred, as_of = full_stub()
        del fred.table['UNRATE']
        p = build_macro_payload(fred, as_of=as_of)
        assert 'UNRATE' not in p['sidecar']['series']
        growth = [s for s in p['sidecar']['sections'] if s['k'] == 'growth'][0]
        assert 'UNRATE' not in growth['ids']

    def test_regime_passthrough(self):
        fred, as_of = full_stub()
        regime = {'regime': 'neutral', 'composite_score': 0.0,
                  'indicator_scores': {}, 'raw_indicators': {}}
        adj = {'erp_adjustment': 0.001}
        p = build_macro_payload(fred, regime_result=regime, macro_adj=adj,
                                as_of=as_of)
        assert p['sidecar']['regime']['regime'] == 'neutral'
        assert p['sidecar']['regime']['adjustments'] == adj
        assert p['summary']['regime']['regime'] == 'neutral'

    def test_no_regime_is_null(self):
        fred, as_of = full_stub()
        p = build_macro_payload(fred, as_of=as_of)
        assert p['sidecar']['regime'] is None


# ---------------------------------------------------------------------------
# Template smoke — the tab's hooks exist and degrade without a payload
# ---------------------------------------------------------------------------

class TestTemplateSmoke:
    def test_template_carries_macro_hooks(self):
        path = os.path.join(os.path.dirname(__file__), '..', 'templates',
                            'report.html')
        with open(path, encoding='utf-8') as fh:
            html = fh.read()
        for needle in ('view-macro', 'renderMacro', 'p.mo', '_MACRO_AVAILABLE',
                       "_loadSidecar('macro.json'"):
            assert needle in html, needle
