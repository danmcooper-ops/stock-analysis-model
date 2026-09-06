"""Tests for scripts/macro_dashboard.py — pure transforms and payload
assembly with a dict-backed stub client. No network."""

import os
import sys
from datetime import date, timedelta

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.macro_dashboard import (
    yoy, mom_diff, percentile_rank, zscore, downsample, recession_bands,
    build_macro_payload, make_narrative_client, _sector_facts,
    MACRO_SERIES, OVERVIEW_IDS,
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
        # spark_d names the observation under the pointer when the sparkline
        # is scrubbed, so it has to stay index-aligned with spark.
        for t in tiles:
            assert len(t['spark_d']) == len(t['spark'])
            hist = sc['series'][t['id']]['hist']
            assert t['spark_d'] == hist['d'][-len(t['spark']):]
            assert t['spark'] == hist['v'][-len(t['spark']):]

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
# Sector facts + Claude narrative attachment
# ---------------------------------------------------------------------------

class TestSectorFacts:
    def test_merges_etf_metrics_and_local_rs(self):
        sector_data = {'Technology': {'etf': 'XLK', 'return_3m': 0.08,
                                      'rel_strength_3m': 0.02,
                                      'return_6m': None}}
        local_rs = {'XLK': {'rs_1m': 0.01, 'rs_3m': 0.03, 'trend': 'improving'},
                    'XLF': {'rs_3m': -0.02, 'trend': 'weakening'}}
        out = _sector_facts(sector_data, local_rs)
        tech = out['Technology']
        assert tech == {'etf': 'XLK', 'return_3m': 0.08,
                        'rel_strength_3m': 0.02, 'rs_1m': 0.01,
                        'rs_3m': 0.03, 'trend': 'improving'}
        # XLF has no entry in MacroClient's 10-sector ETF map, so its RS is
        # the only route Financial Services facts can arrive by.
        assert out['Financial Services'] == {'etf': 'XLF', 'rs_3m': -0.02,
                                             'trend': 'weakening'}
        # sectors with nothing beyond the ETF name are dropped
        assert 'Utilities' not in out

    def test_nothing_usable_is_none(self):
        assert _sector_facts(None, None) is None
        assert _sector_facts({}, {}) is None


class StubNarrativeClient:
    def __init__(self, narrative=None, error=None):
        self.narrative = narrative
        self.error = error
        self.seen = []

    def generate(self, sidecar):
        self.seen.append(sidecar)
        if self.error:
            raise self.error
        return self.narrative


class TestNarrativeAttachment:
    def _narrative(self):
        return {'paragraphs': ['The economy is fine.'],
                'headwinds': [], 'tailwinds': [],
                'sectors': [{'sector': 'Technology', 'stance': 'neutral',
                             'headline': 'Flat is fine', 'outlook': 'Flat.',
                             'tailwinds': ['Core PCE at 2.8%'],
                             'headwinds': ['10Y at 4.3%']}]}

    def test_attached_to_sidecar_and_summary(self):
        fred, as_of = full_stub()
        client = StubNarrativeClient(self._narrative())
        local_rs = {'XLK': {'rs_3m': 0.03, 'trend': 'improving'}}
        p = build_macro_payload(fred, as_of=as_of, local_rs=local_rs,
                                narrative_client=client)
        assert p['sidecar']['narrative'] == self._narrative()
        assert p['summary']['narrative'] == self._narrative()
        # the client saw the sidecar with sector facts already attached
        assert client.seen[0]['sector_data']['Technology']['rs_3m'] == 0.03
        # sector_data rides the inline summary too — the report's metric
        # figs paint at first render from MACRO_SUM, before macro.json loads
        assert p['summary']['sector_data'] == p['sidecar']['sector_data']
        assert p['summary']['sector_data']['Technology']['rs_3m'] == 0.03

    def test_generator_failure_degrades_to_no_narrative(self):
        fred, as_of = full_stub()
        p = build_macro_payload(fred, as_of=as_of,
                                narrative_client=StubNarrativeClient(
                                    error=RuntimeError('api down')))
        assert p is not None
        assert 'narrative' not in p['sidecar']
        assert p['summary']['narrative'] is None

    def test_no_client_no_narrative(self):
        fred, as_of = full_stub()
        p = build_macro_payload(fred, as_of=as_of)
        assert 'narrative' not in p['sidecar']
        assert p['summary']['narrative'] is None


class TestMakeNarrativeClient:
    def test_disabled_by_config_flag(self, monkeypatch):
        import scripts.config as config
        monkeypatch.setattr(config, 'CLAUDE_NARRATIVE_ENABLED', False)
        assert make_narrative_client() is None

    def test_enabled_builds_configured_client(self, monkeypatch):
        import scripts.config as config
        monkeypatch.setattr(config, 'CLAUDE_NARRATIVE_ENABLED', True)
        client = make_narrative_client()
        assert client is not None
        assert client.model == config.CLAUDE_NARRATIVE_MODEL
        assert client.max_tokens == config.CLAUDE_NARRATIVE_MAX_TOKENS


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


# ---------------------------------------------------------------------------
# Re-render path — step 1f of the daily routine
# ---------------------------------------------------------------------------

class TestSummaryFromSidecar:
    def test_rebuilds_tiles_from_sidecar(self):
        fred, as_of = full_stub()
        p = build_macro_payload(fred, as_of=as_of)
        from scripts.macro_dashboard import summary_from_sidecar
        rebuilt = summary_from_sidecar(p['sidecar'])
        assert rebuilt == p['summary']

    def test_replay_carries_the_narrative(self):
        fred, as_of = full_stub()
        nar = {'paragraphs': ['Prose.'], 'headwinds': [], 'tailwinds': [],
               'sectors': []}
        p = build_macro_payload(fred, as_of=as_of,
                                narrative_client=StubNarrativeClient(nar))
        from scripts.macro_dashboard import summary_from_sidecar
        assert summary_from_sidecar(p['sidecar'])['narrative'] == nar

    def test_empty_sidecar_is_none(self):
        from scripts.macro_dashboard import summary_from_sidecar
        assert summary_from_sidecar(None) is None
        assert summary_from_sidecar({}) is None
        assert summary_from_sidecar({'series': {}}) is None


class TestRerenderKeepsMacroTab:
    """Regression: scripts/rescore_and_render.py re-renders the HTML after the
    enrichment passes. Rendering with no payload dropped the Macro Outlook tab
    and deleted the macro.json the main run had just written."""

    def _snap(self):
        return {'date': '2026-08-22',
                'macro_regime': {'regime': 'neutral', 'composite_score': 0.0,
                                 'indicator_scores': {}, 'raw_indicators': {}},
                'macro_adjustments': {'erp_adjustment': 0.001}}

    def test_fresh_build_used_when_fred_reachable(self, tmp_path, monkeypatch):
        import data.fred_client as fc
        import scripts.macro_dashboard as md
        import scripts.rescore_and_render as rr
        fred, as_of = full_stub()
        monkeypatch.setattr(fc, 'FREDClient', lambda *a, **k: fred)
        # keep the test hermetic: never construct a real Claude client here
        monkeypatch.setattr(md, 'make_narrative_client', lambda: None)
        payload = rr._macro_payload_for_render(
            str(tmp_path / 'r.html'), as_of, self._snap())
        assert payload and payload['sidecar']['series']
        # regime from the snapshot rides along, so the banner survives
        assert payload['sidecar']['regime']['regime'] == 'neutral'
        assert payload['sidecar']['regime']['adjustments'] == \
            self._snap()['macro_adjustments']

    def test_falls_back_to_existing_sidecar_when_fred_down(self, tmp_path,
                                                           monkeypatch):
        import json as _json

        import data.fred_client as fc
        import scripts.rescore_and_render as rr
        fred, as_of = full_stub()
        good = build_macro_payload(fred, as_of=as_of)
        (tmp_path / 'macro.json').write_text(_json.dumps(good['sidecar']),
                                             encoding='utf-8')

        class Dead:
            available = False
            def fetch_series(self, *a, **k): return {}
            def fetch_cmt_curve(self, *a, **k): return {}
            def fetch_bucket_oas(self, *a, **k): return {}

        monkeypatch.setattr(fc, 'FREDClient', lambda *a, **k: Dead())
        payload = rr._macro_payload_for_render(
            str(tmp_path / 'r.html'), as_of, self._snap())
        assert payload is not None, 'a FRED outage must not drop the tab'
        assert payload['sidecar'] == good['sidecar']
        assert payload['summary']['tiles']

    def test_none_when_no_fred_and_no_sidecar(self, tmp_path, monkeypatch):
        import data.fred_client as fc
        import scripts.rescore_and_render as rr

        class Dead:
            available = False
            def fetch_series(self, *a, **k): return {}
            def fetch_cmt_curve(self, *a, **k): return {}
            def fetch_bucket_oas(self, *a, **k): return {}

        monkeypatch.setattr(fc, 'FREDClient', lambda *a, **k: Dead())
        assert rr._macro_payload_for_render(
            str(tmp_path / 'r.html'), None, self._snap()) is None
