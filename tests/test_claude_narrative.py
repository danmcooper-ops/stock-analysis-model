"""Tests for data/claude_narrative.py — facts builder purity and the
client's degrade-to-None paths, with the anthropic SDK stubbed. No network."""

import json
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.claude_narrative import (
    GICS_SECTORS, NARRATIVE_SCHEMA, ClaudeNarrativeClient, build_macro_facts,
)


def _sidecar():
    return {
        'as_of': '2026-08-22',
        'regime': {'regime': 'neutral', 'composite_score': 0.1,
                   'indicator_scores': {'vix': 0.2},
                   'raw_indicators': {'vix': 17.0}},
        'series': {
            'UNRATE': {'l': 'Unemployment Rate', 'sec': 'growth',
                       'fmt': 'pct1', 'suffix': '', 'good': 'down',
                       'freq': 'm',
                       'latest': {'d': '2026-08-01', 'v': 4.2},
                       'prior': {'d': '2026-07-01', 'v': 4.1},
                       'chg_1m': 0.1, 'chg_1y': 0.3, 'pctile': 0.44,
                       'pct_win': '10y', 'z': -0.2,
                       'hist': {'d': ['2026-07-01', '2026-08-01'],
                                'v': [4.1, 4.2]}},
        },
        'curve': {'tenors': ['3M', '10Y'], 'yrs': [0.25, 10],
                  'now': {'d': '2026-08-22', 'v': [4.5, 4.2]},
                  'm1': {'d': '2026-07-22', 'v': [4.6, 4.1]}},
        'oas_buckets': {'now': {'BBB': 1.2}, 'm1': {'BBB': 1.1}},
        'sector_data': {'Technology': {'etf': 'XLK', 'return_3m': 0.08,
                                       'rs_3m': 0.02, 'trend': 'improving'}},
    }


class TestBuildMacroFacts:
    def test_compact_series_drops_history(self):
        facts = build_macro_facts(_sidecar())
        s = facts['series']['UNRATE']
        assert 'hist' not in s
        assert s['latest'] == {'d': '2026-08-01', 'v': 4.2}
        assert s['chg_1y'] == 0.3
        assert 'yrs' not in (facts['yield_curve'] or {})
        assert facts['credit_oas_by_rating']['now']['BBB'] == 1.2

    def test_all_11_sectors_present_even_without_metrics(self):
        facts = build_macro_facts(_sidecar())
        assert set(facts['sectors']) == set(GICS_SECTORS)
        assert len(facts['sectors']) == 11
        # metrics merged where available, sensitivities everywhere
        assert facts['sectors']['Technology']['return_3m'] == 0.08
        assert 'macro_sensitivities' in facts['sectors']['Technology']
        # 'Financial Services' resolves the drivers table's legacy key
        assert facts['sectors']['Financial Services'].get(
            'macro_sensitivities'), 'Financials drivers must map to XLF sector'

    def test_empty_sidecar_is_harmless(self):
        facts = build_macro_facts(None)
        assert facts['as_of'] is None
        assert facts['series'] == {}
        assert set(facts['sectors']) == set(GICS_SECTORS)

    def test_schema_pins_three_named_paragraphs(self):
        # "exactly 3" must live in the schema as required object keys — the
        # grammar rejects minItems > 1 and ignored the prompt's own cap
        # (a live run returned 5 array paragraphs).
        p = NARRATIVE_SCHEMA['properties']['paragraphs']
        assert p['type'] == 'object'
        assert p['required'] == ['growth_labor', 'inflation_rates',
                                 'credit_conditions']

    def test_schema_pins_sector_shape(self):
        sec = NARRATIVE_SCHEMA['properties']['sectors']
        # The structured-outputs grammar rejects minItems other than 0/1
        # (live 400 on 2026-08-31), so lengths must NOT be pinned here —
        # the prompt + generate()'s post-parse check own the all-11 rule.
        assert 'minItems' not in sec and 'maxItems' not in sec
        assert set(sec['items']['properties']['sector']['enum']) == \
            set(GICS_SECTORS)
        # the Economist-style kicker: required on new generations, capped so
        # it stays a kicker and not a sentence
        assert sec['items']['properties']['headline']['maxLength'] == 60
        assert set(sec['items']['required']) == \
            {'sector', 'stance', 'headline', 'outlook'}


def _narrative():
    """API-shaped response: paragraphs arrive as the schema's named object
    and generate() flattens them to the list the page renders."""
    return {'paragraphs': {'growth_labor': 'Growth is slowing.',
                           'inflation_rates': 'Inflation is sticky.',
                           'credit_conditions': 'Credit is calm.'},
            'headwinds': ['Curve inverted'], 'tailwinds': ['Credit calm'],
            'sectors': [{'sector': s, 'stance': 'neutral',
                         'headline': 'Flat is fine', 'outlook': 'Flat.'}
                        for s in GICS_SECTORS]}


class _FakeBlock:
    type = 'text'

    def __init__(self, text):
        self.text = text


class _FakeResponse:
    def __init__(self, text, stop_reason='end_turn'):
        self.content = [_FakeBlock(text)]
        self.stop_reason = stop_reason


def _install_fake_anthropic(monkeypatch, response=None, raise_name=None,
                            calls=None):
    """Install a stub `anthropic` module whose messages.create returns
    `response`, or raises the module's own `raise_name` exception class —
    the instance must come from the same module object the client imports,
    or its except clauses would not match. The real SDK need not be
    importable."""
    mod = types.ModuleType('anthropic')

    class RateLimitError(Exception):
        pass

    class APIStatusError(Exception):
        status_code = 500

    class APIConnectionError(Exception):
        pass

    class _Messages:
        def create(self, **kwargs):
            if calls is not None:
                calls.append(kwargs)
            if raise_name is not None:
                raise getattr(mod, raise_name)('boom')
            return response

    class Anthropic:
        def __init__(self, api_key=None):
            self.messages = _Messages()

    mod.RateLimitError = RateLimitError
    mod.APIStatusError = APIStatusError
    mod.APIConnectionError = APIConnectionError
    mod.Anthropic = Anthropic
    monkeypatch.setitem(sys.modules, 'anthropic', mod)
    return mod


class TestClaudeNarrativeClient:
    def _client(self, tmp_path, **kw):
        kw.setdefault('api_key', 'sk-test')
        kw.setdefault('cache_dir', str(tmp_path / 'nar'))
        return ClaudeNarrativeClient(**kw)

    def test_no_key_returns_none_without_import(self, tmp_path, monkeypatch):
        monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
        c = ClaudeNarrativeClient(cache_dir=str(tmp_path / 'nar'))
        assert not c.available
        assert c.generate(_sidecar()) is None

    def test_happy_path_attaches_provenance_and_caches(self, tmp_path,
                                                       monkeypatch):
        calls = []
        _install_fake_anthropic(
            monkeypatch, response=_FakeResponse(json.dumps(_narrative())),
            calls=calls)
        c = self._client(tmp_path, model='claude-opus-5', max_tokens=6000)
        out = c.generate(_sidecar())
        assert out['paragraphs'][0] == 'Growth is slowing.'
        assert len(out['sectors']) == 11
        assert out['model'] == 'claude-opus-5'
        assert out['generated_at']
        # request carried the structured-output schema and the facts
        assert calls[0]['output_config']['format']['schema'] is \
            NARRATIVE_SCHEMA
        assert 'UNRATE' in calls[0]['messages'][0]['content']
        # cached on disk under the as_of date
        cached = json.loads((tmp_path / 'nar' / '2026-08-22.json')
                            .read_text(encoding='utf-8'))
        assert cached['paragraphs'] == out['paragraphs']

    def test_cache_hit_skips_the_api(self, tmp_path, monkeypatch):
        calls = []
        _install_fake_anthropic(
            monkeypatch, response=_FakeResponse(json.dumps(_narrative())),
            calls=calls)
        c = self._client(tmp_path)
        first = c.generate(_sidecar())
        second = c.generate(_sidecar())
        assert len(calls) == 1
        assert second == first

    @pytest.mark.parametrize('stop_reason', ['refusal', 'max_tokens'])
    def test_bad_stop_reasons_return_none(self, tmp_path, monkeypatch,
                                          stop_reason):
        _install_fake_anthropic(
            monkeypatch,
            response=_FakeResponse(json.dumps(_narrative()), stop_reason))
        assert self._client(tmp_path).generate(_sidecar()) is None

    def test_unparseable_json_returns_none(self, tmp_path, monkeypatch):
        _install_fake_anthropic(monkeypatch,
                                response=_FakeResponse('not json'))
        assert self._client(tmp_path).generate(_sidecar()) is None

    @pytest.mark.parametrize('raise_name', ['RateLimitError',
                                            'APIStatusError',
                                            'APIConnectionError'])
    def test_api_errors_return_none(self, tmp_path, monkeypatch, raise_name):
        _install_fake_anthropic(monkeypatch, raise_name=raise_name)
        assert self._client(tmp_path).generate(_sidecar()) is None

    def test_failures_are_not_cached(self, tmp_path, monkeypatch):
        _install_fake_anthropic(monkeypatch,
                                response=_FakeResponse('not json'))
        c = self._client(tmp_path)
        assert c.generate(_sidecar()) is None
        assert not os.path.exists(c._cache_path('2026-08-22'))

    def test_no_as_of_returns_none(self, tmp_path):
        assert self._client(tmp_path).generate({}) is None
        assert self._client(tmp_path).generate(None) is None
