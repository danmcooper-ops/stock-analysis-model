"""Tests for data/claude_narrative.py — facts builder purity and the
client's degrade-to-None paths, with the anthropic SDK stubbed. No network."""

import json
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.claude_narrative import (
    GICS_SECTORS, NARRATIVE_SCHEMA, SCHEMA_VERSION,
    SYSTEM_PROMPT, ClaudeNarrativeClient, build_macro_facts,
    overview_word_count,
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

    def test_series_carry_polarity_and_family(self):
        """'good' and 'sec' are what make a per-sector headwind/tailwind
        split writable: 'good' states which direction the report treats as
        favourable instead of leaving the model to infer it from the label,
        and 'sec' is the indicator family, which is how a sector's forces
        get drawn from different families rather than restating one rate."""
        s = build_macro_facts(_sidecar())['series']['UNRATE']
        assert s['good'] == 'down'
        assert s['sec'] == 'growth'

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

    def test_every_sector_carries_its_structural_forces(self):
        """Each influence weighs today's macro against the sector's standing
        forces, so the facts carry the same lists the Sector Analysis tab
        shows under Sector Headwinds & Tailwinds — for all 11 sectors, and
        whether or not the sidecar has ETF metrics for them."""
        from models.narrative import (
            _SECTOR_THESIS_RISKS, _SECTOR_THESIS_TAILWINDS,
        )
        for sidecar in (_sidecar(), None):
            sectors = build_macro_facts(sidecar)['sectors']
            for name in GICS_SECTORS:
                st = sectors[name].get('structural')
                assert st, '%s is missing its structural forces' % name
                assert st['headwinds'] == _SECTOR_THESIS_RISKS[name]
                assert st['tailwinds'] == _SECTOR_THESIS_TAILWINDS[name]
                # a copy: the facts are serialised into the prompt, and must
                # never alias the module-level tables
                assert st['headwinds'] is not _SECTOR_THESIS_RISKS[name]

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
        # each is a lead plus points, so the supporting sentences are a
        # field the model must fill, not a length it can ignore
        for k in p['required']:
            assert p['properties'][k]['required'] == ['lead', 'points']
            assert 'minItems' not in p['properties'][k]['properties']['points']

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
        # only what the Overview renders; the outlook and per-sector bullets
        # went with the sector tabs' Macro Outlook section (schema v5)
        assert sec['items']['required'] == \
            ['sector', 'influence', 'headline', 'stance']
        # the argument first, then the kicker and stance that summarise it
        assert list(sec['items']['properties']) == \
            ['sector', 'influence', 'headline', 'stance']
        assert sec['items']['additionalProperties'] is False

    def test_prompt_states_the_per_sector_rules(self):
        """How a sector's influence picks and judges its forces lives only
        in the prompt, so pin it — without this the rules can be deleted and
        nothing fails."""
        p = SYSTEM_PROMPT
        # the two fields build_macro_facts passes for this are explained
        assert "'sec' key" in p and "carries 'good'" in p
        assert 'DIFFERENT indicator' in p
        assert "judge each force by THIS sector's exposure" in p
        # the ban once scoped to the outlook now binds all prose
        assert '"may", "could", "likely"' in p and '"bears watching"' in p
        # the influence blends in the sector's standing forces and says
        # which dominates, without quoting the Sector Analysis tab's list
        assert "its 'structural' headwinds and tailwinds" in p
        assert 'ONE structural force' in p and 'which dominates now' in p
        assert 'never quote it' in p
        for gone in ('outlook:', 'tailwinds / headwinds: the macro forces',
                     '3 to 5 bullets in TOTAL'):
            assert gone not in p, '%r is a dropped field' % gone

    def test_overview_word_count_counts_what_the_page_prints(self):
        n = {'paragraphs': ['One two three.', 'Four.'],
             'tailwinds': ['five six'], 'headwinds': [],
             'sectors': [{'headline': 'Seven eight', 'influence': 'Nine.',
                          'stance': 'neutral'}]}
        assert overview_word_count(n) == 9
        assert overview_word_count(None) == 0

    def test_prompt_states_the_overview_prose_rules(self):
        """The Overview's length and readability live only in the prompt:
        the influence band that makes the page a several-minute read, the
        lead-plus-bullets layout the paragraphs are split into, and the
        rounding rule that keeps '1.5th percentile' off the page."""
        p = SYSTEM_PROMPT
        assert 'influence:' in p and '30 to 40 words in all' in p
        assert 'total about 750 words' in p
        assert 'MECHANISM' in p, 'influence explains channels, not bullets'
        assert 'explain it rather than list it' in p
        assert 'lead: ONE sentence' in p and 'points:' in p
        assert 'exactly 3 supporting sentences' in p
        assert 'whole-number ordinal percentiles' in p


def _narrative():
    """API-shaped response: paragraphs arrive as the schema's named object
    and generate() flattens them to the list the page renders."""
    return {'paragraphs': {'growth_labor': {'lead': 'Growth is slowing',
                                            'points': ['Claims are low.']},
                           'inflation_rates': 'Inflation is sticky.',
                           'credit_conditions': 'Credit is calm.'},
            'headwinds': ['Curve inverted'], 'tailwinds': ['Credit calm'],
            'sectors': [{'sector': s, 'influence': 'Rates and credit offset.',
                         'headline': 'Flat is fine', 'stance': 'neutral'}
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
        # BOTH names, or the test passes for the wrong reason on a machine
        # where only one of them happens to be exported.
        monkeypatch.delenv('MACRO_ANTHROPIC_API_KEY', raising=False)
        monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
        c = ClaudeNarrativeClient(cache_dir=str(tmp_path / 'nar'))
        assert not c.available
        assert c.generate(_sidecar()) is None

    def test_macro_key_is_read_and_wins_over_the_anthropic_one(
            self, tmp_path, monkeypatch):
        """The cloud routine's container never receives ANTHROPIC_API_KEY —
        that name belongs to the Claude Code session running the routine — so
        the narrative reads MACRO_ANTHROPIC_API_KEY first. An explicit
        api_key argument still outranks both."""
        cache = str(tmp_path / 'nar')
        monkeypatch.setenv('MACRO_ANTHROPIC_API_KEY', 'sk-macro')
        monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-session')
        assert ClaudeNarrativeClient(cache_dir=cache).api_key == 'sk-macro'
        assert ClaudeNarrativeClient(cache_dir=cache,
                                     api_key='sk-arg').api_key == 'sk-arg'

    def test_anthropic_key_still_works_as_a_fallback(self, tmp_path,
                                                     monkeypatch):
        """The Mac runbook's .env and any existing shell export keep
        working: the old name is still read when the new one is absent."""
        monkeypatch.delenv('MACRO_ANTHROPIC_API_KEY', raising=False)
        monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-legacy')
        c = ClaudeNarrativeClient(cache_dir=str(tmp_path / 'nar'))
        assert c.api_key == 'sk-legacy'
        assert c.available

    def test_happy_path_attaches_provenance_and_caches(self, tmp_path,
                                                       monkeypatch):
        calls = []
        _install_fake_anthropic(
            monkeypatch, response=_FakeResponse(json.dumps(_narrative())),
            calls=calls)
        c = self._client(tmp_path, model='claude-opus-5', max_tokens=6000)
        out = c.generate(_sidecar())
        assert out['paragraphs'][0] == 'Growth is slowing. Claims are low.'  # lead + points, stop added
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

    def test_duplicate_sectors_collapse_to_one_per_sector(self, tmp_path,
                                                          monkeypatch):
        """A live 2026-08-31 run returned 26 entries across the 11 names:
        the grammar cannot pin array length, so dedupe post-parse."""
        payload = _narrative()
        payload['sectors'] = payload['sectors'] + [
            {'sector': s, 'influence': 'Dupe.', 'headline': 'Second take',
             'stance': 'headwind'}
            for s in GICS_SECTORS[:4]]
        _install_fake_anthropic(
            monkeypatch, response=_FakeResponse(json.dumps(payload)))
        out = self._client(tmp_path).generate(_sidecar())
        names = [x['sector'] for x in out['sectors']]
        assert names == GICS_SECTORS            # canonical order, no dupes
        # the FIRST entry per sector wins whole, not a merge of two draws
        assert all(x['influence'] == 'Rates and credit offset.' and
                   x['stance'] == 'neutral' for x in out['sectors'])

    def test_partial_sector_list_survives_dedupe(self, tmp_path, monkeypatch):
        """A short list still renders — dedupe must not invent entries."""
        payload = _narrative()
        payload['sectors'] = payload['sectors'][:5]
        _install_fake_anthropic(
            monkeypatch, response=_FakeResponse(json.dumps(payload)))
        out = self._client(tmp_path).generate(_sidecar())
        assert [x['sector'] for x in out['sectors']] == GICS_SECTORS[:5]

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

    def test_stale_schema_cache_is_a_miss_not_a_hit(self, tmp_path,
                                                    monkeypatch):
        """A cache hit returns before every post-parse check, so a file
        written under an older shape would be served to the page as-is.
        The day cache is keyed by date alone, so without a version gate the
        run on the day of a shape change replays yesterday's shape."""
        calls = []
        _install_fake_anthropic(
            monkeypatch, response=_FakeResponse(json.dumps(_narrative())),
            calls=calls)
        c = self._client(tmp_path)
        c.generate(_sidecar())
        assert len(calls) == 1
        path = c._cache_path('2026-08-22')
        cached = json.loads(open(path, encoding='utf-8').read())
        assert cached['schema_version'] == SCHEMA_VERSION
        # rewrite it as the previous shape, as a pre-upgrade run would have
        cached.pop('schema_version')
        for entry in cached['sectors']:
            entry['outlook'] = 'Flat.'
            entry.pop('influence', None)
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(cached, fh)
        out = c.generate(_sidecar())
        assert len(calls) == 2, 'stale-shape cache must regenerate'
        assert out['sectors'][0]['influence'] == 'Rates and credit offset.'
        assert 'outlook' not in out['sectors'][0]

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


def test_financial_services_drivers_are_keyed_by_the_row_sector():
    """_SECTOR_MACRO_DRIVERS used 'Financials', which no row carries, so
    Financial Services stocks never got the yield-curve signals."""
    from models.narrative import _SECTOR_MACRO_DRIVERS, _sector_signals
    assert 'Financials' not in _SECTOR_MACRO_DRIVERS
    assert _SECTOR_MACRO_DRIVERS['Financial Services']['benefits_from_higher_rates']
    regime = {'regime': 'neutral', 'raw_indicators': {'yield_curve_slope': 0.02}}
    _, tw = _sector_signals({'sector': 'Financial Services'}, {}, regime, {})
    assert any('net interest margins' in t for t in tw)
