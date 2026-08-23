# tests/test_provenance.py
"""Tests for record-level provenance tracking and the run event log."""

import json
import sys
import os
from datetime import date

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.provenance import (
    SCHEMA_VERSION, STALE_CACHE_DAYS, ProvenanceRecorder,
    enrichment_block, attach_enrichment, strip_enrichment,
    append_events, library_versions, make_event,
)


# ======================================================================
# ProvenanceRecorder tests
# ======================================================================

class TestProvenanceRecorder:

    def test_ticker_block_contains_recorded_sources(self):
        rec = ProvenanceRecorder(date(2026, 6, 10))
        rec.record_source('AAPL', 'statements', 'sec_xbrl+yfinance',
                          sec={'cik': '0000320193', 'accn': 'x', 'form': '10-K'})
        rec.record_source('AAPL', 'beta', 'capm (reliable)')
        block = rec.ticker_block('AAPL')
        assert block['schema_version'] == SCHEMA_VERSION
        assert block['run_date'] == '2026-06-10'
        assert block['sources']['statements']['source'] == 'sec_xbrl+yfinance'
        assert block['sources']['statements']['sec']['form'] == '10-K'
        assert block['sources']['beta']['source'] == 'capm (reliable)'
        assert 'fetched_at' in block['sources']['statements']

    def test_none_extras_are_dropped(self):
        rec = ProvenanceRecorder('2026-06-10')
        rec.record_source('MSFT', 'statements', 'yfinance', sec=None)
        assert 'sec' not in rec.ticker_block('MSFT')['sources']['statements']

    def test_unknown_ticker_block_is_empty_not_error(self):
        rec = ProvenanceRecorder('2026-06-10')
        block = rec.ticker_block('ZZZZ')
        assert block['sources'] == {}

    def test_event_counts(self):
        rec = ProvenanceRecorder('2026-06-10')
        rec.record_event('source_fallback', 'NKE', 'sec_xbrl', {'to': 'yfinance'})
        rec.record_event('source_fallback', 'KO', 'sec_xbrl', None)
        rec.record_event('fx_fetch_failed', 'SAP', 'fx_client', None)
        assert rec.event_counts() == {'source_fallback': 2, 'fx_fetch_failed': 1}

    def test_run_block_tallies_from_results_rows(self):
        rec = ProvenanceRecorder('2026-06-10')
        # Phase-1 tally includes a screened-out ticker...
        rec.record_source('TINY', 'statements', 'yfinance')
        rec.record_source('AAPL', 'statements', 'sec_xbrl+yfinance')
        # ...but run_block counts from the final rows when given.
        rows = [{'data_source': 'sec_xbrl+yfinance'},
                {'data_source': 'sec_xbrl+yfinance'},
                {'data_source': 'yfinance'},
                {'ticker': 'NOSRC'}]
        block = rec.run_block(rows)
        assert block['statement_sources'] == {'sec_xbrl+yfinance': 2, 'yfinance': 1}
        assert block['schema_version'] == SCHEMA_VERSION
        assert block['run_started_at'] and block['run_finished_at']
        assert block['events_file'] == 'events_2026-06-10.json'
        assert 'python' in block['versions']

    def test_run_block_falls_back_to_internal_tally(self):
        rec = ProvenanceRecorder('2026-06-10')
        rec.record_source('AAPL', 'statements', 'sec_xbrl')
        assert rec.run_block()['statement_sources'] == {'sec_xbrl': 1}

    def test_recorder_methods_never_raise(self):
        rec = ProvenanceRecorder('2026-06-10')
        rec.record_source(None, 'statements', object())   # weird but tolerated
        rec.record_event('source_fallback')
        assert isinstance(rec.ticker_block(None), dict)
        assert isinstance(rec.run_block(), dict)

    def test_write_events_uses_sidecar(self, tmp_path):
        rec = ProvenanceRecorder('2026-06-10')
        rec.record_event('stale_cache', 'JPM', 'fdic', {'cache_age_days': 12.4})
        rec.write_events(str(tmp_path))
        with open(tmp_path / 'events_2026-06-10.json', encoding='utf-8') as f:
            data = json.load(f)
        events = data['sections']['analyze_stock']['events']
        assert len(events) == 1
        assert events[0]['type'] == 'stale_cache'
        assert events[0]['ticker'] == 'JPM'


# ======================================================================
# Enrichment helper tests
# ======================================================================

class TestEnrichmentHelpers:

    def test_enrichment_block_fields(self):
        block = enrichment_block(applied=True, cache_hit=True,
                                 cache_age_days=3.14159, repdte='20260331',
                                 reason=None)
        assert block['applied'] is True
        assert block['cache_hit'] is True
        assert block['cache_age_days'] == 3.14
        assert block['repdte'] == '20260331'
        assert 'reason' not in block
        assert 'fetched_at' in block

    def test_attach_creates_scaffold_on_legacy_record(self):
        record = {'ticker': 'JPM', 'sector': 'Financial Services'}
        attach_enrichment(record, 'fdic', enrichment_block())
        assert record['_provenance']['schema_version'] == SCHEMA_VERSION
        assert record['_provenance']['enrichments']['fdic']['applied'] is True

    def test_attach_is_idempotent_overwrite(self):
        record = {}
        attach_enrichment(record, 'fdic', enrichment_block(cache_age_days=1.0))
        attach_enrichment(record, 'fdic', enrichment_block(cache_age_days=2.0))
        assert len(record['_provenance']['enrichments']) == 1
        assert record['_provenance']['enrichments']['fdic']['cache_age_days'] == 2.0

    def test_attach_preserves_existing_provenance(self):
        record = {'_provenance': {'schema_version': SCHEMA_VERSION,
                                  'run_date': '2026-06-10',
                                  'sources': {'statements': {'source': 'yfinance'}}}}
        attach_enrichment(record, 'xbrl', enrichment_block(applied=False, reason='no_cik'))
        assert record['_provenance']['sources']['statements']['source'] == 'yfinance'
        assert record['_provenance']['enrichments']['xbrl']['reason'] == 'no_cik'

    def test_strip_enrichment(self):
        record = {}
        attach_enrichment(record, 'fdic', enrichment_block())
        strip_enrichment(record, 'fdic')
        assert record['_provenance']['enrichments'] == {}
        # Stripping again, or on a legacy record, must not raise.
        strip_enrichment(record, 'fdic')
        strip_enrichment({'ticker': 'X'}, 'fdic')


# ======================================================================
# Events sidecar tests
# ======================================================================

class TestAppendEvents:

    def test_sections_coexist(self, tmp_path):
        append_events(str(tmp_path), '2026-06-10', 'enrich_fdic',
                      [make_event('stale_cache', 'JPM', 'fdic')])
        append_events(str(tmp_path), '2026-06-10', 'enrich_pipeline',
                      [make_event('enrichment_skipped', 'PFE', 'clinicaltrials')])
        with open(tmp_path / 'events_2026-06-10.json', encoding='utf-8') as f:
            data = json.load(f)
        assert set(data['sections']) == {'enrich_fdic', 'enrich_pipeline'}
        assert data['date'] == '2026-06-10'

    def test_rerun_replaces_section_not_appends(self, tmp_path):
        append_events(str(tmp_path), '2026-06-10', 'enrich_fdic',
                      [make_event('stale_cache', 'JPM', 'fdic'),
                       make_event('stale_cache', 'BAC', 'fdic')])
        append_events(str(tmp_path), '2026-06-10', 'enrich_fdic',
                      [make_event('stale_cache', 'JPM', 'fdic')])
        with open(tmp_path / 'events_2026-06-10.json', encoding='utf-8') as f:
            data = json.load(f)
        assert len(data['sections']['enrich_fdic']['events']) == 1

    def test_corrupt_sidecar_is_recovered(self, tmp_path):
        path = tmp_path / 'events_2026-06-10.json'
        path.write_text('{not json')
        append_events(str(tmp_path), '2026-06-10', 'enrich_fdic', [])
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        assert data['sections']['enrich_fdic']['events'] == []


def test_library_versions_never_raises():
    versions = library_versions()
    assert versions['python'].count('.') == 2
    assert 'pandas' in versions and 'numpy' in versions


def test_stale_threshold_is_sane_for_daily_pipeline():
    assert 0 < STALE_CACHE_DAYS < 30
