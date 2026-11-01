# tests/test_app_feed.py
"""The iPhone/iPad app's feed (report_html._write_app_feed, design/ios-app-plan.md).

The app must see exactly the rows the page inlines as DATA, the manifest must
vouch only for a complete build, and nothing non-finite may reach it (Swift's
JSONDecoder rejects NaN, the page's parser tolerated it). The positions-leak
check in tests/test_report_portfolio_ui.py walks every output file, app/
included."""
import hashlib
import json
import re

import pytest

from scripts import report_html
from scripts.report_html import (APP_FEED_VERSION, _write_app_feed, build_html,
                                 pack_rows, unpack_rows)

ROWS = [
    {'ticker': 'AAA', 'price': 10.0, 'rating': 'BUY', 'sector': 'Energy', 'pe': float('nan')},
    {'ticker': 'BBB', 'price': 2.5, 'rating': 'HOLD', 'sector': 'Utilities'},
]


def _inline_data(html):
    m = re.search(r'var DATA=_unpackRows\((\{.*?\})\);\n', html)
    assert m, 'packed DATA not found'
    return unpack_rows(json.loads(m.group(1)))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rendered_feed_matches_inline_data(tmp_path):
    out = tmp_path / 'report.html'
    build_html([dict(r) for r in ROWS], str(out), prices_dir=None)
    app = tmp_path / 'app'
    manifest = json.loads((app / 'manifest.json').read_text(encoding='utf-8'))
    rows = unpack_rows(json.loads((app / 'rows.json').read_text(encoding='utf-8')))
    assert rows == _inline_data(out.read_text(encoding='utf-8'))
    assert [r['ticker'] for r in rows] == ['AAA', 'BBB']

    assert manifest['version'] == APP_FEED_VERSION
    assert manifest['rows'] == 2
    assert set(manifest['files']) == {'rows.json', 'portfolios.json', 'gates.json'}
    for name, digest in manifest['files'].items():
        assert _sha(app / name) == digest, name
    assert manifest['build'] == manifest['files']['rows.json'][:12]

    gates = json.loads((app / 'gates.json').read_text(encoding='utf-8'))
    assert gates['gates'] and all('key' in g for g in gates['gates'])


def test_feed_is_strict_json(tmp_path):
    _write_app_feed(str(tmp_path), [{'ticker': 'AAA', 'pe': float('nan'),
                                     'x': [float('inf'), 1.0]}], {}, {})
    text = (tmp_path / 'app' / 'rows.json').read_text(encoding='utf-8')
    assert 'NaN' not in text and 'Infinity' not in text
    rows = unpack_rows(json.loads(text))
    assert rows == [{'ticker': 'AAA', 'pe': None, 'x': [None, 1.0]}]


def test_excluded_keys_never_ship(tmp_path):
    recs = [{'ticker': 'AAA', 'finnhub_peers': ['BBB'], 'price': 1.0}]
    _write_app_feed(str(tmp_path), recs, {}, {}, exclude_keys={'finnhub_peers'})
    text = (tmp_path / 'app' / 'rows.json').read_text(encoding='utf-8')
    assert 'finnhub_peers' not in text
    assert unpack_rows(json.loads(text)) == [{'ticker': 'AAA', 'price': 1.0}]
    assert 'finnhub_peers' in recs[0]   # the page's records are untouched


def test_failed_write_leaves_no_manifest(tmp_path, monkeypatch):
    _write_app_feed(str(tmp_path), [{'ticker': 'AAA'}], {}, {}, run_date='2026-10-30')
    assert (tmp_path / 'app' / 'manifest.json').exists()

    def boom(_obj):
        raise RuntimeError('disk full')
    monkeypatch.setattr(report_html, 'pack_rows', boom)
    with pytest.raises(RuntimeError):
        _write_app_feed(str(tmp_path), [{'ticker': 'BBB'}], {}, {}, run_date='2026-10-31')
    # Yesterday's manifest must not vouch for a half-written build.
    assert not (tmp_path / 'app' / 'manifest.json').exists()


def test_render_survives_feed_failure(tmp_path, monkeypatch):
    def boom(*_a, **_k):
        raise OSError('read-only')
    monkeypatch.setattr(report_html, '_write_app_feed', boom)
    out = tmp_path / 'report.html'
    build_html([dict(r) for r in ROWS], str(out), prices_dir=None)
    assert out.exists()


def test_packed_shape_is_what_the_app_decodes():
    # The Swift RowTable ports exactly this shape; pin its keys.
    assert set(pack_rows([{'a': 1}, {'b': 2}])) == {'n', 'k', 'c', 'p', 'a'}
