"""The iOS app's feed: app/latest.json, app/detail/<TICKER>.json and the
manifest that tells the app a night is complete (StockModel-iOS-Handoff
README section 4), plus its publish path."""
import hashlib
import json
import os

import scripts.publish_vol_shards as pvs
from scripts.report_html import (_APP_LIST_FIELDS, _APP_SCHEMA_VERSION, _build_app_list,
                                 _write_app_feed)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_SH = os.path.join(REPO, 'scripts', 'run_daily.sh')


def _records():
    return [
        {'ticker': 'BBB', 'company_name': 'Bee', 'rating': 'HOLD', 'price': 10.0,
         'mos': None, '_composite_score': 40.0, 'dcf_fv': 11.0, 'pf': []},
        {'ticker': 'AAA', 'company_name': 'Ay', 'sector': 'Tech', 'rating': 'BUY',
         'price': 5.0, '_fv_effective': 9.0, '_fv_source': 'dcf', 'mos': 0.44,
         '_composite_score': 88.0, 'trap_score': 12.0, 'price_data_stale': False,
         'ddm_fv': None, 'nav_fv': 7.5},
    ]


def _rows():
    return [{'ticker': 'AAA', '_risk_free_rate': 0.0484},
            {'ticker': 'BBB', '_risk_free_rate': 0.0484}]


def _details():
    return {'AAA': {'description_full': 'A long description.', 'ceo_bio': 'A bio.'}}


# --- the list feed ----------------------------------------------------------

def test_list_feed_carries_only_its_fields_sorted_by_ticker():
    feed = _build_app_list(_records(), '2026-11-05', 0.0484)
    assert feed['schema_version'] == _APP_SCHEMA_VERSION
    assert (feed['date'], feed['risk_free_rate'], feed['count']) == ('2026-11-05', 0.0484, 2)
    assert [r['ticker'] for r in feed['results']] == ['AAA', 'BBB']
    for row in feed['results']:
        assert set(row) <= set(_APP_LIST_FIELDS)
    # Absent and None-valued fields are omitted, not nulled: the app's types
    # are optional and this is what keeps the feed under its size target.
    assert 'mos' not in feed['results'][1] and 'dcf_fv' not in feed['results'][1]
    assert feed['results'][0]['mos'] == 0.44


def test_list_feed_skips_rows_with_no_ticker():
    assert _build_app_list([{'price': 1.0}, {'ticker': ''}], '2026-11-05', None)['count'] == 0


# --- the written feed -------------------------------------------------------

def _sha256(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def test_feed_writes_latest_detail_and_a_matching_manifest(tmp_path):
    app = tmp_path / 'app'
    assert _write_app_feed(str(app), _records(), _details(), _rows(), '2026-11-05') == ['AAA', 'BBB']

    latest = json.loads((app / 'latest.json').read_text(encoding='utf-8'))
    assert latest['count'] == 2 and latest['risk_free_rate'] == 0.0484

    manifest = json.loads((app / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['schema_version'] == _APP_SCHEMA_VERSION
    assert manifest['date'] == '2026-11-05' and manifest['count'] == 2
    assert manifest['tickers'] == ['AAA', 'BBB']
    assert manifest['generated_at'].endswith('+00:00')
    # Only latest.json is hashed: hashing every shard made the manifest
    # 363 KiB for a file the app fetches on each launch just to ask whether
    # there is a new night.
    assert set(manifest['files']) == {'app/latest.json'}
    for rel, digest in manifest['files'].items():
        assert _sha256(tmp_path / rel) == digest
    # The shards are accounted for by name, and each carries its own night,
    # which is what replaces a per-shard hash.
    assert manifest['tickers'] == ['AAA', 'BBB']
    for tk in manifest['tickers']:
        shard = json.loads((app / 'detail' / f'{tk}.json').read_text(encoding='utf-8'))
        assert shard['date'] == '2026-11-05'

    detail = json.loads((app / 'detail' / 'AAA.json').read_text(encoding='utf-8'))
    # The row's remaining fields, plus the heavy text, plus its own night.
    assert detail['nav_fv'] == 7.5
    assert detail['description_full'] == 'A long description.' and detail['ceo_bio'] == 'A bio.'
    assert detail['ticker'] == 'AAA' and detail['date'] == '2026-11-05'
    # The list feed already carries these; repeating them would double the
    # bytes the app downloads for a company it has already listed.
    assert not (set(detail) & set(_APP_LIST_FIELDS)) - {'ticker'}


def test_feed_nulls_non_finite_numbers_and_stays_valid_json(tmp_path):
    app = tmp_path / 'app'
    recs = [{'ticker': 'AAA', 'price': float('nan'), 'mos': float('inf'),
             'epv_fv': '-Infinity', 'rim_fv': float('-inf')}]
    assert _write_app_feed(str(app), recs, {}, _rows(), '2026-11-05') == ['AAA']
    for rel in ('latest.json', 'detail/AAA.json'):
        raw = (app / rel).read_text(encoding='utf-8')
        # Swift's JSONDecoder rejects these outright, so they must never ship.
        assert 'NaN' not in raw and 'Infinity' not in raw
        json.loads(raw)       # and the file must parse at all
    detail = json.loads((app / 'detail' / 'AAA.json').read_text(encoding='utf-8'))
    assert detail['epv_fv'] is None and detail['rim_fv'] is None
    # A NaN price is dropped from the list feed rather than shipped as null.
    assert 'price' not in json.loads((app / 'latest.json').read_text(encoding='utf-8'))['results'][0]


def test_feed_is_rebuilt_wholesale_and_skips_unsafe_tickers(tmp_path):
    app = tmp_path / 'app'
    _write_app_feed(str(app), _records(), _details(), _rows(), '2026-11-04')
    stale = app / 'detail' / 'GONE.json'
    stale.write_text('{}', encoding='utf-8')
    recs = [{'ticker': 'AAA'}, {'ticker': '../escape'}, {'ticker': 'A/B'}]
    assert _write_app_feed(str(app), recs, {}, _rows(), '2026-11-05') == ['AAA']
    assert os.listdir(app / 'detail') == ['AAA.json']
    assert not stale.exists()
    manifest = json.loads((app / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['date'] == '2026-11-05' and manifest['tickers'] == ['AAA']


def test_feed_is_skipped_without_a_run_date(tmp_path):
    app = tmp_path / 'app'
    assert _write_app_feed(str(app), _records(), _details(), _rows(), None) == []
    assert not app.exists()


# --- publishing -------------------------------------------------------------

def _repo(tmp_path, app=True):
    out = tmp_path / 'repo' / 'output'
    for fam, names in (('vol', ['AAA']), ('px', ['AAA']), ('hist', ['AAA']),
                       ('details', ['0'])):
        (out / fam).mkdir(parents=True)
        for n in names:
            (out / fam / f'{n}.json').write_text(json.dumps({fam: n}), encoding='utf-8')
    (out / 'prices_meta.json').write_text(json.dumps({'vol': ['AAA'], 'manifest': ['AAA']}),
                                          encoding='utf-8')
    (out / 'hist_index.json').write_text(json.dumps({'tickers': ['AAA']}), encoding='utf-8')
    (out / 'details_index.json').write_text(json.dumps({'parts': ['0']}), encoding='utf-8')
    if app:
        _write_app_feed(str(out / 'app'), _records(), _details(), _rows(), '2026-11-05')
    return tmp_path / 'repo', tmp_path / 'docs'


def test_publish_syncs_app_detail_by_manifest(tmp_path, monkeypatch):
    repo, docs = _repo(tmp_path)
    (docs / 'app' / 'detail').mkdir(parents=True)
    (docs / 'app' / 'detail' / 'STALE.json').write_text('{}', encoding='utf-8')
    (docs / 'app' / 'detail' / 'AAA 2.json').write_text('{}', encoding='utf-8')   # conflict copy
    monkeypatch.setenv('STOCK_MODEL_REPO', str(repo))
    monkeypatch.setenv('PAGES_DOCS', str(docs))
    assert pvs.main() == 0
    assert sorted(os.listdir(docs / 'app' / 'detail')) == ['AAA.json', 'BBB.json']


def test_publish_prunes_app_detail_but_does_not_fail_without_the_feed(tmp_path, monkeypatch):
    repo, docs = _repo(tmp_path, app=False)
    (docs / 'app' / 'detail').mkdir(parents=True)
    (docs / 'app' / 'detail' / 'LAST_NIGHT.json').write_text('{}', encoding='utf-8')
    monkeypatch.setenv('STOCK_MODEL_REPO', str(repo))
    monkeypatch.setenv('PAGES_DOCS', str(docs))
    # The app feed must never fail the web publish.
    assert pvs.main() == 0
    assert not (docs / 'app' / 'detail').exists()
    assert sorted(os.listdir(docs / 'px')) == ['AAA.json']


def test_run_daily_publishes_the_app_feed_and_the_change_files():
    sh = open(RUN_SH, encoding='utf-8').read()
    assert 'cp output/app/latest.json "$docs/app/latest.json"' in sh
    assert 'cp output/app/manifest.json "$docs/app/manifest.json"' in sh
    # No manifest means no app/ at all, rather than last night's under
    # today's report.
    assert 'rm -rf "$docs/app"' in sh
    assert 'for f in rating_history.json portfolio_alerts.json; do' in sh
    assert r'docs/app/detail/* *.json\n' in sh
