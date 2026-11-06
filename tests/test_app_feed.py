"""The iOS app's feed: app/latest.json, app/detail/<TICKER>.json and the
manifest that tells the app a night is complete (StockModel-iOS-Handoff
README section 4), plus its publish path."""
import datetime as dt
import hashlib
import json
import os

import scripts.publish_vol_shards as pvs
import scripts.report_html as rh
import re

from scripts.report_html import (_APP_LIST_FIELDS, _APP_SCHEMA_VERSION, _APP_SCREENER_FIELDS,
                                 _app_is_blue_chip, _app_profit_pool, _app_sector_heatmap,
                                 _build_app_list, _build_app_screener, _write_app_feed,
                                 unpack_rows)

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


def test_feed_takes_a_date_run_date_and_date_valued_fields(tmp_path):
    # The nightly render passes run_date as a datetime.date and a row can
    # carry one too; both used to raise inside the wrapped write, which cost
    # the whole feed and left only a warning line.
    app = tmp_path / 'app'
    recs = [{'ticker': 'AAA', 'last_filing': dt.date(2026, 10, 31),
             'fetched_at': dt.datetime(2026, 11, 5, 17, 30, 15)}]
    assert _write_app_feed(str(app), recs, {}, _rows(), dt.date(2026, 11, 5)) == ['AAA']
    latest = json.loads((app / 'latest.json').read_text(encoding='utf-8'))
    manifest = json.loads((app / 'manifest.json').read_text(encoding='utf-8'))
    assert latest['date'] == '2026-11-05' and manifest['date'] == '2026-11-05'
    detail = json.loads((app / 'detail' / 'AAA.json').read_text(encoding='utf-8'))
    assert detail['date'] == '2026-11-05'
    assert detail['last_filing'] == '2026-10-31'
    assert detail['fetched_at'] == '2026-11-05T17:30:15'


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
    # Sector Analysis and the screener ride with the list feed, and a night
    # without one drops last night's copy rather than serving it beside
    # today's list.
    assert 'for f in sectors.json screener.json; do' in sh
    assert 'cp "output/app/$f" "$docs/app/$f"' in sh
    assert 'rm -f "$docs/app/$f"' in sh
    assert r'docs/app/detail/* *.json\n' in sh


# --- Sector Analysis (app/sectors.json) -------------------------------------
# The report computes these in the browser from DATA; the app gets them
# precomputed. Each assertion is a rule of renderCrossSectorDashboard or
# renderCrossSectorProfitPool.

def _sector_rows():
    return [
        {'ticker': 'A1', 'sector': 'Tech', 'pp_revenue_share': 0.75, 'operating_margin': 0.20,
         'pp_sector_hhi': 0.3, 'pp_sector_cr4': 0.9, 'pp_multiple': 2.0, 'rating': 'BUY',
         'mos': 0.4, 'p_tbv': 3.0, 'ev_ebitda': 12.0, 'roic': 0.25, 'rev_cagr_5y': 0.10,
         'fcf_margin': 0.15, 'altman_z': 6.0, 'shareholder_yield': 0.02,
         'revenue': 300.0, 'operating_income': 60.0},
        {'ticker': 'A2', 'sector': 'Tech', 'pp_revenue_share': 0.25, 'operating_margin': 0.40,
         'pp_sector_hhi': 0.3, 'pp_sector_cr4': 0.9, 'pp_multiple': 2.0, 'rating': 'HOLD',
         'mos': 0.0, 'p_tbv': 80.0, 'ev_ebitda': -5.0, 'roic': 0.15, 'rev_cagr_5y': 0.30,
         'fcf_margin': 0.25, 'altman_z': 4.0, 'shareholder_yield': 0.0,
         'share_buyback_rate': 0.01, 'div_yield': 0.02,
         'revenue': 100.0, 'operating_income': 40.0},
        # A bank: in the heatmap, but not in FCF margin or Altman Z.
        {'ticker': 'B1', 'sector': 'Financial Services', 'industry': 'Banks - Regional',
         'pp_revenue_share': 1.0, 'operating_margin': 0.30, 'rating': 'LEAN BUY',
         'fcf_margin': 0.9, 'altman_z': 0.5, 'rev_cagr': 0.05,
         'revenue': 500.0, 'operating_income': 150.0},
        # Its preferred: the parent's statements again, counted once.
        {'ticker': 'B1-P', 'sector': 'Financial Services', 'industry': 'Banks - Regional',
         'revenue': 500.0, 'operating_income': 150.0},
        {'ticker': 'X', 'sector': 'Tech'},   # no pool share: not in the heatmap
    ]


def test_heatmap_matches_the_reports_rules():
    rows = {r['sector']: r for r in _app_sector_heatmap(_sector_rows())}
    tech = rows['Tech']
    assert tech['n'] == 2
    # Revenue-weighted: (0.20 * 0.75 + 0.40 * 0.25) / 1.0.
    assert abs(tech['op_margin'] - 0.25) < 1e-12
    assert (tech['hhi'], tech['cr4']) == (0.3, 0.9)
    assert tech['med_mos'] == 0.2
    # P/TBV keeps 0-50 and EV/EBITDA 0-100: A2's 80x and -5x drop out.
    assert tech['med_p_tbv'] == 3.0 and tech['med_ev_ebitda'] == 12.0
    # A zero shareholder yield falls back to buyback + dividend yield.
    assert abs(tech['med_shr_yld'] - 0.025) < 1e-12
    assert (tech['n_buy'], tech['pct_buy']) == (1, 0.5)
    # A tie on the pool multiple goes to the later row, as reduce() does.
    assert tech['best']['ticker'] == 'A1'

    fin = rows['Financial Services']
    # Lenders are left out of FCF margin and Altman Z.
    assert fin['med_fcf_margin'] is None and fin['med_altman_z'] is None
    # No five-year CAGR anywhere in the sector: the plain CAGR stands in.
    assert fin['med_growth'] == 0.05
    assert fin['pct_buy'] == 1.0


def test_profit_pool_counts_each_issuer_once():
    pool = _app_profit_pool(_sector_rows())
    by = {a['sector']: a for a in pool}
    assert by['Financial Services']['companies'] == 1
    assert by['Financial Services']['revenue'] == 500.0
    assert by['Tech']['revenue'] == 400.0 and by['Tech']['operating_income'] == 100.0
    # Ordered by margin: Financial Services 30%, then Tech 25%.
    assert [a['sector'] for a in pool] == ['Financial Services', 'Tech']
    assert abs(by['Tech']['margin'] - 0.25) < 1e-12
    # One sector is not a pool.
    assert _app_profit_pool([r for r in _sector_rows() if r['sector'] == 'Tech']) == []


def test_feed_writes_a_dated_sectors_file(tmp_path):
    app = tmp_path / 'app'
    pool = {'Tech': {'overview': 'Tech text', 'stats': {'hhi': 0.3}}}
    _write_app_feed(str(app), _records() + _sector_rows(), _details(), _rows(), '2026-11-05',
                    sector_pool=pool)
    sectors = json.loads((app / 'sectors.json').read_text(encoding='utf-8'))
    assert sectors['schema_version'] == _APP_SCHEMA_VERSION
    assert sectors['date'] == '2026-11-05'
    assert {r['sector'] for r in sectors['heatmap']} == {'Tech', 'Financial Services'}
    assert len(sectors['profit_pool']) == 2
    assert sectors['pools'] == pool
    # Dated, not hashed: the manifest still hashes the list feed alone.
    manifest = json.loads((app / 'manifest.json').read_text(encoding='utf-8'))
    assert set(manifest['files']) == {'app/latest.json'}


def test_a_sectors_failure_costs_only_the_sectors_file(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise ValueError('bad pool')
    monkeypatch.setattr(rh, '_build_app_sectors', boom)
    app = tmp_path / 'app'
    assert _write_app_feed(str(app), _records(), _details(), _rows(), '2026-11-05') == ['AAA', 'BBB']
    assert not (app / 'sectors.json').exists()
    assert (app / 'manifest.json').exists()


# --- the screener (app/screener.json) ---------------------------------------

def _template_table_columns():
    """Every column key of the report's Financial Data tabs (TC, plus the
    separately assigned TC.mkt), as the app's screener spec is generated."""
    src = open(os.path.join(REPO, 'templates', 'report.html'), encoding='utf-8').read()
    start = src.index('var TC={')
    block = src[start:src.index('\n};', start)]
    mkt = src.index('TC.mkt=')
    block += src[mkt:src.index('\n', mkt)]
    return {k for k in re.findall(r"\{k:'([A-Za-z0-9_]+)'", block)} - {'ticker'}


def test_screener_fields_track_the_reports_table():
    # The app builds its screener columns from the template; a column added
    # there must be published here, or it would show N/A for every company.
    assert set(_APP_SCREENER_FIELDS) == _template_table_columns()


def test_screener_packs_rows_in_ticker_order(tmp_path):
    recs = [
        {'ticker': 'ZZZ', 'rating': 'HOLD', 'pe': 12.3456789, 'country': 'US', 'r2000': True},
        {'ticker': 'AAA', 'rating': 'BUY', 'pe': float('nan'), 'mos': 0.25, '_gate_mos': None,
         'founder_led': True, 'rating_chg': 1},
    ]
    out = _build_app_screener(recs, '2026-11-05')
    assert out['date'] == '2026-11-05' and out['tickers'] == ['AAA', 'ZZZ']
    rows = unpack_rows(out['rows'])
    # Six significant digits; a NaN is absent rather than null.
    assert rows[1]['pe'] == 12.3457 and 'pe' not in rows[0]
    assert rows[0]['founder_led'] is True and rows[0]['rating_chg'] == 1
    assert rows[1]['country'] == 'US' and rows[1]['r2000'] is True
    # AAA's MoS gate is null: the value ships, and the gate is listed.
    assert rows[0]['mos'] == 0.25 and out['gate_null'] == {'mos': [0]}


def test_blue_chip_is_the_reports_rule():
    base = {'mcap': 60e9, 'operating_income': 1e9, 'roe': 0.15}
    assert _app_is_blue_chip({**base, 'nd_ebitda': 3.0})
    assert _app_is_blue_chip({**base, 'altman_z': 3.5})
    assert not _app_is_blue_chip({**base, 'nd_ebitda': 5.0, 'int_cov': 2.0})
    assert not _app_is_blue_chip({**base, 'mcap': 40e9, 'nd_ebitda': 1.0})
    assert not _app_is_blue_chip({**base, 'roe': 0.05, 'nd_ebitda': 1.0})


def test_feed_writes_a_dated_screener(tmp_path):
    app = tmp_path / 'app'
    _write_app_feed(str(app), _records(), _details(), _rows(), '2026-11-05')
    screener = json.loads((app / 'screener.json').read_text(encoding='utf-8'))
    assert screener['date'] == '2026-11-05' and screener['tickers'] == ['AAA', 'BBB']
    assert len(unpack_rows(screener['rows'])) == 2
