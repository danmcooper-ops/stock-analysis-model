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
                                 _app_pool_companies,
                                 _build_app_portfolios, _app_portfolio_news, _pfn_matchers,
                                 _app_is_blue_chip, _app_profit_pool, _app_sector_heatmap,
                                 _build_app_etfs, _build_app_gates, _build_app_list,
                                 _build_app_screener,
                                 _write_app_feed,
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


def test_list_feed_carries_the_prior_score_and_the_history_dates():
    recs = _records()
    recs[1].update(score_prev=85.5, score_chg=2.5, score_prev_basis='rescored',
                   score_hist=[80, 85])
    feed = _build_app_list(recs, '2026-11-05', 0.0484, ['2026-10-24', '2026-10-31'])
    assert feed['score_hist_dates'] == ['2026-10-24', '2026-10-31']
    aaa = feed['results'][0]
    assert (aaa['score_prev'], aaa['score_chg'], aaa['score_prev_basis']) == \
        (85.5, 2.5, 'rescored')
    # The history is per company, so it rides the detail shard, not the list.
    assert 'score_hist' not in aaa
    assert _build_app_list(_records(), '2026-11-05', None)['score_hist_dates'] == []


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
    assert 'for f in sectors.json screener.json gates.json etfs.json portfolios.json; do' in sh
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
    # Each pool as built, plus its largest companies.
    assert sectors['pools']['Tech']['overview'] == 'Tech text'
    assert [c['ticker'] for c in sectors['pools']['Tech']['companies']['shown']] == ['A1', 'A2']
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
    # The template's tabs lead with the composite score where they once led
    # with the rating: the score rides latest.json (the app's list feed), and
    # 'rating' stays in this feed (every rating field is still published).
    assert set(_APP_SCREENER_FIELDS) - {'rating'} | {'_composite_score'} == \
        _template_table_columns()
    assert 'rating' in _APP_SCREENER_FIELDS and '_composite_score' in _APP_LIST_FIELDS


def test_screener_packs_rows_in_ticker_order(tmp_path):
    recs = [
        {'ticker': 'ZZZ', 'rating': 'HOLD', 'pe': 12.3456789, 'country': 'US', 'r2000': True,
         'mcap': 5.5e9, 'net_debt': -1e8, 'earnings_next_date': '2026-11-10',
         'revenue': 2e9, 'operating_income': -3e7, 'pp_revenue_share': 0.0123, 'pp_sector_cr4': 0.41},
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
    # The Filters panel's Debt filters, which are not table columns.
    assert rows[1]['mcap'] == 5.5e9 and rows[1]['net_debt'] == -1e8
    # The portfolio Summary's earnings-within-a-week count.
    assert rows[1]['earnings_next_date'] == '2026-11-10'
    # A sector's Trading tab.
    assert rows[1]['revenue'] == 2e9 and rows[1]['operating_income'] == -3e7
    assert rows[1]['pp_revenue_share'] == 0.0123 and rows[1]['pp_sector_cr4'] == 0.41
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


# --- the gates (app/gates.json) -------------------------------------------
# The Gate Matrix and the Stocks Dashboard's pass rates, which the report
# reads from DATA and GM.

_GM = {
    'gates': [
        {'key': '_gate_mos', 'gpKey': '_gp_mos', 'scoreKey': '_score_mos', 'label': 'MoS',
         'category': 'Valuation', 'fmt': 'pct1', 'threshold': 'MoS > 10%', 'weight': 2.0},
        {'key': '_gate_roic', 'gpKey': '_gp_roic', 'scoreKey': '_score_roic', 'label': 'ROIC',
         'category': 'Quality', 'fmt': 'pct1', 'threshold': 'ROIC > 12%', 'weight': 1.0},
    ],
    'categories': [{'name': 'Valuation', 'scoreKey': '_score_valuation', 'weight': 0.5},
                   {'name': 'Quality', 'scoreKey': '_score_quality', 'weight': 0.5}],
    'ratings': ['BUY', 'LEAN BUY', 'HOLD', 'PASS'],
}


def test_gates_pack_values_and_pass_flags_by_gate_in_ticker_order():
    recs = [
        {'ticker': 'ZZZ', '_gate_mos': 0.123456789, '_gp_mos': True, '_gate_roic': None, '_gp_roic': None},
        {'ticker': 'AAA', '_gate_mos': float('nan'), '_gp_mos': 0, '_gate_roic': 0.2, '_gp_roic': 1},
        {'company_name': 'no ticker'},
    ]
    out = _build_app_gates(recs, '2026-11-06', _GM)
    assert out['date'] == '2026-11-06' and out['tickers'] == ['AAA', 'ZZZ']
    assert out['gate_meta'] is _GM
    # Six significant digits; a NaN is null, never a JavaScript literal.
    assert out['values']['_gate_mos'] == [None, 0.123457]
    assert out['values']['_gate_roic'] == [0.2, None]
    # Pass flags are booleans, null where the gate does not apply.
    assert out['pass']['_gp_mos'] == [False, True]
    assert out['pass']['_gp_roic'] == [True, None]
    # Every column is as long as the ticker list, so the app can index it.
    for col in list(out['values'].values()) + list(out['pass'].values()):
        assert len(col) == len(out['tickers'])


def test_gates_default_to_the_reports_gate_metadata():
    from scripts.scoring import gate_metadata
    out = _build_app_gates([{'ticker': 'AAA'}], '2026-11-06')
    assert out['gate_meta'] == gate_metadata()
    assert set(out['values']) == {g['key'] for g in gate_metadata()['gates']}
    assert set(out['pass']) == {g['gpKey'] for g in gate_metadata()['gates']}


def test_feed_writes_a_dated_gates_file(tmp_path):
    app = tmp_path / 'app'
    _write_app_feed(str(app), _records(), _details(), _rows(), '2026-11-05')
    gates = json.loads((app / 'gates.json').read_text(encoding='utf-8'))
    assert gates['date'] == '2026-11-05' and gates['tickers'] == ['AAA', 'BBB']
    assert gates['gate_meta']['gates']


# --- ETFs (app/etfs.json) --------------------------------------------------

def _etf_payload():
    return {'base': 'SPY', 'sector_fund': {'Technology': 'XLK'},
            'funds': {'XLK': {'fund': 'XLK', 'status': 'ok', 'mos': float('nan'), 'sector': 'Technology',
                              'category': 'sector', 'top': [{'t': 'AAPL', 'w': 0.12}]}}}


def test_etfs_are_the_look_through_dated():
    out = _build_app_etfs(_etf_payload(), '2026-11-06')
    assert out['date'] == '2026-11-06' and out['schema_version'] == _APP_SCHEMA_VERSION
    assert out['base'] == 'SPY' and out['sector_fund'] == {'Technology': 'XLK'}
    assert out['funds']['XLK']['top'] == [{'t': 'AAPL', 'w': 0.12}]


def test_no_look_through_means_no_etf_file(tmp_path):
    assert _build_app_etfs(None, '2026-11-06') is None
    assert _build_app_etfs({'base': 'SPY', 'funds': {}}, '2026-11-06') is None
    app = tmp_path / 'app'
    _write_app_feed(str(app), _records(), _details(), _rows(), '2026-11-05')
    assert not (app / 'etfs.json').exists()


def test_feed_writes_a_dated_etf_file_with_nan_nulled(tmp_path):
    app = tmp_path / 'app'
    _write_app_feed(str(app), _records(), _details(), _rows(), '2026-11-05', etfs=_etf_payload())
    text = (app / 'etfs.json').read_text(encoding='utf-8')
    assert 'NaN' not in text
    etfs = json.loads(text)
    assert etfs['date'] == '2026-11-05' and etfs['funds']['XLK']['mos'] is None


# --- portfolios (app/portfolios.json) --------------------------------------

def _pf_payload():
    return {
        'rev': 'abc', 'prev_date': '2026-11-04', 'changes': [], 'events': [],
        'portfolios': [
            {'id': 'semis', 'name': 'Semiconductors', 'color': '#2a78d6', 'description': '',
             'created': '2026-09-26', 'tickers': ['NVDA', 'TSM', 'ZZZ'], 'exclude': [], 'rule': None,
             'alerts': 'buy_line', 'missing': ['ZZZ'],
             # Anything not on the allow-list stays out of the app's copy.
             'targets': {'NVDA': 0.5}},
            {'id': 'paper', 'name': 'Paper book', 'color': '#2a78d6',
             'rule': {'rating': ['BUY']}, 'alerts': 'all', 'missing': []},
        ],
        'nav': {'spy': [['2026-11-04', 100.0], ['2026-11-05', 101.0]], 'universe': [],
                'pf': {'semis': [['2026-11-04', 100.0, 0], ['2026-11-05', 97.0, 0]]}},
        'analytics': {'semis': {'val': {'status': 'ok', 'rel': 0.1}}},
    }


def _template_set(name):
    src = open(os.path.join(REPO, 'templates', 'report.html'), encoding='utf-8').read()
    body = re.search(r'var ' + name + r'=\{(.*?)\};', src).group(1)
    return {k.strip("'") for k in re.findall(r"('[^']*'|[A-Za-z0-9_]+):1", body)}


def test_news_word_lists_are_the_templates():
    assert set(rh._PFN_LEGAL) == _template_set('_PFN_LEGAL')
    assert set(rh._PFN_GENERIC) == _template_set('_PFN_GENERIC')
    assert set(rh._PFN_STOPTK) == _template_set('_PFN_STOPTK')
    assert str(rh._PFN_MAX) in re.search(r'var _PFN_MAX=(\d+);', open(
        os.path.join(REPO, 'templates', 'report.html'), encoding='utf-8').read()).group(1)


def _names(m, text):
    return any(r.search(text) for r in m)


def test_news_matchers_follow_the_reports_rules():
    nv = _pfn_matchers('NVDA', {'company_name': 'NVIDIA Corporation'})
    assert _names(nv, 'Nvidia beats') and _names(nv, '$NVDA jumps')
    # Tickers are case-sensitive and whole: no "nvda", no XNVDA or NVDAQ.
    assert not _names(nv, 'nvda lower') and not _names(nv, 'XNVDA') and not _names(nv, 'NVDAQ moves')
    # A longer name matches by its full name or first two words, never one.
    ttd = _pfn_matchers('TTD', {'company_name': 'The Trade Desk, Inc.'})
    assert _names(ttd, 'Trade Desk stock') and not _names(ttd, 'Trade war')
    # Stop-listed and one-letter tickers never match on their own.
    assert not _names(_pfn_matchers('ON', {'company_name': 'ON Semiconductor Corp'}), 'turned on today')
    # A generic one-word name does not match alone (Ford the verb, Target).
    assert not _names(_pfn_matchers('TGT', {'company_name': 'Target Corporation'}), 'price target raised')
    # A share-class suffix is dropped from the ticker.
    assert _names(_pfn_matchers('BRK-B', {'company_name': 'Berkshire Hathaway Inc.'}), 'BRK results')


def test_portfolio_news_folds_credits_filters_and_sorts():
    by_tk = {'AAA': {'company_name': 'Alphacorp Inc.'}, 'BBB': {'company_name': 'Betaworks Ltd'}}
    details = {
        'AAA': {'news_headlines': [
            {'title': 'Alphacorp and Betaworks merge', 'link': 'L1', 'date': '2026-11-05', 'tier': 0},
            {'title': 'Sector rally', 'link': 'L2', 'date': '2026-11-06', 'scope': 'sector'},
            {'title': 'Markets drift', 'link': 'L3', 'date': '2026-11-06', 'origin': 'google_news'},
            {'title': 'Alphacorp upgraded', 'link': 'L4', 'date': '2026-11-06', 'timestamp': 1, 'tags': ['rating']},
            {'title': 'Unrelated', 'link': 'L5', 'date': '2026-11-06'},
        ]},
        'BBB': {'news_headlines': [
            {'title': 'Alphacorp and Betaworks merge', 'link': 'L1', 'date': '2026-11-05', 'tier': 1},
            {'title': 'Company news for Betaworks', 'link': 'L6', 'date': '2026-11-06', 'timestamp': 2},
            {'title': 'Earnings call held', 'link': 'L7', 'date': '2026-11-04', 'tier': 1},
        ]},
    }
    news = _app_portfolio_news(['AAA', 'BBB'], by_tk, details)
    titles = [i['title'] for i in news['items']]
    # Sector-scope and Google items without a scope are dropped, as is one
    # that names no member; the duplicate is folded.
    assert 'Sector rally' not in titles and 'Markets drift' not in titles and 'Unrelated' not in titles
    assert titles.count('Alphacorp and Betaworks merge') == 1
    # Newest day first; within a day the later timestamp first.
    assert titles == ['Company news for Betaworks', 'Alphacorp upgraded',
                      'Alphacorp and Betaworks merge', 'Earnings call held']
    merged = news['items'][2]
    assert merged['tickers'] == ['AAA', 'BBB'] and merged['link'] == 'L1'
    # A tier 0/1 item is credited to the member whose feed it came from.
    assert news['items'][3]['tickers'] == ['BBB']
    assert news['items'][1]['tags'] == ['rating'] and news['total'] == 4


def test_portfolio_news_keeps_the_newest_sixty():
    by_tk = {'AAA': {'company_name': 'Alphacorp Inc.'}}
    feed = [{'title': f'Alphacorp item {i}', 'link': f'L{i}', 'date': f'2026-10-{i % 28 + 1:02d}'} for i in range(80)]
    news = _app_portfolio_news(['AAA'], by_tk, {'AAA': {'news_headlines': feed}})
    assert news['total'] == 80 and len(news['items']) == rh._PFN_MAX
    assert news['items'][0]['date'] >= news['items'][-1]['date']


def test_portfolios_carry_only_the_published_fields():
    index = {'NVDA': ['semis'], 'TSM': ['semis'], 'AAA': ['paper']}
    out = _build_app_portfolios(_pf_payload(), index, '2026-11-05')
    assert out['date'] == '2026-11-05' and out['schema_version'] == _APP_SCHEMA_VERSION
    semis, paper = out['portfolios']
    assert set(semis) == {'id', 'name', 'color', 'description', 'created', 'alerts',
                          'members', 'missing'}
    assert 'targets' not in semis and 'tickers' not in semis
    # Members as the engine resolved them (rules included), sorted.
    assert semis['members'] == ['NVDA', 'TSM'] and semis['missing'] == ['ZZZ']
    assert paper['members'] == ['AAA'] and paper['rule'] == {'rating': ['BUY']}
    # The public NAV, as the report's payload carries it.
    assert out['nav']['pf']['semis'][-1] == ['2026-11-05', 97.0, 0]
    # Value & risk and the changes' comparison day, as PF_PUB carries them;
    # no rows given, so no news.
    assert out['analytics'] == {'semis': {'val': {'status': 'ok', 'rel': 0.1}}}
    assert out['prev_date'] == '2026-11-04'
    assert 'news' not in semis and out['news_tag_labels']


def test_no_portfolios_means_no_file(tmp_path):
    assert _build_app_portfolios({'portfolios': []}, {}, '2026-11-05') is None
    app = tmp_path / 'app'
    _write_app_feed(str(app), _records(), _details(), _rows(), '2026-11-05')
    assert not (app / 'portfolios.json').exists()


def test_feed_writes_portfolios_when_given(tmp_path):
    app = tmp_path / 'app'
    _write_app_feed(str(app), _records(), _details(), _rows(), '2026-11-05',
                    portfolios=_pf_payload(), pf_index={'AAA': ['semis']})
    pfs = json.loads((app / 'portfolios.json').read_text(encoding='utf-8'))
    assert [p['id'] for p in pfs['portfolios']] == ['semis', 'paper']
    assert pfs['portfolios'][0]['members'] == ['AAA']
    # The rows are there, so each portfolio carries its News tab.
    assert pfs['portfolios'][0]['news'] == {'items': [], 'total': 0}


def test_pool_companies_are_the_top_25_and_the_rest_pooled():
    rows = [{'ticker': f'T{i:02d}', 'pp_revenue_share': (30 - i) / 1000, 'pp_profit_share': 0.01,
             'operating_margin': 0.10 if i < 28 else 0.40, 'rating': 'HOLD', 'mcap': 1e9,
             '_gate_pool_share': 0.02} for i in range(30)]
    out = _app_pool_companies(rows)
    assert out['n'] == 30 and len(out['shown']) == 25
    # Largest by revenue share first; the rest pooled.
    assert out['shown'][0]['ticker'] == 'T00' and out['others']['n'] == 5
    t = out['shown'][0]
    assert (t['revenue_share'], t['pool_share'], t['margin'], t['pool_share_trend']) == (0.03, 0.01, 0.10, 0.02)
    # Others' margin is revenue-weighted, as the report pools it: T25-T27 at
    # 10% on 5, 4, 3 and T28-T29 at 40% on 2, 1 (thousandths).
    assert abs(out['others']['margin'] - (0.10 * 12 + 0.40 * 3) / 15) < 1e-12
    assert out['mcap_total'] == 30e9 and out['others']['mcap'] == 5e9


def test_a_small_sector_has_no_others():
    out = _app_pool_companies([{'ticker': 'A', 'pp_revenue_share': 1.0}])
    assert len(out['shown']) == 1 and 'others' not in out
