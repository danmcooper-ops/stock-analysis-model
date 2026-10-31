"""SPDR look-through: holdings parsing (data/etf_holdings.py), the math
(models/etf_lookthrough.py) and the render-time payload (report_html)."""
import json
import math
from datetime import date

import openpyxl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from data import etf_holdings
from models.etf_lookthrough import FVP_CLIP, lookthrough, relative_to


def _ssga_xlsx(path, fund='XLK', as_of='20-Oct-2026', lines=None):
    """A workbook in SSGA's daily-holdings layout: four header lines, a blank,
    the column row, holdings, then the disclaimer footer."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(['Fund Name:', f'State Street® Test {fund} ETF'])
    ws.append(['Ticker Symbol:', fund])
    ws.append(['Holdings:', f'As of {as_of}'])
    ws.append([])
    ws.append(['Name', 'Ticker', 'Identifier', 'SEDOL', 'Weight', 'Sector', 'Shares Held', 'Local Currency'])
    for row in lines:
        ws.append(row)
    ws.append([])
    ws.append(['Distributor: State Street Global Advisors Funds Distributors, LLC ...'])
    wb.save(path)
    return path


def _lines(n=12):
    rows = [[f'CO {i}', f'C{i}', f'00000{i:04d}', '-', 90.0 / (n - 1), '-', 1000, 'USD']
            for i in range(n - 1)]
    rows[0][1] = 'BRK.B'
    rows.append(['SSI US GOV MONEY MARKET CLASS', '-', '-', '-', 9.5, '-', 1, 'USD'])
    rows.append(['US DOLLAR', '-', '-', '-', 0.5, '-', 1, 'USD'])
    return rows


def test_parse_ssga_layout(tmp_path):
    p = _ssga_xlsx(tmp_path / 'x.xlsx', lines=_lines())
    h = etf_holdings.parse_ssga_xlsx(str(p), fund='XLK')
    assert h['fund'] == 'XLK' and h['as_of'] == '2026-10-20'
    assert len(h['holdings']) == 13
    assert h['holdings'][0]['ticker'] == 'BRK-B'          # dot -> dash
    kinds = [x['kind'] for x in h['holdings']]
    assert kinds.count('cash') == 2 and kinds.count('security') == 11
    assert math.isclose(sum(x['weight'] for x in h['holdings']), 1.0)


def test_parse_refuses_partial_file(tmp_path):
    p = _ssga_xlsx(tmp_path / 'x.xlsx', lines=_lines()[:6])   # weights sum ~45%
    with pytest.raises(etf_holdings.HoldingsError):
        etf_holdings.parse_ssga_xlsx(str(p), fund='XLK')


def test_parse_refuses_wrong_fund(tmp_path):
    p = _ssga_xlsx(tmp_path / 'x.xlsx', fund='XLE', lines=_lines())
    with pytest.raises(etf_holdings.HoldingsError):
        etf_holdings.parse_ssga_xlsx(str(p), fund='XLK')


def test_archive_is_point_in_time(tmp_path):
    base = etf_holdings.parse_ssga_xlsx(str(_ssga_xlsx(tmp_path / 'a.xlsx', lines=_lines())), fund='XLK')
    for d in ('2026-10-16', '2026-10-20', '2026-10-21'):
        etf_holdings.save_holdings(str(tmp_path / 'root'), dict(base, as_of=d))
    root = str(tmp_path / 'root')
    assert etf_holdings.load_holdings(root, 'XLK', date(2026, 10, 20))['as_of'] == '2026-10-20'
    assert etf_holdings.load_holdings(root, 'XLK', '2026-10-19')['as_of'] == '2026-10-16'
    assert etf_holdings.load_holdings(root, 'XLK', '2026-10-01') is None
    assert etf_holdings.load_holdings(root, 'XLK')['as_of'] == '2026-10-21'
    # Deterministic: re-saving the same day rewrites identical bytes.
    p = etf_holdings.save_holdings(root, dict(base, as_of='2026-10-20'))
    b1 = open(p, 'rb').read()
    etf_holdings.save_holdings(root, dict(base, as_of='2026-10-20'))
    assert open(p, 'rb').read() == b1


def _fund(lines):
    return {'fund': 'TST', 'as_of': '2026-10-20', 'holdings': [
        {'ticker': t, 'name': t, 'weight': w, 'kind': k} for t, w, k in lines]}


def _rows(spec):
    return {t: {'ticker': t, 'price': p, '_fv_effective': fv, 'rating': 'HOLD', 'sector': 'S'}
            for t, p, fv in spec}


def test_value_is_weighted_fvp_not_mean_mos():
    rows = _rows([('A', 100, 200), ('B', 100, 50)])
    r = lookthrough(_fund([('A', 0.5, 'security'), ('B', 0.5, 'security')]), rows)
    assert r['status'] == 'ok'
    assert math.isclose(r['V'], 1.25)                 # 0.5*2 + 0.5*0.5
    assert math.isclose(r['mos'], 1 - 1 / 1.25)       # not mean(0.5, -1.0)


def test_share_classes_fold_onto_kept_row():
    rows = _rows([('GOOGL', 100, 50), ('X', 100, 100)])
    f = _fund([('GOOG', 0.3, 'security'), ('GOOGL', 0.3, 'security'), ('X', 0.4, 'security')])
    r = lookthrough(f, rows, aliases={'GOOG': 'GOOGL'})
    assert [e['t'] for e in r['top']].count('GOOGL') == 1
    assert math.isclose(r['top'][0]['w'], 0.6)


def test_missing_carried_cash_and_unmatched():
    rows = _rows([('A', 100, 100), ('B', 100, None), ('C', 100, None)])
    f = _fund([('A', 0.5, 'security'), ('B', 0.2, 'security'), ('C', 0.1, 'security'),
               ('', 0.1, 'cash'), ('Z', 0.1, 'security')])
    r = lookthrough(f, rows, prev_fv={'B': 300}, min_coverage=0.5)
    assert math.isclose(r['carried_w'], 0.2) and math.isclose(r['missing_w'], 0.1)
    assert math.isclose(r['cash_w'], 0.1) and math.isclose(r['unmatched_w'], 0.1)
    # In scope: A 0.5@1, B 0.2@3 (carried, at the clip), C 0.1@1, cash 0.1@1.
    assert math.isclose(r['V'], (0.5 + 0.6 + 0.1 + 0.1) / 0.9)
    assert math.isclose(r['V_low'], (0.5 + 0.6 + 0.1 / 3 + 0.1) / 0.9)


def test_insufficient_below_coverage():
    rows = _rows([('A', 100, 100), ('B', 100, None)])
    r = lookthrough(_fund([('A', 0.5, 'security'), ('B', 0.5, 'security')]), rows)
    assert r['status'] == 'insufficient' and 'V' not in r


def test_relative_to_base():
    rows = _rows([('A', 100, 100), ('B', 100, 120)])
    res = {'SPY': lookthrough(_fund([('A', 1.0, 'security')]), rows),
           'X': lookthrough(_fund([('B', 1.0, 'security')]), rows)}
    relative_to(res)
    assert res['SPY']['rel'] == 0 and math.isclose(res['X']['rel'], 0.2)


_pos = st.floats(min_value=0.01, max_value=1e4, allow_nan=False)


@settings(max_examples=150, deadline=None)
@given(st.lists(st.tuples(_pos, _pos, st.one_of(st.none(), _pos), _pos), min_size=1, max_size=20),
       st.floats(min_value=0.1, max_value=10))
def test_properties(spec, split):
    tks = [f'T{i}' for i in range(len(spec))]
    rows = _rows([(t, p, fv) for t, (w, p, fv, _) in zip(tks, spec, strict=True)])
    f = _fund([(t, w, 'security') for t, (w, _p, _fv, _x) in zip(tks, spec, strict=True)])
    r = lookthrough(f, rows, min_coverage=0.0)
    lo, hi = FVP_CLIP
    # Coverage shares partition the fund.
    assert math.isclose(r['valued_w'] + r['missing_w'] + r['unmatched_w'] - r['cash_w'], 1.0,
                        rel_tol=1e-9, abs_tol=1e-9)
    # The clip bounds V, and the low band never exceeds the base value.
    assert lo - 1e-12 <= r['V_low'] <= r['V'] <= hi + 1e-12
    # Weight scale is irrelevant: the same fund at any multiple reads the same.
    f2 = _fund([(t, w * split, 'security') for t, (w, _p, _fv, _x) in zip(tks, spec, strict=True)])
    assert math.isclose(lookthrough(f2, rows, min_coverage=0.0)['V'], r['V'], rel_tol=1e-9)
    # Uniform drift (every holding up by the same return) leaves V unchanged.
    d = lookthrough(f, rows, drift={t: split for t in tks}, min_coverage=0.0)
    assert math.isclose(d['V'], r['V'], rel_tol=1e-9)
    # Mix shares sum to 1.
    assert math.isclose(sum(r['rating_mix'].values()), 1.0)


def test_payload_and_degraded_suppression(tmp_path, monkeypatch):
    from scripts import report_html
    monkeypatch.delenv('ETF_HOLDINGS_DIR', raising=False)
    root = tmp_path / 'etf_holdings'
    for fund in ('SPY', 'XLK'):
        etf_holdings.save_holdings(str(root), {
            'fund': fund, 'fund_name': fund, 'as_of': '2026-10-20', 'source': 'ssga',
            'holdings': [{'ticker': 'A', 'name': 'A', 'weight': 0.6, 'kind': 'security'},
                         {'ticker': 'B' if fund == 'SPY' else 'A2', 'name': 'B', 'weight': 0.4,
                          'kind': 'security'}]})
    rows = [{'ticker': 'A', 'price': 10, '_fv_effective': 10, 'rating': 'BUY', 'sector': 'Technology',
             'listing_aliases': ['A2']},
            {'ticker': 'B', 'price': 10, '_fv_effective': 5, 'rating': 'PASS', 'sector': 'Energy'}]
    p = report_html._build_etf_payload(rows, str(tmp_path), date(2026, 10, 20))
    assert set(p['funds']) == {'SPY', 'XLK'}
    assert p['sector_fund'] == {'Technology': 'XLK'}
    assert math.isclose(p['funds']['XLK']['rel'], 1.0 / 0.8 - 1)
    saved = json.load(open(tmp_path / 'etfs_2026-10-20.json', encoding='utf-8'))
    assert saved['date'] == '2026-10-20' and 'XLK' in saved['funds']
    assert report_html._build_etf_payload(rows, str(tmp_path), date(2026, 10, 20),
                                          run_provenance={'coverage': {'degraded': True}}) is None
    assert report_html._build_etf_payload(rows, str(tmp_path / 'none'), date(2026, 10, 20)) is None


def test_registry_is_well_formed():
    reg = etf_holdings.load_registry()
    assert set(etf_holdings.SPDR_FUNDS) <= set(reg)
    assert len(reg) > len(etf_holdings.SPDR_FUNDS)
    assert etf_holdings.fund_tickers() == list(etf_holdings.ETF_FUNDS)
    sectors = {s for s in etf_holdings.SPDR_FUNDS.values() if s}
    for t, e in reg.items():
        assert e['ticker'] == t and e['category'] in etf_holdings.CATEGORIES
        # Only the Select Sectors carry a sector: sector pages key on it.
        assert e['sector'] == etf_holdings.SPDR_FUNDS.get(t)
        assert e['sector'] is None or e['sector'] in sectors


def test_registry_keeps_the_14_whatever_the_file(tmp_path):
    assert set(etf_holdings.load_registry(str(tmp_path / 'missing.json'))) == set(etf_holdings.SPDR_FUNDS)
    p = tmp_path / 'r.json'
    p.write_text(json.dumps({'funds': [
        {'ticker': 'KRE', 'name': 'Regional Banking', 'category': 'industry', 'sector': None},
        {'ticker': 'XLK', 'name': 'Technology', 'category': 'broad', 'sector': 'Energy'}]}), encoding='utf-8')
    reg = etf_holdings.load_registry(str(p))
    assert reg['KRE']['category'] == 'industry' and reg['KRE']['sector'] is None
    assert reg['XLK']['sector'] == 'Technology' and reg['XLK']['category'] == 'sector'
    assert set(etf_holdings.SPDR_FUNDS) < set(reg)


def test_fetch_defaults_to_the_registry(monkeypatch, tmp_path):
    from scripts import fetch_etf_holdings
    seen = []
    monkeypatch.setattr(etf_holdings, 'fetch_ssga', lambda t: seen.append(t) or (_ for _ in ()).throw(OSError('x')))
    assert fetch_etf_holdings.main(['--out', str(tmp_path), '--delay', '0']) == 1
    assert seen == etf_holdings.fund_tickers()


def test_update_list_matching_and_categories():
    from scripts import update_etf_list as u
    h = {'holdings': [{'ticker': 'A', 'weight': 0.5, 'kind': 'security'},
                      {'ticker': 'A2', 'weight': 0.2, 'kind': 'security'},
                      {'ticker': 'ZZ', 'weight': 0.2, 'kind': 'security'},
                      {'ticker': '', 'weight': 0.1, 'kind': 'cash'}]}
    m, cash = u.matched_weight(h, {'A': {}}, {'A2': 'A'})
    assert math.isclose(m, 0.8) and math.isclose(cash, 0.1)
    assert u.categorize('XLK', 'Technology Select Sector', set()) == 'sector'
    assert u.categorize('SDY', 'SPDR S&P Dividend ETF', {'equity'}) == 'dividend'
    assert u.categorize('KRE', 'SPDR S&P Regional Banking', {'Sectors---Industries'}) == 'industry'
    assert u.categorize('SPYG', 'Portfolio S&P 500 Growth', {'equity-style', 'Growth'}) == 'style'
    assert u.categorize('DECO', 'Galaxy Digital Asset Ecosystem', {'equity-style', 'Growth'}) == 'thematic'
    assert u.categorize('SPSM', 'Portfolio S&P 600 Small Cap', {'Small-Cap'}) == 'size'
    assert u.categorize('SPTM', 'Portfolio S&P 1500 Composite', {'Core'}) == 'broad'
    finder = {'data': {'funds': {'etfs': {
        'datas': [{'fundTicker': 'KRE®', 'fundName': 'SPDR® S&P® Regional Banking ETF'}],
        'categories': [{'key': 'assetclass', 'funds': 'KRE', 'subCategories': [
            {'key': 'equity', 'funds': 'KRE', 'subCategories': [
                {'key': 'Sectors---Industries', 'funds': 'KRE'}]}]}]}}}}
    names, tags = u.parse_finder(finder)
    assert names == {'KRE': 'SPDR S&P Regional Banking ETF'}
    assert tags['KRE'] == {'assetclass', 'equity', 'Sectors---Industries'}


def test_payload_carries_category_and_keeps_sector_fund(tmp_path, monkeypatch):
    from scripts import report_html
    monkeypatch.delenv('ETF_HOLDINGS_DIR', raising=False)
    monkeypatch.setattr(etf_holdings, 'ETF_FUNDS', dict(
        etf_holdings.load_registry(str(tmp_path / 'none.json')),
        KRE={'ticker': 'KRE', 'name': 'Regional Banking', 'category': 'industry', 'sector': None}))
    root = tmp_path / 'etf_holdings'
    for fund in ('SPY', 'XLF', 'KRE'):
        etf_holdings.save_holdings(str(root), {
            'fund': fund, 'fund_name': fund, 'as_of': '2026-10-20', 'source': 'ssga',
            'holdings': [{'ticker': 'A', 'name': 'A', 'weight': 1.0, 'kind': 'security'}]})
    rows = [{'ticker': 'A', 'price': 10, '_fv_effective': 10, 'rating': 'BUY', 'sector': 'Financial Services'}]
    p = report_html._build_etf_payload(rows, str(tmp_path), date(2026, 10, 20))
    assert set(p['funds']) == {'SPY', 'XLF', 'KRE'}
    assert p['funds']['KRE']['category'] == 'industry' and p['funds']['KRE']['sector'] is None
    assert p['funds']['XLF']['category'] == 'sector'
    assert p['sector_fund'] == {'Financial Services': 'XLF'}
