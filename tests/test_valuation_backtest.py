"""Historical valuation backtest (scripts/valuation_backtest.py) and the
delisted-filer fetch (scripts/fetch_delisted.py) helpers."""
import json

import numpy as np
import pandas as pd
import pytest

from scripts import fetch_delisted as fd
from scripts import growth_hindcast as gh
from scripts import valuation_backtest as vb


def test_raw_closes_undo_the_dividend_adjustment():
    # Raw prices 100 then 98 after a $2 dividend: yfinance scales the bar
    # before the ex-date by (1 - 2/100), so the adjusted series is flat.
    idx = pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03'])
    df = pd.DataFrame({'Close': [98.0, 98.0, 98.0], 'Dividends': [0.0, 0.0, 2.0]}, index=idx)
    raw = vb.raw_closes(df)
    assert raw.iloc[2] == pytest.approx(98.0)
    assert raw.iloc[0] == pytest.approx(100.0, rel=1e-3)


def test_asof_respects_the_gap():
    idx = pd.to_datetime(['2020-01-01', '2020-01-10'])
    df = pd.DataFrame({'Close': [1.0, 2.0]}, index=idx)
    assert vb.Prices.asof(df, '2020-01-05') == 1.0
    assert vb.Prices.asof(df, '2020-01-09', max_gap=3) is None
    assert vb.Prices.asof(df, '2019-12-31') is None


def test_beta_recovers_a_known_loading():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range('2018-01-01', '2020-12-31')
    m = np.cumprod(1 + rng.normal(0, 0.01, len(idx)))
    s = np.cumprod(1 + 1.5 * (np.r_[0, np.diff(m) / m[:-1]]) + rng.normal(0, 0.002, len(idx)))
    spy = pd.DataFrame({'Close': m}, index=idx)
    stk = pd.DataFrame({'Close': s}, index=idx)
    b = vb.beta(stk, spy, '2020-12-31')
    assert b == pytest.approx(2 / 3 * 1.5 + 1 / 3, abs=0.08)


def test_fair_value_uses_the_capped_base_and_net_debt():
    eh = {'operating_cf_history': {str(y): 110 for y in range(2015, 2021)},
          'capex_history': {str(y): 10 for y in range(2015, 2021)},
          'total_debt_history': {'2020': 300}, 'cash_history': {'2020': 100}}
    eq = vb.fair_value(eh, 2020, 'Industrials', 0.03, 0.09, 0.025)
    from models.dcf import two_stage_ev
    assert eq == pytest.approx(two_stage_ev(100, 0.03, 0.09, 0.025) - 200)
    eh['operating_cf_history']['2020'] = 410      # a spike is capped at 1.25x
    capped = vb.fair_value(eh, 2020, 'Industrials', 0.03, 0.09, 0.025)
    base = 1.25 * np.mean([100] * 4 + [400])
    assert capped == pytest.approx(two_stage_ev(base, 0.03, 0.09, 0.025) - 200)


def _obs(years=range(2010, 2022), n=80, seed=1, skill=0.3):
    rng = np.random.default_rng(seed)
    out = []
    for y in years:
        for i in range(n):
            mos = rng.normal()
            out.append({'ticker': f'T{i}', 'year': y, 'v1h': mos,
                        'v2h': mos + rng.normal(0, 0.1),
                        'x1': skill * mos * 0.1 + rng.normal(0, 0.1),
                        'x3': skill * mos * 0.2 + rng.normal(0, 0.2)})
    return out


def test_evaluate_finds_skill_and_none_without_it():
    res = vb.evaluate(_obs(), n_boot=200)
    assert res['v1h_1y']['mean_ic'] > 0.1 and res['v1h_1y']['ic_ci90'][0] > 0
    assert res['v1h_1y']['years'] == 12 and res['v1h_3y']['overlapping']
    null = vb.evaluate(_obs(skill=0.0), n_boot=200)
    assert null['v1h_1y']['ic_ci90'][0] < 0 < null['v1h_1y']['ic_ci90'][1]


def test_paired_compares_on_the_same_company_years():
    obs = _obs(n=120)
    for o in obs[::2]:
        o['v2h'] = None                     # only common rows count
    p = vb.paired(obs, n_boot=200)
    assert p['1y']['years'] == 12 and abs(p['1y']['mean_diff']) < 0.05


# --- fetch_delisted helpers -----------------------------------------------

IDX = """Form Type   Company Name                                                  CIK         Date Filed  File Name
---------------------------------------------------------------------------------------------------------------------------------------------
10-K        ACME CORP                                                     1234        2012-03-01  edgar/data/1234/0001.txt
10-K/A      ACME CORP                                                     1234        2012-04-01  edgar/data/1234/0002.txt
10-K        LISTED INC                                                    320193      2012-03-02  edgar/data/320193/0003.txt
10-Q        OTHER CO                                                      999         2012-05-01  edgar/data/999/0004.txt
"""


def test_index_parsing_and_candidate_filter(tmp_path, monkeypatch):
    monkeypatch.setattr(fd, 'INDEX_DIR', str(tmp_path))
    for q in (1, 2, 3, 4):
        (tmp_path / f'2012Q{q}.idx').write_text(IDX if q == 1 else '', encoding='utf-8')
    filers = fd.ten_k_filers([2012], 'x@example.com')
    assert filers == {'0000001234': 'ACME CORP', '0000320193': 'LISTED INC'}
    issuers = tmp_path / 'issuers.csv'
    issuers.write_text('ticker,cik,exchange,rank\nAAPL,320193,Nasdaq,0\n', encoding='utf-8')
    monkeypatch.setattr(fd, 'listed_ciks', lambda path=str(issuers): {'0000320193'})
    assert fd.candidates([2012], 'x@example.com') == [('0000001234', 'ACME CORP')]


def test_already_fetched_and_extra_rows(tmp_path):
    p = tmp_path / 'h.jsonl'
    p.write_text(json.dumps({'cik': '1', 'ticker': 'CIK1', 'edgar_history': None}) + '\n'
                 + json.dumps({'cik': '2', 'ticker': 'CIK2', 'edgar_history': {'a': 1}}) + '\n',
                 encoding='utf-8')
    assert fd.already_fetched(str(p)) == {'1', '2'}
    assert [r['ticker'] for r in gh.load_extra_rows(str(p))] == ['CIK2']


def test_survivorship_report_shows_the_gap():
    def rows(g, n, tag):
        out = []
        for i in range(n):
            fcf = {str(2006 + k): 100 * (1 + g) ** k + 10 for k in range(14)}
            out.append({'ticker': f'{tag}{i}', 'sector': 'Industrials', 'edgar_history': {
                'operating_cf_history': fcf, 'capex_history': {y: 10 for y in fcf},
                'revenue_history': {y: 1000 * (1 + g) ** (int(y) - 2006) for y in fcf}}})
        return out
    rep = gh.survivorship_report(rows(0.08, 40, 'S'), rows(0.0, 50, 'D'))
    assert rep['per_origin'] and all(p['fcf'][0] > p['fcf'][1] for p in rep['per_origin'])
    assert rep['base_rate_all']['with_delisted'] < rep['base_rate_all']['survivors']
