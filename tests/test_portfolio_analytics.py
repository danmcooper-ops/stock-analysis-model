"""Portfolio look-through valuation, ex-ante risk and NAV attribution."""
import math
import os

import numpy as np
import pandas as pd
import pytest

from data import portfolio_nav as pn
from models import portfolio_analytics as pa
from models import portfolio_groups as pg


def _pf(**kw):
    base = {'id': 'p', 'name': 'P', 'tickers': [], 'exclude': [], 'rule': None}
    base.update(kw)
    return pg.normalize_portfolio(base, today='2026-09-01')


@pytest.fixture(scope='module')
def market(tmp_path_factory):
    """Business days 2007..2026: SPY a random walk, LEV twice SPY's daily
    return, NOISE independent, LATE listed in 2021, CRASH halving in 2020."""
    d = tmp_path_factory.mktemp('px')
    idx = pd.bdate_range('2007-01-01', '2026-09-11')
    rng = np.random.default_rng(7)
    spy_r = rng.normal(0.0004, 0.01, len(idx))
    noise_r = rng.normal(0.0004, 0.01, len(idx))

    def write(t, rets, start=None):
        s = pd.Series(100.0 * np.cumprod(1.0 + rets), index=idx, name='Close')
        if start:
            s = s[s.index >= start]
        s.index.name = 'Date'
        s.to_frame().to_parquet(os.path.join(d, f'{t}.parquet'))
        return s
    closes = {'SPY': write('SPY', spy_r), 'LEV': write('LEV', 2 * spy_r),
              'NOISE': write('NOISE', noise_r), 'LATE': write('LATE', spy_r, '2021-01-04')}
    crash = spy_r.copy()
    crash[(idx >= '2020-02-03') & (idx < '2020-03-02')] = math.pow(0.5, 1 / 20) - 1
    closes['CRASH'] = write('CRASH', crash)
    return str(d), closes


class TestValuation:
    def test_lookthrough_with_cash_and_relative(self):
        by = {'A': {'ticker': 'A', 'price': 10.0, '_fv_effective': 20.0},
              'B': {'ticker': 'B', 'price': 10.0, '_fv_effective': 10.0}}
        v = pa.valuation({'A': 0.25, 'B': 0.25, pa.CASH: 0.5}, by)
        assert v['status'] == 'ok' and v['V'] == pytest.approx(0.25 * 2 + 0.25 + 0.5)
        uni = pa.valuation({'A': 1, 'B': 1}, by)
        s = pa.valuation_summary(v, uni)
        assert s['rel'] == pytest.approx(1.25 / 1.5 - 1, abs=1e-4)
        assert s['lifts'][0]['t'] == 'A'

    def test_insufficient_fair_values(self):
        by = {'A': {'ticker': 'A', 'price': 10.0}}
        s = pa.valuation_summary(pa.valuation({'A': 1.0}, by), None)
        assert s['status'] == 'insufficient' and 'V' not in s


class TestRisk:
    def test_leveraged_member_has_beta_two(self, market):
        _, closes = market
        r = pa.risk({'LEV': 1.0}, closes, rf=0.04)
        assert r['status'] == 'ok' and r['beta'] == pytest.approx(2.0, abs=0.1)
        assert r['vol'] == pytest.approx(2 * 0.01 * math.sqrt(252), rel=0.15)
        assert r['sharpe'] == pytest.approx((r['ret'] - 0.04) / r['vol'], abs=1e-3)
        assert r['mdd'] <= 0 and r['rf'] == 0.04

    def test_cash_scales_and_pairs_rank(self, market):
        _, closes = market
        full = pa.risk({'LEV': 0.5, 'NOISE': 0.5}, closes)
        half = pa.risk({'LEV': 0.25, 'NOISE': 0.25, pa.CASH: 0.5}, closes)
        assert half['vol'] == pytest.approx(full['vol'] / 2, rel=0.02)
        assert half['beta'] == pytest.approx(full['beta'] / 2, rel=0.05)
        r = pa.risk({'LEV': 1, 'LATE': 1, 'NOISE': 1}, closes)
        assert {r['pairs'][0]['a'], r['pairs'][0]['b']} == {'LATE', 'LEV'}
        assert r['pairs'][0]['r'] > 0.95 and r['n_high'] == 1

    def test_crises_need_members_through_the_window(self, market):
        _, closes = market
        r = pa.risk({'LATE': 1.0}, closes)
        assert r['crises']['2008'] is None and r['crises']['2020'] is None
        r = pa.risk({'CRASH': 1.0}, closes)
        assert r['crises']['2020']['dd'] < -0.45 and r['crises']['2020']['cov'] == 1.0

    def test_insufficient_and_asof(self, market):
        _, closes = market
        r = pa.risk({'LEV': 0.5, 'GONE': 0.5}, closes)
        assert r == {'status': 'insufficient', 'cov': 0.5}
        assert pa.risk({'LEV': 1.0}, {'LEV': closes['LEV']}) is None      # no SPY
        r = pa.risk({'LEV': 1.0}, closes, asof='2015-06-30')
        assert r['to'] <= '2015-06-30'


class TestAttribution:
    def _series(self):
        # Two members, the second step weighted 3:1, with a definition change.
        return [
            {'d': '2025-12-31', 'm': 'm0', 'nav': 100.0, 'members': ['A', 'B']},
            {'d': '2026-06-01', 'm': 'm1', 'nav': 105.0, 'members': ['A', 'B'],
             'w': {'A': 0.75, 'B': 0.25}},
            {'d': '2026-09-01', 'm': 'm2', 'nav': 105.0 * (1 + 0.75 * 0.2 - 0.25 * 0.1),
             'members': ['A']},
        ]

    def _closes(self):
        return {'m0': {'A': ('m0', 10.0), 'B': ('m0', 10.0)},
                'm1': {'A': ('m1', 11.0), 'B': ('m1', 10.0)},
                'm2': {'A': ('m2', 13.2), 'B': ('m2', 9.0)}}

    def test_contributions_sum_to_the_compounded_return(self):
        s = self._series()
        a = pa.attribution(s, self._closes(), top=10)
        assert set(a) == {'1M', '3M', 'YTD', 'All'}
        allw = a['All']
        assert allw['ret'] == pytest.approx(s[-1]['nav'] / 100 - 1, abs=1e-4)
        tot = sum(e['c'] for e in allw['top'] + allw['bottom'])
        assert tot == pytest.approx(allw['ret'], abs=1e-3)
        assert [e['t'] for e in allw['top']] == ['A'] and allw['bottom'][0]['t'] == 'B'
        # 3M starts at the 06-01 point: only the weighted step counts.
        three = a['3M']
        assert three['ret'] == pytest.approx(0.125, abs=1e-4)
        assert three['top'][0]['c'] == pytest.approx(0.15, abs=1e-3)

    def test_series_too_short_or_young(self):
        assert pa.attribution(self._series()[:1], {}) == {}
        late = [dict(e, d='2026-08-2' + str(i)) for i, e in enumerate(self._series())]
        assert '3M' not in pa.attribution(late, self._closes())


class TestAnalyze:
    def test_weights_by_strategy_and_held_stays_equal(self, market):
        prices, _ = market
        rows = [{'ticker': t, 'price': 10.0, '_fv_effective': 12.0, '_risk_free_rate': 0.04,
                 'position_weight': w} for t, w in (('LEV', 0.06), ('NOISE', 0.02), ('LATE', None))]
        pfs = [_pf(id='eq', tickers=['LEV', 'NOISE']),
               _pf(id='mod', tickers=['LEV', 'NOISE', 'LATE'], weighting='model'),
               _pf(id='held', tickers=['LEV', 'NOISE'], weighting='held'),
               _pf(id='none', tickers=['ZZZ'])]
        out = pa.analyze(pfs, rows, prices_dir=prices)
        assert out['mod']['risk']['beta'] == pytest.approx(0.75 * 2, abs=0.15)
        assert out['held']['risk']['beta'] == out['eq']['risk']['beta']
        assert out['held']['w'] == 'held'
        assert out['eq']['val']['V'] == pytest.approx(1.2)
        assert out['none']['val'] is None and out['none']['risk'] is None
        assert out['eq']['risk']['rf'] == 0.04


class TestReportAndCli:
    def test_render_ships_analytics(self, tmp_path, market, monkeypatch):
        import json
        import re
        from datetime import date
        from data.snapshot_store import write_snapshot_file
        from scripts.report_html import build_html
        prices, _ = market
        res = tmp_path / 'out'
        res.mkdir()
        rows = [{'ticker': t, 'price': 10.0, '_fv_effective': 15.0, 'rating': 'BUY'}
                for t in ('LEV', 'NOISE')]
        write_snapshot_file(str(res / 'results_2026-09-11.json'), {'results': rows})
        pf = tmp_path / 'pf.json'
        pg.save_portfolios({'version': 1, 'portfolios': [
            {'id': 'b', 'name': 'B', 'tickers': ['LEV', 'NOISE']}]}, str(pf))
        monkeypatch.setenv('PORTFOLIOS_FILE', str(pf))
        build_html([dict(r) for r in rows], str(res / 'r.html'), prices_dir=prices,
                   run_date=date(2026, 9, 11))
        html = (res / 'r.html').read_text(encoding='utf-8')
        payload = json.loads(re.search(r'var PF_PUB=\(function\(\)\{var p=(.*?);return', html).group(1))
        an = payload['analytics']
        assert an['b']['val']['V'] == pytest.approx(1.5) and an['b']['risk']['status'] == 'ok'
        assert an['_universe']['status'] == 'ok'

    def test_show_prints_risk(self, tmp_path, market, capsys, monkeypatch):
        from scripts import portfolios as cli
        prices, _ = market
        f = tmp_path / 'pf.json'
        cli.main(['--file', str(f), 'create', 'x', '--name', 'X', '--tickers', 'LEV,NOISE'])
        rows = [{'ticker': 'LEV', 'price': 10.0, '_fv_effective': 15.0},
                {'ticker': 'NOISE', 'price': 10.0, '_fv_effective': 15.0}]
        monkeypatch.setattr(cli, '_latest_rows', lambda d, day=None: ('2026-09-11', rows))
        capsys.readouterr()
        cli.main(['--file', str(f), 'show', 'x', '--results-dir', str(tmp_path),
                  '--prices-dir', prices])
        out = capsys.readouterr().out
        assert 'look-through: value/price 1.50' in out and 'risk, 1y to 2026-09-11: vol' in out
        assert 'crisis drawdowns: 2008' in out


def test_nav_step_contributions_match_step_return():
    closes = {'a': {'X': ('a', 10.0), 'Y': ('a', 20.0)}, 'b': {'X': ('b', 11.0), 'Y': ('b', 19.0)}}
    c, ret, cov = pn.step_contributions(['X', 'Y', 'Z'], 'a', 'b', closes, {}, {},
                                        weights={'X': 0.5, 'Y': 0.3, 'Z': 0.1, pn.CASH: 0.1})
    assert sum(c.values()) == pytest.approx(ret)
    assert (ret, cov) == pn._step_return(['X'], 'a', 'b', closes, {}, {},
                                         weights={'X': 0.5, 'Y': 0.3, 'Z': 0.1, pn.CASH: 0.1})
    assert cov == pytest.approx(0.9)
