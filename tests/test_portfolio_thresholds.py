"""Buy-below alerts, per-portfolio thresholds and private holdings alerts."""
import json
import os
import re
import shutil
import subprocess

import pytest

from models import portfolio_groups as pg
from models import portfolio_positions as pp
from scripts import portfolios as cli

TEMPLATE = os.path.join(os.path.dirname(__file__), '..', 'templates', 'report.html')


def _row(t, **kw):
    return dict({'ticker': t, 'price': 10.0, 'company_name': t + ' Corp', 'sector': 'Energy',
                 'rating': 'HOLD'}, **kw)


def _by(*rows):
    return pg.rows_by_ticker(list(rows))


def _kinds(entries, kind):
    return [e for e in entries if e['kind'] == kind]


class TestBuyBelow:
    def test_fresh_crossing_is_fyi_confirmed_is_watch(self):
        today = _by(_row('A', mos=0.20, _fv_effective=20.0), _row('B', mos=0.30), _row('C', mos=0.10))
        prev = _by(_row('A', mos=0.10), _row('B', mos=0.18), _row('C', mos=0.20))
        prev2 = _by(_row('A', mos=0.05), _row('B', mos=0.12), _row('C', mos=0.30))
        e = {x['ticker']: x for x in _kinds(pg.classify_changes(today, prev, '2026-10-28',
                                                                 prev2_by_tk=prev2)[0], 'buy_below')}
        assert e['A']['level'] == 'fyi' and e['A']['into'] and 'a Watch follows' in e['A']['message']
        assert '$17.00' in e['A']['message']                     # 20 x (1 - 15%)
        assert e['B']['level'] == 'watch' and e['B']['confirmed']
        assert e['C']['level'] == 'fyi' and not e['C']['into']

    def test_no_prev2_means_no_watch_and_missing_data_is_silent(self):
        today = _by(_row('B', mos=0.30), _row('D', mos=0.30, price=None))
        prev = _by(_row('B', mos=0.18), _row('D', mos=0.0))
        e = _kinds(pg.classify_changes(today, prev, '2026-10-28')[0], 'buy_below')
        assert e == []                                            # B held, but nothing confirms it
        prev2 = _by(_row('B', mos=0.12, price=None))
        assert _kinds(pg.classify_changes(today, prev, '2026-10-28', prev2_by_tk=prev2)[0],
                      'buy_below') == []


class TestThresholds:
    def test_schema(self):
        p = pg.normalize_portfolio({'id': 'a', 'name': 'A', 'thresholds': {
            'score_drop': 5, 'drawdown': 0.3, 'fv_jump': None}})
        assert p['thresholds'] == {'drawdown': 0.3, 'score_drop': 5.0}
        assert pg.normalize({'portfolios': [p]})['version'] == pg.POSITIONS_VERSION
        for bad, msg in (({'drift': 1.5}, 'between'), ({'mos': 0.1}, 'unknown'), ([1], 'object')):
            with pytest.raises(ValueError, match=msg):
                pg.normalize_portfolio({'id': 'a', 'name': 'A', 'thresholds': bad})

    def test_floors_and_per_portfolio_filter(self):
        loose = pg.normalize_portfolio({'id': 'loose', 'name': 'L', 'tickers': ['A', 'B'],
                                        'thresholds': {'score_drop': 5, 'fv_jump': 0.2}})
        plain = pg.normalize_portfolio({'id': 'plain', 'name': 'P', 'tickers': ['A', 'B']})
        assert pg.alert_floors([loose, plain]) == (5, 0.2)
        today = _by(_row('A', _composite_score=50.0, _fv_effective=13.0),
                    _row('B', _composite_score=30.0, _fv_effective=10.0))
        prev = _by(_row('A', _composite_score=57.0, _fv_effective=10.0),
                   _row('B', _composite_score=45.0, _fv_effective=10.0))
        by_id, _ = pg.portfolio_alerts([loose, plain], today, prev, '2026-10-28')
        got = {pid: sorted((a['ticker'], a['kind']) for a in al) for pid, al in by_id.items()}
        assert got['loose'] == [('A', 'fv_jump'), ('A', 'score_drop'), ('B', 'score_drop')]
        assert got['plain'] == [('B', 'score_drop')]


class TestHoldingsAlerts:
    BOOK = {'positions': {'AAA': [{'shares': 10.0, 'cost': 20.0, 'date': '2024-01-01'}],
                          'BBB': [{'shares': 10.0, 'cost': 10.0, 'date': '2024-01-01'}]},
            'cash': 0.0}
    ROWS = {'AAA': {'ticker': 'AAA', 'price': 15.0}, 'BBB': {'ticker': 'BBB', 'price': 10.0}}

    def _p(self, **kw):
        return pg.normalize_portfolio(dict({'id': 'p', 'name': 'P', 'tickers': ['AAA', 'BBB']}, **kw))

    def test_below_cost_and_drift(self):
        p = self._p(targets={'AAA': 0.5, 'BBB': 0.3, 'CCC': 0.1})
        hs = pp.holdings_summary(self.BOOK, self.ROWS)
        al = pp.holdings_alerts(p, hs)
        # Value 150 / 100, so weights 60% / 40% against targets 50% / 30% / 10%.
        assert [(a['kind'], a['ticker']) for a in al] == [
            ('below_cost', 'AAA'), ('drift', 'AAA'), ('drift', 'BBB'), ('drift', 'CCC')]
        assert al[0]['pct'] == pytest.approx(-0.25)
        assert [a['drift'] for a in al[1:]] == [pytest.approx(0.1), pytest.approx(0.1),
                                                pytest.approx(-0.1)]
        # A looser threshold silences them.
        p2 = self._p(targets={'AAA': 0.5, 'BBB': 0.3, 'CCC': 0.1},
                     thresholds={'drawdown': 0.3, 'drift': 0.15})
        assert pp.holdings_alerts(p2, hs) == []
        assert pp.holdings_alerts(p, None) == []

    @pytest.mark.skipif(shutil.which('node') is None, reason='node not installed')
    def test_js_twins_match(self, tmp_path):
        src = open(TEMPLATE, encoding='utf-8').read()

        def fn(name):
            return re.search(r'^function ' + name + r'\(.*?^\}', src, re.S | re.M).group(0)
        p = self._p(targets={'AAA': 0.5, 'BBB': 0.3, 'CCC': 0.1}, thresholds={'score_drop': 5})
        hs = pp.holdings_summary(self.BOOK, self.ROWS)
        entries = [{'kind': 'score_drop', 'drop': 6}, {'kind': 'score_drop', 'drop': 4},
                   {'kind': 'fv_jump', 'move': -0.6}, {'kind': 'fv_jump', 'move': 0.3},
                   {'kind': 'buy_below'}]
        js = ('var PF_PUB={thresholds:{score_drop:10,fv_jump:0.5}};\n'
              + re.search(r'var _PF_HOLD_TH=\{.*?\};', src).group(0) + '\n'
              + fn('_pfPasses') + '\n' + fn('_pfHoldAlerts') + '\n'
              + f'var p={json.dumps(p)},hs={json.dumps(hs)},E={json.dumps(entries)};\n'
              + 'console.log(JSON.stringify({pass:E.map(function(e){return _pfPasses(e,p);}),'
              + 'plain:E.map(function(e){return _pfPasses(e,{});}),al:_pfHoldAlerts(p,hs)}));')
        (tmp_path / 't.js').write_text(js, encoding='utf-8')
        r = subprocess.run(['node', str(tmp_path / 't.js')], capture_output=True, text=True,
                           check=False)
        assert r.returncode == 0, r.stderr
        out = json.loads(r.stdout)
        assert out['pass'] == [pg.passes(e, p) for e in entries]
        assert out['plain'] == [pg.passes(e, {}) for e in entries]
        want = pp.holdings_alerts(p, hs)
        assert [(a['kind'], a['ticker']) for a in out['al']] == [(a['kind'], a['ticker']) for a in want]


class TestCli:
    def test_alerts_keep_holdings_private(self, tmp_path, capsys):
        from data.snapshot_store import write_snapshot_file
        res = tmp_path / 'out'
        res.mkdir()
        rows = [_row('AAA', price=15.0, mos=0.3), _row('BBB', price=10.0, mos=0.3)]
        for d in ('2026-10-26', '2026-10-27', '2026-10-28'):
            write_snapshot_file(str(res / f'results_{d}.json'), {'results': rows})
        f, pos = tmp_path / 'pf.json', tmp_path / 'priv' / 'positions.json'
        pg.save_portfolios({'version': 1, 'portfolios': [
            {'id': 'mine', 'name': 'Mine', 'tickers': ['AAA', 'BBB']}]}, str(f))
        pp.save_books({'mine': {'positions': {
            'AAA': [{'shares': 10.0, 'cost': 20.0, 'date': '2024-01-01'}]}}}, str(pos))
        jout = tmp_path / 'digest.json'
        cli.main(['--file', str(f), '--positions-file', str(pos), 'edit', 'mine',
                  '--threshold', 'drawdown=20%', '--threshold', 'score_drop=5'])
        assert pg.load_portfolios(str(f))['portfolios'][0]['thresholds'] == {
            'drawdown': 0.2, 'score_drop': 5.0}
        capsys.readouterr()
        cli.main(['--file', str(f), '--positions-file', str(pos), 'alerts', '--results-dir',
                  str(res), '--prices-dir', '', '--json', str(jout)])
        out = capsys.readouterr().out
        priv = tmp_path / 'priv' / 'positions_alerts.txt'
        assert 'private holdings alerts: 1' in out and 'below your cost' not in out
        assert 'AAA is 25% below your cost' in priv.read_text(encoding='utf-8')
        assert (priv.stat().st_mode & 0o777) == 0o600
        assert 'below your cost' not in jout.read_text(encoding='utf-8')
        cli.main(['--file', str(f), '--positions-file', str(pos), 'edit', 'mine',
                  '--threshold', 'drawdown=default', '--threshold', 'score_drop=default'])
        assert 'thresholds' not in pg.load_portfolios(str(f))['portfolios'][0]
