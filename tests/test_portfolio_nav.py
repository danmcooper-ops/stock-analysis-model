"""Portfolio NAV ledger: as-of prices, point-in-time steps, rebuild/splice."""
import json
import os
from datetime import date

import pandas as pd
import pytest

from data import portfolio_nav as pn
from data.price_store import asof_closes
from models import portfolio_groups as pg

D1, D2, D3, D4 = '2026-09-08', '2026-09-09', '2026-09-10', '2026-09-11'


def _px(dirpath, ticker, closes):
    """Write {date: close} as a price parquet shaped like download_prices'."""
    idx = pd.DatetimeIndex(pd.to_datetime(list(closes)), name='Date')
    pd.DataFrame({'Close': list(closes.values())}, index=idx).to_parquet(
        os.path.join(dirpath, f'{ticker}.parquet'))


@pytest.fixture
def prices(tmp_path):
    d = tmp_path / 'prices'
    d.mkdir()
    _px(d, 'SPY', {D1: 100.0, D2: 101.0, D3: 102.01, D4: 103.0})
    _px(d, 'AAA', {D1: 10.0, D2: 11.0, D3: 12.1, D4: 12.1})
    _px(d, 'BBB', {D1: 20.0, D2: 19.0, D3: 19.0, D4: 20.9})
    _px(d, 'CCC', {D1: 50.0, D2: 50.0, D3: 55.0, D4: 55.0})
    _px(d, 'JUNK', {D1: 0.001, D2: 15.0, D3: 15.0, D4: 15.0})
    return str(d)


def _pf(**kw):
    base = {'id': 'p', 'name': 'P', 'tickers': [], 'exclude': [], 'rule': None}
    base.update(kw)
    return pg.normalize_portfolio(base, today='2026-09-01')


def _rows(*tickers, **fields):
    return [dict({'ticker': t}, **fields.get(t, {})) for t in tickers]


class TestAsofCloses:
    def test_asof_never_peeks_forward_and_skips_bad_bars(self, tmp_path):
        d = tmp_path / 'px'
        d.mkdir()
        _px(d, 'X', {D1: 5.0, D2: float('nan'), D3: -1.0})
        _px(d, 'OLD', {'2026-08-01': 9.0})
        got = asof_closes(str(d), [D1, D3, '2026-09-07'])
        assert got[D1]['X'] == (D1, 5.0)
        assert got[D3]['X'] == (D1, 5.0)          # NaN and negative bars skipped
        assert 'X' not in got['2026-09-07']        # before the first bar
        assert 'OLD' not in got[D3]                # stale beyond the 7-day gap

    def test_missing_dir(self):
        assert asof_closes('/nonexistent', [D1]) is None


class TestUpdate:
    def test_steps_rebalance_and_idempotence(self, prices):
        led = pn.empty_ledger()
        cf = pn.prices_closes_fn(prices)
        p = _pf(tickers=['AAA', 'BBB'])
        pn.update(led, [p], D1, _rows('AAA', 'BBB', 'CCC'), cf)
        s = led['portfolios']['p']
        assert s[0]['nav'] == 100.0 and s[0]['members'] == ['AAA', 'BBB'] and s[0]['m'] == D1
        # Day 2: AAA +10%, BBB -5% -> +2.5%. Then the definition changes to CCC.
        pn.update(led, [p], D2, _rows('AAA', 'BBB', 'CCC'), cf, prev_rows=_rows('AAA', 'BBB', 'CCC'))
        assert s[-1]['nav'] == pytest.approx(102.5)
        p2 = _pf(tickers=['CCC'])
        pn.update(led, [p2], D3, _rows('AAA', 'BBB', 'CCC'), cf)
        # Day 3 is earned by what was held over the step (AAA, BBB: +10%, 0%),
        # not by the new definition; the new members apply from here on.
        assert s[-1]['nav'] == pytest.approx(102.5 * 1.05)
        assert s[-1]['members'] == ['CCC'] and s[-1]['h'] != s[-2]['h']
        before = json.dumps(led, sort_keys=True)
        pn.update(led, [p2], D3, _rows('AAA', 'BBB', 'CCC'), cf)   # re-run of the same day
        assert json.dumps(led, sort_keys=True) == before
        assert pn.update(led, [p2], D2, _rows('CCC'), cf) == D2      # older day: series untouched
        assert [e['d'] for e in s] == [D1, D2, D3]

    def test_benchmarks(self, prices):
        led = pn.empty_ledger()
        cf = pn.prices_closes_fn(prices)
        rows = _rows('AAA', 'BBB', 'JUNK')
        pn.update(led, [_pf()], D1, rows, cf)
        pn.update(led, [_pf()], D2, rows, cf, prev_rows=rows)
        assert led['bench']['spy'][-1]['nav'] == pytest.approx(101.0)
        uni = led['bench']['universe'][-1]
        # JUNK's $0.001 -> $15 print is rejected; AAA +10%, BBB -5%.
        assert uni['ret'] == pytest.approx(0.025) and uni['cov'] == pytest.approx(2 / 3, abs=1e-4)

    def test_empty_holding_is_cash_and_fallback_prices(self, prices):
        led = pn.empty_ledger()
        cf = pn.prices_closes_fn(prices)
        p = _pf(tickers=['NOPX', 'AAA'])
        day1 = _rows('NOPX', 'AAA', NOPX={'price': 4.0})
        pn.update(led, [p, _pf(id='empty')], D1, day1, cf)
        day2 = _rows('NOPX', 'AAA', NOPX={'price': 5.0})
        pn.update(led, [p, _pf(id='empty')], D2, day2, cf, prev_rows=day1)
        # NOPX has no parquet: its snapshot prices (+25%) stand in; AAA +10%.
        assert led['portfolios']['p'][-1]['ret'] == pytest.approx(0.175)
        assert led['portfolios']['empty'][-1]['ret'] == 0.0
        # An implausible snapshot move (a split) is left unpriced.
        day3 = _rows('NOPX', 'AAA', NOPX={'price': 1.0})
        pn.update(led, [p], D3, day3, cf, prev_rows=day2)
        assert led['portfolios']['p'][-1]['cov'] == 0.5

    def test_split_readjusted_history_is_not_a_crash(self, prices):
        led = pn.empty_ledger()
        cf = pn.prices_closes_fn(prices)
        p = _pf(tickers=['AAA'])
        pn.update(led, [p], D1, _rows('AAA'), cf)
        pn.update(led, [p], D2, _rows('AAA'), cf)
        # A 2:1 split re-downloads AAA with every past close halved; the next
        # step reads both ends from the new file, so it is +10%, not -45%.
        _px(prices, 'AAA', {D1: 5.0, D2: 5.5, D3: 6.05})
        pn.update(led, [p], D3, _rows('AAA'), cf)
        assert led['portfolios']['p'][-1]['ret'] == pytest.approx(0.10)

    def test_lagging_data_spans_the_missing_bar(self, tmp_path):
        d = tmp_path / 'px'
        d.mkdir()
        _px(d, 'SPY', {D1: 100.0, D2: 100.0})
        _px(d, 'AAA', {D1: 10.0, D2: 11.0})
        led = pn.empty_ledger()
        p = _pf(tickers=['AAA'])
        pn.update(led, [p], D1, _rows('AAA'), pn.prices_closes_fn(str(d)))
        pn.update(led, [p], D2, _rows('AAA'), pn.prices_closes_fn(str(d)))
        pn.update(led, [p], D3, _rows('AAA'), pn.prices_closes_fn(str(d)))  # D3 bar not in yet
        assert led['portfolios']['p'][-1]['m'] == D2 and led['portfolios']['p'][-1]['ret'] == 0.0
        _px(d, 'SPY', {D1: 100.0, D2: 100.0, D3: 100.0, D4: 100.0})
        _px(d, 'AAA', {D1: 10.0, D2: 11.0, D3: 12.0, D4: 13.2})
        pn.update(led, [p], D4, _rows('AAA'), pn.prices_closes_fn(str(d)))
        # D2 -> D4 in one step: the D3 move is not lost.
        assert led['portfolios']['p'][-1]['nav'] == pytest.approx(132.0)

    def test_no_market_bar_writes_nothing(self, tmp_path):
        led = pn.empty_ledger()
        assert pn.update(led, [_pf()], D1, [], pn.prices_closes_fn(str(tmp_path))) is None
        assert led == pn.empty_ledger()


class TestRebuildAndSplice:
    def test_rebuild_matches_live_updates(self, prices):
        rows = _rows('AAA', 'BBB', 'CCC', AAA={'rating': 'BUY'}, BBB={'rating': 'BUY'}, CCC={'rating': 'HOLD'})
        snaps = [(D1, rows), (D2, rows), (D3, rows)]
        p = _pf(rule={'ratings': ['BUY']})
        rebuilt = pn.rebuild([p], snaps, prices)
        live = pn.empty_ledger()
        prev = None
        for d, r in snaps:
            pn.update(live, [p], d, r, pn.prices_closes_fn(prices), prev_rows=prev)
            prev = r
        assert [e['nav'] for e in rebuilt['portfolios']['p']] == \
            pytest.approx([e['nav'] for e in live['portfolios']['p']])
        assert all(e['bf'] for e in rebuilt['portfolios']['p'])

    def test_splice_rescales_live_history(self):
        bf = [{'d': D1, 'nav': 100.0, 'bf': True}, {'d': D2, 'nav': 110.0, 'bf': True, 'ret': 0.1, 'cov': 1.0},
              {'d': D3, 'nav': 121.0, 'bf': True}]
        live = [{'d': D2, 'nav': 100.0, 'ret': None, 'cov': None}, {'d': D3, 'nav': 105.0}]
        out = pn.splice(live, bf)
        assert [(e['d'], e['nav'], bool(e.get('bf'))) for e in out] == \
            [(D1, 100.0, True), (D2, 110.0, False), (D3, 115.5, False)]
        assert out[1]['ret'] == 0.1
        assert pn.splice([], bf) == bf and pn.splice(live, []) == live

    def test_payload_flags_and_window_return(self):
        led = pn.empty_ledger()
        led['portfolios']['p'] = [
            {'d': D1, 'nav': 100.0, 'bf': True, 'h': 'a', 'cov': None, 'n': 2},
            {'d': D2, 'nav': 90.0, 'h': 'b', 'cov': 0.5, 'n': 0}]
        out = pn.payload(led, ['p', 'missing'])
        assert out['pf'] == {'p': [[D1, 100.0, 1], [D2, 90.0, 2 | 4 | 8]]}
        assert pn.window_return(led['portfolios']['p'], None) == pytest.approx(-0.1)
        assert pn.window_return(led['portfolios']['p'], 30) is None

    def test_ledger_io(self, tmp_path):
        path = str(tmp_path / 'n.json')
        assert pn.load_ledger(path) == pn.empty_ledger()
        led = pn.empty_ledger()
        led['bench']['spy'].append({'d': D1, 'nav': 100.0})
        pn.save_ledger(path, led)
        assert pn.load_ledger(path) == led
        with open(path, 'w', encoding='utf-8') as f:
            f.write('{"version": 99}')
        assert pn.load_ledger(path) == pn.empty_ledger()


class TestRenderAndCli:
    def _setup(self, tmp_path, prices):
        from data.snapshot_store import write_snapshot_file
        res = tmp_path / 'out'
        res.mkdir()
        rows = _rows('AAA', 'BBB', AAA={'rating': 'BUY', 'price': 10.0}, BBB={'rating': 'HOLD', 'price': 20.0})
        rows2 = _rows('AAA', 'BBB', AAA={'rating': 'BUY', 'price': 11.0}, BBB={'rating': 'BUY', 'price': 19.0})
        write_snapshot_file(str(res / f'results_{D1}.json'), {'results': rows})
        write_snapshot_file(str(res / f'results_{D2}.json'), {'results': rows2})
        pf = tmp_path / 'pf.json'
        pg.save_portfolios({'version': 1, 'portfolios': [
            {'id': 'buys', 'name': 'BUYs', 'rule': {'ratings': ['BUY']}}]}, str(pf))
        return res, pf, rows, rows2

    def test_render_advances_ledger_only_for_its_own_snapshot(self, tmp_path, prices, monkeypatch):
        from scripts.report_html import build_html
        res, pf, rows, rows2 = self._setup(tmp_path, prices)
        monkeypatch.setenv('PORTFOLIOS_FILE', str(pf))
        # Rows rendered without an explicit date, or for a date with no
        # snapshot beside the HTML, never touch the ledger.
        build_html([dict(r) for r in rows2], str(res / 'r.html'), prices_dir=prices)
        build_html([dict(r) for r in rows2], str(res / 'r.html'), prices_dir=prices, run_date=date(2026, 9, 10))
        assert not (res / pn.LEDGER_NAME).exists()
        build_html([dict(r) for r in rows], str(res / 'r.html'), prices_dir=prices, run_date=date(2026, 9, 8))
        build_html([dict(r) for r in rows2], str(res / 'r.html'), prices_dir=prices, run_date=date(2026, 9, 9))
        led = pn.load_ledger(str(res / pn.LEDGER_NAME))
        assert [(e['d'], e['members']) for e in led['portfolios']['buys']] == [(D1, ['AAA']), (D2, ['AAA', 'BBB'])]
        assert led['portfolios']['buys'][-1]['nav'] == pytest.approx(110.0)
        import re
        html = (res / 'r.html').read_text(encoding='utf-8')
        payload = json.loads(re.search(r'var PF_PUB=\(function\(\)\{var p=(.*?);return', html).group(1))
        assert payload['nav']['pf'] == {'buys': [[D1, 100.0, 0], [D2, 110.0, 0]]}
        assert payload['nav']['spy'] == [[D1, 100.0], [D2, 101.0]]

    def test_cli_nav_rebuild(self, tmp_path, prices, capsys):
        from scripts import portfolios as cli
        res, pf, _, _ = self._setup(tmp_path, prices)
        cli.main(['--file', str(pf), 'nav', '--rebuild', '--results-dir', str(res), '--prices-dir', prices])
        out = capsys.readouterr().out
        assert 'replaying 1 portfolio(s) over 2 snapshot(s)' in out
        assert 'BUYs' in out and '+10%' in out and '1 backfilled day(s)' not in out
        led = pn.load_ledger(str(res / pn.LEDGER_NAME))
        assert all(e.get('bf') for e in led['portfolios']['buys'])


class TestWeighting:
    """model / target weights in the public ledger; held weights only in
    the private one (the book is never published)."""

    def test_model_weights_recorded_and_applied(self, prices):
        led = pn.empty_ledger()
        cf = pn.prices_closes_fn(prices)
        p = _pf(tickers=['AAA', 'BBB', 'CCC'], weighting='model')
        rows = _rows('AAA', 'BBB', 'CCC', AAA={'position_weight': 0.06},
                     BBB={'position_weight': 0.02})            # CCC: not sized, holds nothing
        pn.update(led, [p], D1, rows, cf)
        assert led['portfolios']['p'][0]['w'] == {'AAA': 0.75, 'BBB': 0.25}
        pn.update(led, [p], D2, rows, cf, prev_rows=rows)
        # 0.75 * +10% + 0.25 * -5%
        assert led['portfolios']['p'][-1]['ret'] == pytest.approx(0.0625)

    def test_model_without_sizes_is_equal(self, prices):
        led = pn.empty_ledger()
        p = _pf(tickers=['AAA', 'BBB'], weighting='model')
        pn.update(led, [p], D1, _rows('AAA', 'BBB'), pn.prices_closes_fn(prices))
        assert 'w' not in led['portfolios']['p'][0]

    def test_target_weights_hold_cash(self, prices):
        led = pn.empty_ledger()
        cf = pn.prices_closes_fn(prices)
        p = _pf(targets={'AAA': 0.5}, weighting='target')
        rows = _rows('AAA', 'BBB')
        pn.update(led, [p], D1, rows, cf)
        assert led['portfolios']['p'][0]['w'] == {pn.CASH: 0.5, 'AAA': 0.5}
        pn.update(led, [p], D2, rows, cf, prev_rows=rows)
        e = led['portfolios']['p'][-1]
        assert e['ret'] == pytest.approx(0.05) and e['cov'] == 1.0

    def test_unpriced_member_weight_spreads_over_priced(self):
        closes = {D1: {'A': (D1, 10.0)}, D2: {'A': (D2, 11.0)}}
        ret, cov = pn._step_return([], D1, D2, closes, {}, {},
                                   weights={'A': 0.3, 'GONE': 0.3, pn.CASH: 0.4})
        assert ret == pytest.approx(0.10 * 0.6) and cov == pytest.approx(0.7)

    def test_plain_hash_unchanged(self):
        """Adding strategy keys to def_hash must not flag every existing
        portfolio as 'definition changed' on upgrade."""
        import hashlib
        p = _pf(tickers=['AAA'])
        old = hashlib.sha256(json.dumps({'tickers': ['AAA'], 'exclude': [], 'rule': None},
                                        sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:10]
        assert pn.def_hash(p) == old
        assert pn.def_hash(_pf(tickers=['AAA'], weighting='model')) != old

    def test_held_goes_to_private_ledger_only(self, prices):
        led, priv = pn.empty_ledger(), pn.empty_ledger()
        cf = pn.prices_closes_fn(prices)
        p = _pf(tickers=['AAA', 'BBB'], weighting='held')
        book = {'positions': {'AAA': [{'shares': 30.0, 'cost': 1.0, 'date': '2026-01-01'}],
                              'BBB': [{'shares': 5.0, 'cost': 1.0, 'date': '2026-01-01'}]},
                'cash': 100.0}
        rows = _rows('AAA', 'BBB', AAA={'price': 10.0}, BBB={'price': 20.0})
        for d, prev in ((D1, None), (D2, rows)):
            pn.update(led, [p], d, rows, cf, prev_rows=prev, books={'p': book}, private=priv)
        pub = led['portfolios']['p']
        assert all('w' not in e for e in pub)
        assert pub[-1]['ret'] == pytest.approx(0.025)             # equal-weighted
        w = priv['portfolios']['p'][0]['w']
        assert w == {pn.CASH: 0.2, 'AAA': 0.6, 'BBB': 0.2}       # 300 / 100 / 100 of 500
        assert priv['portfolios']['p'][-1]['ret'] == pytest.approx(0.6 * 0.10 + 0.2 * -0.05)
        assert 'p' in pn.payload(led, ['p'])['pf'] and '"w"' not in json.dumps(pn.payload(led, ['p']))

    def test_held_backfill_replays_lots_as_of_each_day(self, prices):
        from models import portfolio_positions as pp
        book = {'positions': {'AAA': [{'shares': 1.0, 'cost': 1.0, 'date': D2}]},
                'realized': [{'ticker': 'BBB', 'shares': 2.0, 'cost': 1.0, 'price': 2.0,
                              'date': D3, 'bought': D1},
                             {'ticker': 'OLD', 'shares': 2.0, 'cost': 1.0, 'price': 2.0,
                              'date': D3}],
                'cash': 50.0}
        assert pp.shares_as_of(book, D1) == {'BBB': 2.0}
        assert pp.shares_as_of(book, D2) == {'AAA': 1.0, 'BBB': 2.0}
        assert pp.shares_as_of(book, D3) == {'AAA': 1.0}
        assert pp.shares_as_of(book) == {'AAA': 1.0}
        p = _pf(tickers=['AAA', 'BBB'], weighting='held')
        rows = _rows('AAA', 'BBB', AAA={'price': 10.0}, BBB={'price': 20.0})
        priv = pn.empty_ledger()
        pn.rebuild([p], [(D1, rows), (D2, rows)], prices, books={'p': book}, private=priv)
        s = priv['portfolios']['p']
        assert s[0]['w'] == {'BBB': 1.0}                          # no cash in a backfill
        assert s[1]['ret'] == pytest.approx(-0.05) and all(e['bf'] for e in s)

    def test_returns_are_dividend_adjusted(self):
        """The NAV is a total return because the parquets hold adjusted
        closes; pin the download setting that makes it so."""
        src = open(os.path.join(os.path.dirname(__file__), '..', 'scripts', 'download_prices.py'),
                   encoding='utf-8').read()
        assert 'auto_adjust=True' in src

    def test_cli_rebuild_writes_private_ledger(self, tmp_path, prices, capsys):
        from data.snapshot_store import write_snapshot_file
        from models import portfolio_positions as pp
        from scripts import portfolios as cli
        res = tmp_path / 'out'
        res.mkdir()
        rows = _rows('AAA', 'BBB', AAA={'price': 10.0}, BBB={'price': 20.0})
        write_snapshot_file(str(res / f'results_{D1}.json'), {'results': rows})
        write_snapshot_file(str(res / f'results_{D2}.json'), {'results': rows})
        pf, pos = tmp_path / 'pf.json', tmp_path / 'priv' / 'positions.json'
        pg.save_portfolios({'version': 1, 'portfolios': [
            {'id': 'mine', 'name': 'Mine', 'tickers': ['AAA'], 'weighting': 'held'}]}, str(pf))
        pp.save_books({'mine': {'positions': {'AAA': [{'shares': 1.0, 'cost': 1.0, 'date': D1}]}}},
                      str(pos))
        cli.main(['--file', str(pf), '--positions-file', str(pos), 'nav', '--rebuild',
                  '--results-dir', str(res), '--prices-dir', prices])
        out = capsys.readouterr().out
        assert 'equal-weighted (public)' in out and 'held weights (private)' in out
        ppath = tmp_path / 'priv' / pn.PRIVATE_LEDGER_NAME
        assert (ppath.stat().st_mode & 0o777) == 0o600
        assert '"w"' not in json.dumps(pn.load_ledger(str(res / pn.LEDGER_NAME)))
        assert pn.load_ledger(str(ppath))['portfolios']['mine'][0]['w'] == {'AAA': 1.0}
