"""Positions on portfolios: the private book (lots / cash / realized), the
public strategy fields (targets / weighting), FIFO sales, P&L and the CLI."""
import json

import pytest

from models import portfolio_groups as pg
from models import portfolio_positions as pp
from scripts import portfolios as cli


def _run(f, pos, *args):
    return cli.main(['--file', str(f), '--positions-file', str(pos), *args])


def _lot(shares, cost, day, **kw):
    return dict({'shares': shares, 'cost': cost, 'date': day}, **kw)


class TestNormalize:
    def test_book_sorts_lots_and_drops_empty_fields(self):
        b = pp.normalize_book({'positions': {'deck': [_lot(2, 10, '2025-02-01'),
                                                     _lot(1, 9, '2024-01-01', note=' x ')]},
                               'cash': 0}, 'p')
        assert b == {'positions': {'DECK': [_lot(1.0, 9.0, '2024-01-01', note='x'),
                                            _lot(2.0, 10.0, '2025-02-01')]}}

    @pytest.mark.parametrize('book, msg', [
        ({'positions': {'X': [_lot(0, 1, '2024-01-01')]}}, 'shares'),
        ({'positions': {'X': [_lot(1, -1, '2024-01-01')]}}, 'cost'),
        ({'positions': {'X': [_lot(1, 1, '2024-13-01')]}}, 'YYYY-MM-DD'),
        ({'positions': {'X': [dict(_lot(1, 1, '2024-01-01'), lots=1)]}}, 'unknown'),
        ({'cash': float('nan')}, 'finite'),
        ({'weighting': 'held'}, 'unknown book key'),
    ])
    def test_book_rejects(self, book, msg):
        with pytest.raises(ValueError, match=msg):
            pp.normalize_book(book, 'p')

    def test_strategy_in_definitions(self):
        doc = pg.normalize({'portfolios': [{'id': 'a', 'name': 'A', 'tickers': ['X'],
                                            'targets': {'y': 0.25}, 'weighting': 'target'}]})
        p = doc['portfolios'][0]
        assert doc['version'] == pg.POSITIONS_VERSION
        assert p['tickers'] == ['X', 'Y'] and p['targets'] == {'Y': 0.25}

    def test_plain_file_unchanged(self):
        """A portfolio with no strategy keeps v1 and the same revision as
        before the fields existed (null from a browser export included)."""
        base = {'id': 'a', 'name': 'A', 'created': '2026-01-01', 'tickers': ['X']}
        a = {'version': 1, 'portfolios': [base]}
        b = {'version': 1, 'portfolios': [dict(base, targets=None, weighting=None)]}
        assert pg.normalize(a)['version'] == 1
        assert pg.revision(a) == pg.revision(b)
        assert 'weighting' not in pg.normalize(b)['portfolios'][0]

    @pytest.mark.parametrize('p, msg', [
        ({'targets': {'X': 0.7, 'Y': 0.5}}, 'sum to at most 1'),
        ({'weighting': 'target'}, "needs targets"),
        ({'weighting': 'cap'}, 'weighting must be'),
        ({'targets': {'X': 0.1}, 'exclude': ['X']}, 'cannot be excluded'),
    ])
    def test_strategy_rejects(self, p, msg):
        with pytest.raises(ValueError, match=msg):
            pg.normalize({'portfolios': [dict({'id': 'a', 'name': 'A'}, **p)]})

    def test_positions_never_in_definitions(self):
        """Book fields pasted into the public file are dropped on save."""
        out = pg.normalize({'portfolios': [{'id': 'a', 'name': 'A',
                                            'positions': {'X': [_lot(1, 1, '2024-01-01')]}}]})
        assert 'positions' not in out['portfolios'][0]


class TestTrades:
    def test_sell_fifo_across_lots(self):
        b = {'positions': {'X': [_lot(10, 5, '2024-01-01'), _lot(10, 8, '2025-01-01')]}}
        sales = pp.sell(b, 'x', 15, 12, '2026-01-02')
        assert [(s['shares'], s['cost'], s['bought']) for s in sales] == \
            [(10, 5, '2024-01-01'), (5, 8, '2025-01-01')]
        assert b['positions']['X'] == [_lot(5, 8, '2025-01-01')]
        assert pp.realized_by_year(b['realized']) == {'2026': 10 * 7 + 5 * 4}

    def test_sell_whole_position_and_oversell(self):
        b = {'positions': {'X': [_lot(3, 5, '2024-01-01')]}}
        with pytest.raises(ValueError, match='holds 3'):
            pp.sell(b, 'X', 4, 1, '2026-01-01')
        pp.sell(b, 'X', 3, 6, '2026-01-01')
        assert 'positions' not in b and len(b['realized']) == 1

    def test_buy_keeps_lots_separate(self):
        b = {}
        pp.buy(b, 'x', 1, 10, '2025-01-01')
        pp.buy(b, 'X', 2, 12, '2024-01-01', note='add')
        assert b['positions']['X'] == [_lot(2.0, 12.0, '2024-01-01', note='add'),
                                       _lot(1.0, 10.0, '2025-01-01')]


class TestSummary:
    def test_pnl_weights_cash_and_aliases(self):
        book = {'positions': {'A': [_lot(10, 10, '2024-01-01'), _lot(10, 20, '2025-01-01')],
                              'BRK-A': [_lot(1, 100, '2024-01-01')],
                              'GONE': [_lot(5, 1, '2024-01-01')]},
                'cash': 100.0,
                'realized': [{'ticker': 'Z', 'shares': 1.0, 'cost': 1.0, 'price': 3.0,
                              'date': '2026-03-01'},
                             {'ticker': 'Z', 'shares': 1.0, 'cost': 5.0, 'price': 4.0,
                              'date': '2025-03-01'}]}
        by_tk = {'A': {'ticker': 'A', 'price': 20.0},
                 'BRK-B': {'ticker': 'BRK-B', 'price': 200.0, 'listing_aliases': ['BRK-A']}}
        hs = pp.holdings_summary(book, by_tk, pg.listing_aliases(by_tk), today='2026-10-24')
        rows = {r['ticker']: r for r in hs['rows']}
        assert rows['A']['cost'] == 15 and rows['A']['pnl'] == 100 and rows['A']['lots'] == 2
        assert rows['BRK-A']['row'] == 'BRK-B' and rows['BRK-A']['market_value'] == 200
        assert rows['GONE']['market_value'] is None and rows['GONE']['weight'] is None
        t = hs['totals']
        assert t['market_value'] == 600 and t['value'] == 700 and t['unpriced'] == ['GONE']
        assert t['pnl'] == 600 - 400 and t['pnl_pct'] == pytest.approx(0.5)
        assert rows['A']['weight'] == pytest.approx(400 / 700)
        assert t['realized_ytd'] == 2 and t['realized'] == 1

    def test_live_price_override_and_empty(self):
        book = {'positions': {'A': [_lot(1, 10, '2024-01-01')]}}
        hs = pp.holdings_summary(book, {'A': {'price': 20}}, prices={'A': 30.0})
        assert hs['rows'][0]['market_value'] == 30
        assert pp.holdings_summary({}, {}) is None


class TestFile:
    def test_roundtrip_private_mode(self, tmp_path):
        f = tmp_path / 'positions.json'
        pp.save_books({'b': {}, 'a': {'cash': 5}}, str(f))
        assert pp.load_books(str(f)) == {'a': {'cash': 5.0}}
        assert (f.stat().st_mode & 0o777) == 0o600
        assert pp.load_books(str(tmp_path / 'none.json')) == {}

    def test_env_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv('POSITIONS_FILE', str(tmp_path / 'p.json'))
        assert pp.positions_path() == str(tmp_path / 'p.json')

    def test_gitignored(self):
        import subprocess
        for f in ('portfolio/positions.json', 'portfolio/positions_nav.json'):
            r = subprocess.run(['git', 'check-ignore', f], capture_output=True, text=True,
                               check=False)
            assert r.returncode == 0, f"{f} is not gitignored"


class TestCli:
    def test_migrate_buy_sell_holdings(self, tmp_path, capsys, monkeypatch):
        f, pos = tmp_path / 'pf.json', tmp_path / 'positions.json'
        src = tmp_path / 'holdings.json'
        src.write_text(json.dumps({
            'portfolio_name': 'Mine', 'holdings': [
                {'ticker': 'AAA', 'shares': 10, 'cost_basis': 5, 'purchase_date': '2024-01-01',
                 'notes': 'core'}],
            'realized_gains': [{'ticker': 'ZZ', 'shares_sold': 1, 'cost_basis': 1,
                                'sale_price': 2, 'sale_date': '2025-01-01'}]}), encoding='utf-8')
        _run(f, pos, 'migrate-holdings', '--source', str(src))
        _run(f, pos, 'buy', 'held', 'bbb', '4', '20', '--date', '2025-06-01')
        _run(f, pos, 'sell', 'held', 'AAA', '4', '9', '--date', '2026-01-02')
        _run(f, pos, 'cash', 'held', '50')
        _run(f, pos, 'target', 'held', 'AAA=10%', 'CCC=0.2')
        doc = pg.load_portfolios(str(f))
        p = doc['portfolios'][0]
        assert p['name'] == 'Mine' and p['weighting'] == 'held'
        assert p['tickers'] == ['AAA', 'BBB', 'CCC'] and p['targets'] == {'AAA': 0.1, 'CCC': 0.2}
        # Nothing about dollars reaches the public definitions.
        assert 'cost' not in f.read_text(encoding='utf-8')
        book = pp.load_books(str(pos))['held']
        assert book['cash'] == 50 and book['positions']['AAA'][0]['shares'] == 6
        assert len(book['realized']) == 2

        with pytest.raises(SystemExit, match='sell before removing'):
            _run(f, pos, 'remove', 'held', 'AAA')
        _run(f, pos, 'remove', 'held', 'CCC')
        assert pg.load_portfolios(str(f))['portfolios'][0]['targets'] == {'AAA': 0.1}
        with pytest.raises(SystemExit, match='--with-book'):
            _run(f, pos, 'delete', 'held')

        rows = [{'ticker': 'AAA', 'price': 8.0}, {'ticker': 'BBB', 'price': 25.0}]
        monkeypatch.setattr(cli, '_latest_rows', lambda d, day=None: ('2026-10-24', rows))
        capsys.readouterr()
        _run(f, pos, 'holdings')
        out = capsys.readouterr().out
        assert 'AAA' in out and 'realized YTD 16.00' in out and 'cash 50.00' in out

        _run(f, pos, 'delete', 'held', '--with-book')
        assert pp.load_books(str(pos)) == {}

    def test_weighting_target_needs_targets(self, tmp_path):
        f, pos = tmp_path / 'pf.json', tmp_path / 'positions.json'
        _run(f, pos, 'create', 'a', '--name', 'A')
        with pytest.raises(SystemExit, match='needs targets'):
            _run(f, pos, 'edit', 'a', '--weighting', 'target')
