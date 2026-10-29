"""Paper trading: rebalance plan, safety checks, paper-only client, CLI."""
import argparse
import json
from datetime import datetime

import pytest

from data import alpaca_client as ac
from data.snapshot_store import write_snapshot_file
from models import paper_trade as pt
from scripts import paper_trade as cli

ACC = {'equity': 10_000.0, 'cash': 10_000.0, 'last_equity': 10_000.0}
FRAC = {'tradable': True, 'fractionable': True}
# Thursday 2026-10-29, 10:00 ET.
THU_10 = datetime(2026, 10, 29, 10, 0, tzinfo=cli.ET)


def _assets(*tks, **over):
    a = {t: dict(FRAC) for t in tks}
    a.update(over)
    return a


def _plan(targets, positions=None, account=ACC, prices=None, closes=None, assets=None,
          **limits):
    limits.setdefault('max_position', 1.0)
    tks = set(targets) | set(positions or {})
    prices = prices or {t: 10.0 for t in tks}
    return pt.plan_orders(targets, positions or {}, account, prices, closes or prices,
                          assets or _assets(*tks), pt.Limits(**limits))


# -- plan_orders ------------------------------------------------------------

def test_equal_build_from_cash():
    p = _plan({'A': 0.5, 'B': 0.5}, max_turnover=1.0)
    assert {o['ticker']: o['qty'] for o in p['orders']} == {'A': 500, 'B': 500}
    assert all(o['side'] == 'buy' for o in p['orders'])
    assert p['orders'][0]['limit_price'] == 10.05


def test_band_is_relative_to_the_target():
    # A holds $4,500 of a $5,000 target: a 10% gap, inside the 25% band.
    p = _plan({'A': 0.5, 'B': 0.5}, positions={'A': 450}, max_turnover=1.0,
              account={'equity': 10_000.0, 'cash': 5_500.0})
    assert [o['ticker'] for o in p['orders']] == ['B']


def test_min_order():
    p = _plan({'A': 1.0}, positions={'A': 999.95}, band=0.0, max_turnover=1.0)
    assert p['orders'] == []


def test_exit_sells_everything_whatever_its_size():
    p = _plan({'A': 1.0}, positions={'A': 999, 'X': 0.01}, band=0.5, max_turnover=1.0)
    sells = [o for o in p['orders'] if o['side'] == 'sell']
    assert sells == [dict(sells[0], ticker='X', qty=0.01, reason='left the book')]


def test_position_cap():
    p = _plan({'A': 0.9, 'B': 0.1}, max_turnover=1.0, max_position=0.10)  # A capped
    assert {o['ticker']: o['dollars'] for o in p['orders']} == {'A': 1000.0, 'B': 1000.0}


def test_turnover_cap_takes_exits_first_then_largest_gaps():
    p = _plan({'A': 0.6, 'B': 0.4}, positions={'X': 100},
              account={'equity': 10_000.0, 'cash': 9_000.0}, max_turnover=0.8,
              max_position=1.0)
    # X exit $1,000 + A $6,000 = $7,000; B's $4,000 would pass the $8,000 cap.
    assert [o['ticker'] for o in p['orders']] == ['X', 'A']
    assert {'ticker': 'B', 'reason': 'daily turnover cap'} in p['skipped']


def test_sells_first_and_buys_scaled_to_cash():
    p = _plan({'A': 0.5, 'B': 0.5}, positions={'X': 100},
              account={'equity': 10_000.0, 'cash': 0.0}, max_turnover=2.0)
    assert p['orders'][0]['side'] == 'sell'
    buys = [o for o in p['orders'] if o['side'] == 'buy']
    assert sum(o['dollars'] for o in buys) == pytest.approx(1000.0, abs=0.01)
    assert p['buy_scale'] == 0.1


def test_quote_far_from_close_is_skipped():
    p = _plan({'A': 1.0}, prices={'A': 12.0}, closes={'A': 10.0}, max_turnover=1.0)
    assert p['orders'] == [] and 'from the snapshot close' in p['skipped'][0]['reason']


def test_untradable_and_whole_share_assets():
    p = _plan({'A': 0.5, 'B': 0.5}, prices={'A': 10.0, 'B': 6000.0},
              assets={'A': {'tradable': False}, 'B': {'tradable': True, 'fractionable': False}},
              max_turnover=1.0)
    assert p['orders'] == []
    reasons = {s['ticker']: s['reason'] for s in p['skipped']}
    assert reasons == {'A': 'not tradable at the broker',
                       'B': 'rounds to zero shares (not fractionable)'}


def test_sub_dollar_limit_keeps_four_decimals():
    p = _plan({'A': 1.0}, prices={'A': 0.5123}, max_turnover=1.0, max_position=1.0)
    assert p['orders'][0]['limit_price'] == 0.5149


# -- target_weights ---------------------------------------------------------

def _row(t, rating='BUY', price=10.0, **kw):
    return dict(ticker=t, rating=rating, price=price, company_name=t, sector='Tech', **kw)


PAPER = {'id': 'paper', 'tickers': [], 'exclude': [], 'weighting': None,
         'rule': {'ratings': ['BUY', 'LEAN BUY'], 'sectors': None, 'countries': None, 'cf': []}}


def test_target_weights_equal_over_priced_members():
    rows = [_row('A'), _row('B', 'LEAN BUY'), _row('C', 'HOLD'), _row('D', price=None)]
    assert pt.target_weights(PAPER, rows) == {'A': 0.5, 'B': 0.5}


# -- safety_checks ----------------------------------------------------------

OK = dict(stop_file=False, coverage={'degraded': False}, systemic={'flood': False},
          snapshot_age=1, max_age=3, account=dict(ACC), loss_cap=0.03,
          market_open=True, open_orders=0, in_window=True)


def test_safety_go():
    assert pt.safety_checks(**OK) == []


@pytest.mark.parametrize('over,needle', [
    ({'stop_file': True}, 'kill switch'),
    ({'coverage': {'degraded': True, 'rows': 700, 'prior_rows': 2500}}, 'coverage-degraded'),
    ({'systemic': {'flood': True, 'cause': 'model', 'changed_share': 0.4, 'changed': 1500}},
     'systemic day'),
    ({'snapshot_age': 4}, 'trading days old'),
    ({'snapshot_age': None}, 'trading days old'),
    ({'account': {'equity': 9_600.0, 'last_equity': 10_000.0}}, 'since the prior close'),
    ({'account': None}, 'no account data'),
    ({'account': dict(ACC, trading_blocked=True)}, 'blocked'),
    ({'market_open': False}, 'market is closed'),
    ({'open_orders': 2}, 'still open'),
    ({'in_window': False}, 'trading window'),
])
def test_safety_halts(over, needle):
    reasons = pt.safety_checks(**dict(OK, **over))
    assert len(reasons) == 1 and needle in reasons[0]


# -- client -----------------------------------------------------------------

@pytest.mark.parametrize('url', ['https://api.alpaca.markets',
                                 'https://paper-api.alpaca.markets.evil.example',
                                 'http://localhost:8080'])
def test_client_refuses_live_endpoints(url, monkeypatch):
    with pytest.raises(ac.LiveEndpointError):
        ac.AlpacaPaperClient('k', 's', base_url=url)
    monkeypatch.setenv('ALPACA_BASE_URL', url)
    with pytest.raises(ac.LiveEndpointError):
        ac.AlpacaPaperClient('k', 's')


def test_client_symbols():
    assert ac.to_alpaca_symbol('BRK-A') == 'BRK.A'
    assert ac.from_alpaca_symbol('BRK.A') == 'BRK-A'


# -- CLI --------------------------------------------------------------------

class FakeBroker(cli.OfflineBroker):
    def __init__(self, closes, positions=None, is_open=True):
        super().__init__(closes)
        self._pos, self._open, self.sent = positions or {}, is_open, []

    def positions(self):
        return dict(self._pos)

    def clock(self):
        return {'is_open': self._open}

    def submit_order(self, ticker, qty, side, limit_price, client_order_id=None):
        self.sent.append((ticker, side, qty, limit_price, client_order_id))
        return {'id': f'o{len(self.sent)}'}

    def orders_since(self, after):
        return [{'client_order_id': cid, 'symbol': t, 'side': side, 'status': 'filled',
                 'qty': q, 'filled_qty': q, 'filled_avg_price': lp}
                for t, side, q, lp, cid in self.sent]


@pytest.fixture
def env(tmp_path, monkeypatch):
    out = tmp_path / 'output'
    out.mkdir()
    rows = [_row('A'), _row('B'), _row('C', 'PASS')]
    meta = {'provenance': {'coverage': {'degraded': False}}}
    for d in ('2026-10-27', '2026-10-28'):
        write_snapshot_file(str(out / f'results_{d}.json'), dict(meta, results=rows))
        (out / f'run_summary_{d}.json').write_text('{}', encoding='utf-8')
    pf = tmp_path / 'portfolios.json'
    pf.write_text(json.dumps({'version': 1, 'portfolios': [
        dict(PAPER, name='Paper', color='#2a78d6', description='', created='2026-10-28',
             alerts='buy_line')]}), encoding='utf-8')
    monkeypatch.setattr(cli, 'PAPER_DIR', str(tmp_path / 'paper'))
    args = argparse.Namespace(results_dir=str(out), portfolio='paper',
                              portfolios_file=str(pf), execute=False, snapshot=None, show=5)
    return args, {'A': 10.0, 'B': 10.0, 'C': 10.0}


def test_dry_run_sends_nothing(env, capsys):
    args, closes = env
    b = FakeBroker(closes)
    assert cli.cmd_rebalance(args, broker=b, now=THU_10) == 0
    assert b.sent == [] and not cli.read_ledger()
    assert 'dry run' in capsys.readouterr().out


def test_execute_sends_and_reconcile_records(env):
    args, closes = env
    args.execute = True
    b = FakeBroker(closes, positions={'C': 5})
    assert cli.cmd_rebalance(args, broker=b, now=THU_10) == 0
    assert [(t, s) for t, s, *_ in b.sent] == [('C', 'sell'), ('A', 'buy'), ('B', 'buy')]
    assert cli.read_ledger()[-1]['sent']
    assert cli.cmd_reconcile(args, broker=b, now=THU_10) == 0
    rec = cli.read_ledger()[-1]
    assert rec['type'] == 'reconcile' and rec['filled'] == 3
    assert cli.load_nav()[-1]['d'] == '2026-10-29'


def test_kill_switch_stops_execution(env):
    args, closes = env
    cli.cmd_stop(args)
    args.execute = True
    b = FakeBroker(closes)
    assert cli.cmd_rebalance(args, broker=b, now=THU_10) == 0
    assert b.sent == []
    assert 'kill switch' in cli.read_ledger()[-1]['halted'][0]
    cli.cmd_resume(args)
    assert cli.cmd_rebalance(args, broker=b, now=THU_10) == 0 and b.sent


def test_tick_rebalances_once_then_reconciles(env):
    args, closes = env
    b = FakeBroker(closes)
    cli.cmd_tick(args, broker=b, now=THU_10)
    n = len(b.sent)
    assert n and cli.read_ledger()[-1]['type'] == 'rebalance'
    cli.cmd_tick(args, broker=b, now=THU_10.replace(hour=16, minute=10))
    assert len(b.sent) == n and cli.read_ledger()[-1]['type'] == 'reconcile'


def test_finished_snapshot_skips_one_still_running(tmp_path):
    for d in ('2026-10-27', '2026-10-28'):
        write_snapshot_file(str(tmp_path / f'results_{d}.json'), {'results': []})
    (tmp_path / 'run_summary_2026-10-27.json').write_text('{}', encoding='utf-8')
    lock = tmp_path / '.run_daily.lock'
    lock.write_text(str(__import__('os').getpid()), encoding='utf-8')
    assert cli.finished_snapshot(str(tmp_path), lock)[0] == '2026-10-27'
    lock.unlink()
    assert cli.finished_snapshot(str(tmp_path), lock)[0] == '2026-10-28'


def test_trading_days_between_skips_weekends_and_holidays():
    from datetime import date
    # Fri 2026-11-20 -> Mon 11-30: Thanksgiving (11-26) is closed.
    assert cli.trading_days_between(date(2026, 11, 20), date(2026, 11, 30)) == 5


# -- review fixes -----------------------------------------------------------

def test_format_qty_is_plain_and_never_rounds_up():
    assert ac.format_qty(1310270.5) == '1310270.5'
    assert ac.format_qty(999.9995) == '999.9995'
    assert ac.format_qty(3.0) == '3'
    assert 'e' not in ac.format_qty(0.000123456)


def test_data_gap_is_held_not_sold():
    rows = [_row('A'), _row('B', price=None), _row('C', 'PASS')]
    hold = pt.hold_set({'B': 5, 'C': 5, 'Z': 5}, rows)
    assert hold == {'B', 'Z'}          # C has a good row and left the book
    p = _plan({'A': 1.0}, positions={'B': 5, 'C': 5, 'Z': 5}, max_turnover=2.0)
    assert {o['ticker'] for o in p['orders'] if o['side'] == 'sell'} == {'B', 'C', 'Z'}
    tks = ('A', 'B', 'C', 'Z')
    p = pt.plan_orders({'A': 1.0}, {'B': 5, 'C': 5, 'Z': 5}, ACC, {t: 10.0 for t in tks},
                       {t: 10.0 for t in tks}, _assets(*tks),
                       pt.Limits(max_turnover=2.0, max_position=1.0), hold=hold)
    assert {o['ticker'] for o in p['orders'] if o['side'] == 'sell'} == {'C'}


def test_folded_listing_is_sold():
    rows = [_row('GOOGL', listing_aliases=['GOOG'])]
    assert pt.hold_set({'GOOG': 1}, rows) == set()


def test_tick_retries_after_every_order_failed(env):
    args, closes = env

    class Down(FakeBroker):
        def submit_order(self, *a, **k):
            raise ac.AlpacaError('HTTP 503')

    cli.cmd_tick(args, broker=Down(closes), now=THU_10)
    assert cli.read_ledger()[-1]['failed'] and not cli.read_ledger()[-1]['sent']
    b = FakeBroker(closes)
    cli.cmd_tick(args, broker=b, now=THU_10.replace(minute=20))
    assert b.sent and cli.read_ledger()[-1]['type'] == 'rebalance'


def test_orders_since_pages():
    pages = [[{'id': str(i), 'submitted_at': f't{i:04}'} for i in range(500)],
             [{'id': str(i), 'submitted_at': f't{i:04}'} for i in range(499, 620)]]
    c = ac.AlpacaPaperClient('k', 's', throttle=lambda: None)
    c._api = lambda *a, **k: pages.pop(0)
    assert len(c.orders_since('2026-10-29')) == 620


def test_network_error_becomes_alpaca_error():
    import requests

    class S(requests.Session):
        def request(self, *a, **k):
            raise requests.ConnectionError('reset')

    c = ac.AlpacaPaperClient('k', 's', session=S(), throttle=lambda: None)
    with pytest.raises(ac.AlpacaError):
        c.account()


def test_untradable_weight_is_spread_over_the_rest():
    w, dropped = pt.tradable_only({'A': 0.25, 'B': 0.25, 'OTC': 0.5},
                                  {'A': FRAC, 'B': FRAC, 'OTC': {'tradable': False}})
    assert w == {'A': 0.5, 'B': 0.5} and dropped == 0.5
