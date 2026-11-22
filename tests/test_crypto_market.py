"""Investing › Crypto's market table: the universe rule, the risk measures,
the report, the CoinGecko client's rate-limit handling and the build script,
all offline."""
import json
import math
from datetime import date, datetime, timedelta, timezone

import pytest

from data import crypto_client as cc
from models import crypto_market as cm
from scripts import crypto_market as script

AS_OF = date(2026, 11, 20)


def _series(n, start=100.0, step=0.01, end=AS_OF):
    """n daily closes ending on *end*, growing by *step* a day."""
    return [(end - timedelta(days=n - 1 - i), start * (1 + step) ** i) for i in range(n)]


def _market(cid, sym, rank, price=10.0, mcap=1e9, **kw):
    m = {'id': cid, 'symbol': sym, 'name': cid.title(), 'market_cap_rank': rank,
         'current_price': price, 'market_cap': mcap, 'total_volume': mcap / 20,
         'circulating_supply': 900.0, 'max_supply': 1000.0,
         'price_change_percentage_24h_in_currency': 1.5,
         'price_change_percentage_7d_in_currency': -6.0,
         'price_change_percentage_30d_in_currency': 12.0,
         'price_change_percentage_1y_in_currency': -28.0,
         'ath_change_percentage': -51.7, 'ath_date': '2025-10-06T10:57:42.000Z'}
    m.update(kw)
    return m


# ---------------------------------------------------------------- universe
@pytest.mark.parametrize('cats,why', [
    (['Stablecoins', 'USD Stablecoin'], 'stablecoin'),
    (['Tokenized Gold', 'Commodity-backed Stablecoin'], 'tokenized asset'),
    (['Tokenized Assets', 'Tokenized Private Credit'], 'tokenized asset'),
    (['Wrapped-Tokens', 'Tokenized BTC'], 'wrapped copy'),
    (['Liquid Staking Tokens'], 'liquid-staking token'),
    (['Bridged Tokens'], 'bridged copy'),
])
def test_claims_on_another_asset_are_left_out(cats, why):
    assert why in cm.excluded_reason('x', cats)


def test_real_coins_are_kept():
    for cats in ([], None, ['Layer 1 (L1)', 'Proof of Work (PoW)'],
                 ['Smart Contract Platform', 'Real World Assets (RWA)', 'Oracle']):
        assert cm.excluded_reason('x', cats) is None


def test_overrides_win(monkeypatch):
    monkeypatch.setattr(cm, 'ALWAYS_INCLUDE', frozenset({'keep'}))
    monkeypatch.setattr(cm, 'ALWAYS_EXCLUDE', frozenset({'drop'}))
    assert cm.excluded_reason('keep', ['Stablecoins']) is None
    assert cm.excluded_reason('drop', []) == 'excluded by hand'


# ---------------------------------------------------------------- measures
def test_volatility_is_annualised_over_365_days():
    # Alternating +1% / -1% log moves: sd ~0.01 a day.
    h, p = [], 100.0
    for i in range(400):
        h.append((AS_OF - timedelta(days=399 - i), p))
        p *= math.exp(0.01 if i % 2 else -0.01)
    v = cm.volatility(h, AS_OF, 365)
    assert v == pytest.approx(0.01 * math.sqrt(365), rel=0.01)
    assert cm.volatility(h[:5], AS_OF, 365) is None


def test_max_drawdown():
    assert cm.max_drawdown([100, 120, 60, 90, 130]) == pytest.approx(-0.5)
    assert cm.max_drawdown([1, 2, 3]) == 0.0
    assert cm.max_drawdown([]) is None


def test_btc_correlation_is_daily_and_needs_enough_days():
    btc = _series(365, step=0.0)
    btc = [(d, 100 * (1.02 if i % 3 == 0 else 0.99) ** (i % 7)) for i, (d, _) in enumerate(btc)]
    assert cm.corr_daily(btc, btc, AS_OF) == pytest.approx(1.0)
    inv = [(d, 1e4 / p) for d, p in btc]
    assert cm.corr_daily(inv, btc, AS_OF) == pytest.approx(-1.0)
    assert cm.corr_daily(btc[-50:], btc, AS_OF) is None


def test_spy_correlation_pairs_each_week_with_the_next_utc_close():
    # SPY on weekdays; the coin's close stamped the day after each SPY day
    # moves with it, the one on the SPY day itself does not.
    spy, coin, p = [], {}, 100.0
    d = AS_OF - timedelta(days=364)
    i = 0
    while d <= AS_OF:
        if d.weekday() < 5:
            p *= 1.03 if (i // 5) % 2 else 0.98
            p *= 1 + (i % 5) * 0.001
            spy.append((d, p))
            coin[d + timedelta(days=1)] = p * 2
            i += 1
        coin.setdefault(d, 1.0 + (i % 4))
        d += timedelta(days=1)
    r = cm.corr_weekly_spy(sorted(coin.items()), spy, AS_OF)
    assert r == pytest.approx(1.0)


def test_ytd_reads_the_close_stamped_1_january():
    h = [(date(2026, 1, 1), 50.0), (date(2026, 6, 1), 70.0)]
    assert cm.ytd_return(h, 60.0, AS_OF) == pytest.approx(0.2)
    assert cm.ytd_return(h[1:], 60.0, AS_OF) is None


# ---------------------------------------------------------------- report
def test_report_takes_the_top_n_real_coins_in_rank_order():
    markets = [_market('bitcoin', 'btc', 1, price=60863, mcap=1.2e12),
               _market('tether', 'usdt', 2, price=1.0, mcap=1.7e11),
               _market('ethereum', 'eth', 3, mcap=2.1e11),
               _market('figure-heloc', 'figr_heloc', 4),
               _market('solana', 'sol', 5)]
    cats = {'tether': ['Stablecoins'], 'figure-heloc': ['Tokenized Assets'],
            'bitcoin': ['Layer 1 (L1)']}
    wiggly = [(d, p * (1.03 if i % 3 else 0.98)) for i, (d, p) in enumerate(_series(400))]
    hist = {'bitcoin': wiggly, 'ethereum': _series(400, step=0.02)}
    rep = cm.build_report(markets, hist, cats, AS_OF, '2026-11-20 17:00 EST',
                          spy=None, top_n=2)
    assert [r['symbol'] for r in rep['rows']] == ['BTC', 'ETH']
    assert [e['symbol'] for e in rep['excluded']] == ['USDT']   # stops once two are kept
    assert rep['version'] == cm.REPORT_VERSION == 1
    assert rep['run_date'] == '2026-11-20' and rep['total'] == 2
    btc, eth = rep['rows']
    assert btc['chg_7d'] == pytest.approx(-0.06) and btc['chg_1y'] == pytest.approx(-0.28)
    assert btc['ath_off'] == pytest.approx(-0.517) and btc['ath_date'] == '2025-10-06'
    assert btc['supply_issued'] == pytest.approx(0.9)
    assert btc['weight'] + eth['weight'] == pytest.approx(1.0)
    assert btc['corr_btc_1y'] == pytest.approx(1.0)   # bitcoin against itself
    assert eth['corr_btc_1y'] is None        # both grow evenly: no variance
    assert len(btc['spark']) == 366 and btc['spark'][-1][0] == '2026-11-20'
    assert btc['corr_spy_1y'] is None        # no SPY given
    json.dumps(rep, allow_nan=False)         # strict JSON for the app


def test_a_coin_without_history_keeps_its_market_columns():
    rep = cm.build_report([_market('solana', 'sol', 7)], {}, {}, AS_OF, 'g')
    r = rep['rows'][0]
    assert r['price'] == 10.0 and r['chg_24h'] == pytest.approx(0.015)
    assert r['vol_1y'] is None and r['history_days'] == 0 and r['spark'] == []


def test_every_table_column_has_a_tooltip():
    for k in ('price', 'market_cap', 'chg_24h', 'chg_7d', 'chg_30d', 'ytd', 'chg_1y',
              'weight', 'volume_24h', 'vol_30d', 'vol_1y', 'max_dd_1y', 'off_high_1y',
              'ath_off', 'corr_btc_1y', 'corr_spy_1y'):
        assert cm.UI.get(k)


# ---------------------------------------------------------------- client
class _Resp:
    def __init__(self, status, body=None, headers=None):
        self.status_code, self._body, self.headers = status, body, headers or {}

    def json(self):
        return self._body


class _Session:
    def __init__(self, *resps):
        self.resps, self.calls = list(resps), []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params, headers))
        return self.resps.pop(0)


def _client(tmp_path, *resps, key=None):
    slept = []
    c = cc.CoinGeckoClient(api_key=key or '', cache_dir=str(tmp_path), interval=0,
                           session=_Session(*resps), sleep=slept.append)
    return c, slept


def test_a_429_waits_and_retries(tmp_path):
    c, slept = _client(tmp_path, _Resp(429, headers={'Retry-After': '30'}), _Resp(200, [{'id': 'bitcoin'}]))
    assert c.markets(5) == [{'id': 'bitcoin'}]
    assert slept == [30.0] and c.calls == 2


def test_persistent_429_raises(tmp_path):
    c, _ = _client(tmp_path, *[_Resp(429)] * (cc.RETRIES_429 + 1))
    with pytest.raises(cc.CoinGeckoError):
        c.markets()


def test_demo_key_is_sent(tmp_path):
    c, _ = _client(tmp_path, _Resp(200, []), key='k123')
    c.markets()
    assert c.session.calls[0][2]['x-cg-demo-api-key'] == 'k123'
    assert cc.CoinGeckoClient(api_key='k').throttle.delay == cc.KEYED_INTERVAL
    assert cc.CoinGeckoClient(api_key='').throttle.delay == cc.PUBLIC_INTERVAL


def test_history_keeps_whole_days_only(tmp_path):
    ms = lambda y, m, d, h=0: datetime(y, m, d, h, tzinfo=timezone.utc).timestamp() * 1000  # noqa: E731
    body = {'prices': [[ms(2026, 11, 19), 1.0], [ms(2026, 11, 20), 2.0], [ms(2026, 11, 20, 21), 2.5]]}
    c, _ = _client(tmp_path, _Resp(200, body))
    assert c.history('bitcoin') == [(date(2026, 11, 19), 1.0), (date(2026, 11, 20), 2.0)]


def test_categories_are_cached_for_a_month(tmp_path):
    c, _ = _client(tmp_path, _Resp(200, {'categories': ['Stablecoins', None]}),
                   _Resp(200, {'categories': ['Changed']}))
    assert c.categories('tether', today=date(2026, 11, 1)) == ['Stablecoins']
    assert c.categories('tether', today=date(2026, 11, 20)) == ['Stablecoins']
    assert c.calls == 1
    assert c.categories('tether', today=date(2026, 12, 15)) == ['Changed']


# ---------------------------------------------------------------- script
class _Fake:
    def __init__(self, markets, cats, hist, fail=()):
        self.m, self.cats, self.hist, self.fail, self.calls = markets, cats, hist, set(fail), 0

    def markets(self, n):
        self.calls += 1
        return self.m

    def categories(self, cid):
        self.calls += 1
        if ('cat', cid) in self.fail:
            raise cc.CoinGeckoError('429')
        return self.cats.get(cid, [])

    def history(self, cid):
        self.calls += 1
        if ('hist', cid) in self.fail:
            raise cc.CoinGeckoError('429')
        return self.hist.get(cid, [])


def test_build_falls_back_to_the_price_for_an_unknown_stablecoin(tmp_path):
    markets = [_market('bitcoin', 'btc', 1), _market('mystery-usd', 'musd', 2, price=1.001),
               _market('ethereum', 'eth', 3)]
    fake = _Fake(markets, {}, {'bitcoin': _series(400), 'ethereum': _series(400)},
                 fail={('cat', 'mystery-usd'), ('hist', 'ethereum')})
    rep = script.build(fake, AS_OF, 2, str(tmp_path))
    assert [r['symbol'] for r in rep['rows']] == ['BTC', 'ETH']
    assert rep['excluded'][0]['symbol'] == 'MUSD'
    assert any('ethereum: history' in e for e in rep['errors'])
    path = script.write_report(rep, str(tmp_path / 'out'))
    assert path.endswith('crypto_report_2026-11-20.json')


def test_spy_is_read_from_the_price_parquet(tmp_path):
    pd = pytest.importorskip('pandas')
    pytest.importorskip('pyarrow')
    idx = pd.to_datetime(['2026-11-18', '2026-11-19'])
    pd.DataFrame({'Close': [600.0, float('nan')], 'Open': [1.0, 1.0]}, index=idx).to_parquet(tmp_path / 'SPY.parquet')
    assert script.load_spy(str(tmp_path)) == [(date(2026, 11, 18), 600.0)]
    assert script.load_spy(str(tmp_path / 'none')) is None
