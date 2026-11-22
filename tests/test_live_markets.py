"""The intraday Markets board (scripts/live_markets.py): rebuilt from Yahoo in
market hours, uploaded to the site's LIVE KV for the iOS app."""
from datetime import date, timedelta

from scripts.live_markets import (
    build_live_board, fx_symbol, merge_closes, session_ticks, yahoo_symbols,
)
from scripts.macro_dashboard import BOARD

TODAY = date(2026, 11, 20)


def _series(n=400, start=100.0, step=0.1):
    return {TODAY - timedelta(days=n - i): start + i * step for i in range(n + 1)}


def _nightly():
    return {'as_of': '2026-11-19', 'panels': [
        {'k': 'eq', 'l': 'Equities', 'rows': [{'k': 'spx', 'l': 'old', 'hist': {'d': [], 'v': []}}]},
        {'k': 'rt', 'l': 'Rates', 'rows': [{'k': 'us10y', 'l': 'US 10Y', 'chg': {'1d': 3.0},
                                            'hist': {'d': ['2026-11-18'], 'v': [5.1]}}]},
    ]}


def test_symbols_cover_live_panels_only():
    yf_syms, fx = yahoo_symbols()
    assert '^GSPC' in yf_syms and '^DJI' in yf_syms and 'CL=F' in yf_syms
    assert fx['EUR'] == 'EURUSD=X' == fx_symbol('EUR')
    rates = {m['sym'] for p in BOARD if p['k'] == 'rt' for m in p['rows']}
    assert not rates & set(yf_syms)


def test_merge_closes_lays_recent_over_history():
    out = merge_closes({TODAY: 1.0, date(2026, 11, 19): 2.0}, {TODAY: 3.0, date(2026, 11, 18): None})
    assert out == {TODAY: 3.0, date(2026, 11, 19): 2.0}
    assert merge_closes(None, {TODAY: 1.0}) == {TODAY: 1.0}


def test_board_rebuilds_yahoo_rows_and_copies_rates_without_hist():
    yf_syms, fx = yahoo_symbols()
    board = build_live_board(_nightly(), {'^GSPC': _series(), '^DJI': _series()},
                             {'EUR': _series(start=1.0, step=0.001)}, TODAY, '2026-11-20T15:00:00Z')
    assert board['as_of'] == '2026-11-20' and board['live'] is True
    assert board['night'] == '2026-11-19' and board['updated'] == '2026-11-20T15:00:00Z'
    panels = {p['k']: p for p in board['panels']}
    assert [r['k'] for r in panels['eq']['rows']] == ['spx', 'dji']     # BOARD order; others missing
    spx = panels['eq']['rows'][0]
    assert spx['latest']['d'] == '2026-11-20' and spx['chg']['1d'] is not None
    assert 'hist' not in spx and spx['spark']
    assert panels['rt']['rows'] == [{'k': 'us10y', 'l': 'US 10Y', 'chg': {'1d': 3.0}}]
    assert [r['k'] for r in panels['fx']['rows']] == ['eur']
    assert 'cm' not in panels


def test_no_live_rows_is_no_board():
    assert build_live_board(_nightly(), {}, {}, TODAY, 'x') is None


def test_session_ticks_every_15_minutes_to_16_15():
    ticks = session_ticks(TODAY)
    assert (ticks[0].hour, ticks[0].minute) == (9, 30)
    assert (ticks[-1].hour, ticks[-1].minute) == (16, 15)
    assert len(ticks) == 28


def test_history_cache_retries_symbols_yahoo_missed(tmp_path, monkeypatch):
    import data.macro_client as mc
    import scripts.live_markets as lm
    monkeypatch.setattr(lm, 'OUT_DIR', tmp_path)
    calls = []
    answers = {'A': _series(10), 'B': {}}
    monkeypatch.setattr(mc, 'fetch_yahoo_closes', lambda s, start: calls.append(s) or answers[s])
    out = lm.history_cache(TODAY, ['A', 'B'])
    assert out['A'] and not out.get('B') and calls == ['A', 'B']
    answers['B'] = _series(10)                       # Yahoo recovers next tick
    out = lm.history_cache(TODAY, ['A', 'B'])
    assert calls == ['A', 'B', 'B'] and out['A'] and out['B']
    lm.history_cache(TODAY, ['A', 'B'])              # both cached now: no fetch
    assert calls == ['A', 'B', 'B']


def test_a_crashing_tick_does_not_end_the_loop(monkeypatch, capsys):
    import scripts.live_markets as lm
    import scripts.market_open as mo
    monkeypatch.setattr(mo, 'market_status', lambda d: (True, 'open'))
    monkeypatch.setattr(lm, 'session_ticks', lambda d: [lm.datetime.now(lm.ET)] * 3)
    monkeypatch.setattr(lm.time, 'sleep', lambda s: None)
    results = iter([RuntimeError('boom'), True, True])

    def fake_tick(dry_run):
        r = next(results)
        if isinstance(r, Exception):
            raise r
        return r
    monkeypatch.setattr(lm, 'tick', fake_tick)
    assert lm.loop() == 0
    out = capsys.readouterr().out
    assert 'tick failed: RuntimeError: boom' in out and 'RESULT ok=2 failed=1' in out


def test_launcher_counts_only_real_failures(tmp_path):
    import subprocess
    log = tmp_path / 'p.log'
    log.write_text('10:00 uploaded 1 bytes\n10:15 upload failed: x\n10:30 no rows built\n'
                   'RESULT ok=1 failed=2\n', encoding='utf-8')
    script = open('scheduled-tasks/live-markets/mac_run.sh', encoding='utf-8').read()
    line = next(ln for ln in script.splitlines() if 'previous failed ticks' in ln).strip()
    out = subprocess.run(['bash', '-c', f'PREV={log}; {line}'], capture_output=True, text=True).stdout
    assert out.strip() == 'previous failed ticks: 2'
    log.write_text('10:00 uploaded 1 bytes\nRESULT ok=1 failed=0\n', encoding='utf-8')
    out = subprocess.run(['bash', '-c', f'PREV={log}; {line}'], capture_output=True, text=True).stdout
    assert out.strip() == 'previous failed ticks: 0'


def _ts(d, hh, mm):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    return int(datetime(d.year, d.month, d.day, hh, mm, tzinfo=ZoneInfo('America/New_York')).timestamp())


def test_live_prices_keep_only_todays_trades():
    from scripts.live_markets import build_live_prices
    quotes = [
        {'symbol': 'AAPL', 'regularMarketPrice': 231.456789, 'regularMarketTime': _ts(TODAY, 10, 44),
         'regularMarketChangePercent': 1.2345},
        {'symbol': 'HALT', 'regularMarketPrice': 5.0, 'regularMarketTime': _ts(TODAY - timedelta(days=1), 15, 59)},
        {'symbol': 'ZERO', 'regularMarketPrice': 0, 'regularMarketTime': _ts(TODAY, 10, 44)},
        {'symbol': 'NOPX', 'regularMarketTime': _ts(TODAY, 10, 44)},
        {'symbol': 'NODP', 'regularMarketPrice': 12, 'regularMarketTime': _ts(TODAY, 9, 31)},
    ]
    out = build_live_prices(quotes, TODAY, '2026-11-20T15:45:00Z', '2026-11-19')
    assert out['as_of'] == '2026-11-20' and out['night'] == '2026-11-19' and out['live'] is True
    assert out['q'] == {'AAPL': [231.4568, 0.012345, _ts(TODAY, 10, 44)],
                        'NODP': [12.0, None, _ts(TODAY, 9, 31)]}


def test_live_prices_none_when_nothing_traded_today():
    from scripts.live_markets import build_live_prices
    old = [{'symbol': 'AAPL', 'regularMarketPrice': 1.0, 'regularMarketTime': _ts(TODAY - timedelta(days=3), 16, 0)}]
    assert build_live_prices(old, TODAY, 'x', None) is None
    assert build_live_prices([], TODAY, 'x', None) is None
