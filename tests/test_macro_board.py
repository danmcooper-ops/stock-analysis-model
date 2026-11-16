"""The Macro Overview's market board (scripts/macro_dashboard.BOARD): change
math, quote conventions, as-of slicing and per-row failure. No network."""
from datetime import date, timedelta

import pandas as pd
import pytest

from data.fx_store import FXStore
from scripts.macro_dashboard import (
    BOARD, _board_summary, _build_board, board_changes, build_macro_payload,
)

AS_OF = date(2026, 11, 13)  # a Friday


def _daily(end, n, start_v, step):
    """Weekday closes ending at ``end``, rising by ``step`` a bar."""
    out, d, v = {}, end, start_v + step * (n - 1)
    while len(out) < n:
        if d.weekday() < 5:
            out[d] = v
            v -= step
        d -= timedelta(days=1)
    return out


class _Fred:
    def __init__(self, table):
        self.table = table

    def fetch_series(self, sid, start=None, end=None, force=False):
        return dict(self.table.get(sid, {}))


def _row(board, k):
    for p in board['panels']:
        for r in p['rows']:
            if r['k'] == k:
                return r
    return None


def test_pct_changes_by_window():
    obs = {date(2025, 11, 13): 80.0, date(2025, 12, 31): 90.0,
           date(2026, 10, 14): 95.0, date(2026, 11, 6): 98.0,
           date(2026, 11, 12): 99.0, date(2026, 11, 13): 100.0}
    c = board_changes(obs, date(2026, 11, 13), 'pct')
    assert c['1d'] == pytest.approx(1.01, abs=0.01)
    assert c['1w'] == pytest.approx(2.04, abs=0.01)
    assert c['1m'] == pytest.approx(5.26, abs=0.01)
    assert c['ytd'] == pytest.approx(11.11, abs=0.01)
    assert c['1y'] == pytest.approx(25.0)


def test_bp_changes_and_missing_bases():
    obs = {date(2026, 11, 12): 4.60, date(2026, 11, 13): 4.63}
    c = board_changes(obs, date(2026, 11, 13), 'bp')
    assert c['1d'] == pytest.approx(3.0)
    # No bar within the lookback of a week, a month, a year ago: no change.
    assert c['1w'] is None and c['1m'] is None and c['1y'] is None


def test_monthly_row_has_no_short_window_changes():
    obs = {date(2026, 8, 1): 2.5, date(2026, 9, 1): 2.6, date(2026, 10, 1): 2.71}
    c = board_changes(obs, date(2026, 10, 1), 'bp', freq='m')
    assert c['1d'] is None and c['1w'] is None
    assert c['1m'] == pytest.approx(11.0)


def test_bars_after_the_run_date_are_ignored():
    closes = _daily(AS_OF + timedelta(days=7), 300, 100.0, 0.1)
    b = _build_board(None, lambda s, start: closes if s == '^GSPC' else {},
                     None, AS_OF)
    r = _row(b, 'spx')
    assert r['latest']['d'] == AS_OF.isoformat()
    assert all(d <= AS_OF.isoformat() for d in r['hist']['d'])


def test_fx_quotes_follow_market_convention(tmp_path):
    store = FXStore(str(tmp_path))
    idx = pd.to_datetime([AS_OF - timedelta(days=1), AS_OF])
    store._write('JPY', pd.DataFrame({'rate': [1 / 150.0, 1 / 151.5],
                                      'source': ['yahoo', 'yahoo']}, index=idx))
    store._write('EUR', pd.DataFrame({'rate': [1.10, 1.12],
                                      'source': ['yahoo', 'yahoo']}, index=idx))
    b = _build_board(None, None, store, AS_OF)
    jpy, eur = _row(b, 'jpy'), _row(b, 'eur')
    assert jpy['l'] == 'USD/JPY' and jpy['latest']['v'] == pytest.approx(151.5)
    assert jpy['chg']['1d'] == pytest.approx(1.0)
    assert eur['latest']['v'] == pytest.approx(1.12)


def test_a_failed_row_drops_alone_and_an_empty_board_is_none():
    def yahoo(sym, start):
        if sym == '^NDX':
            raise RuntimeError('Too Many Requests')
        return _daily(AS_OF, 30, 50.0, 1.0) if sym == '^GSPC' else {}
    b = _build_board(None, yahoo, None, AS_OF)
    assert [p['k'] for p in b['panels']] == ['eq']
    assert [r['k'] for r in b['panels'][0]['rows']] == ['spx']
    assert _build_board(None, None, None, AS_OF) is None


def test_stale_row_is_flagged():
    closes = _daily(AS_OF - timedelta(days=30), 40, 10.0, 0.1)
    b = _build_board(None, lambda s, start: closes if s == 'CL=F' else {},
                     None, AS_OF)
    assert _row(b, 'wti')['stale'] is True


def test_rows_reuse_closes_the_series_cards_fetched():
    calls = []

    def yahoo(sym, start):
        calls.append(sym)
        return {}
    gold = _daily(AS_OF, 30, 3000.0, 5.0)
    b = _build_board(None, yahoo, None, AS_OF, cache={'GC=F': gold})
    assert 'GC=F' not in calls and _row(b, 'gold') is not None


def test_payload_carries_board_and_slim_summary():
    table = {sid: _daily(AS_OF, 400, 4.0, 0.001)
             for sid in ('DGS10', 'T10Y2Y', 'T10Y3M', 'DFF', 'DFII10', 'DGS2',
                         'DGS30', 'BAMLC0A0CM', 'BAMLH0A0HYM2')}
    p = build_macro_payload(_Fred(table), as_of=AS_OF)
    board = p['sidecar']['board']
    assert _row(board, 'us2y')['chg']['1d'] == pytest.approx(0.1)
    slim = p['summary']['board']
    assert all('hist' not in r for pn in slim['panels'] for r in pn['rows'])
    assert all('spark' in r for pn in slim['panels'] for r in pn['rows'])
    assert _board_summary(None) is None


def test_board_row_keys_are_unique():
    keys = [r['k'] for p in BOARD for r in p['rows']]
    assert len(keys) == len(set(keys))
