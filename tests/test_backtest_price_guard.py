"""Bad price bars must not become forward returns (scripts/backtest.py).

On 2026-09-13 GRKZF's price parquet held a $0.001 close for 2026-08-14
against a recorded snapshot price of $23.35: a +1,552,400% "return" that
swamped every mean it entered.
"""

import json
from datetime import date

import pytest

from scripts.backtest import annotate_snapshot_returns, implausible_forward_return


def _fwd(start, ret):
    return {'start_price': start, 'ret': ret, 'end_price': start * (1 + ret),
            'excess_return': ret, 'spy_return': 0.0}


@pytest.mark.parametrize('snapshot_price, fwd, bad', [
    (23.35, _fwd(0.001, 15523.99), True),    # GRKZF: bad start bar
    (0.43, _fwd(0.03, 13.33), True),         # THFRF: 14x off, +1,333%
    (1697.5, _fwd(33.95, 0.0), False),       # CFNB: split-adjusted history, valid return
    (1.87, _fwd(9.35, 0.17), False),         # TOP: 5x reverse split, valid return
    (26.71, _fwd(10.71, 1.49), False),       # RYKKF: real +149% move, prices agree
    (50.0, _fwd(1.0, -0.95), True),          # disagreement plus a collapse
    (None, _fwd(0.001, 15523.99), False),    # nothing to compare against
    (23.35, {'ret': 5.0}, False),            # old sidecar without start_price
])
def test_implausible_forward_return(snapshot_price, fwd, bad):
    assert implausible_forward_return(snapshot_price, fwd) is bad


def test_guard_filters_returns_from_an_existing_cache(tmp_path):
    # A sidecar cached before the guard existed still carries the bad bar.
    cache = tmp_path / 'returns'
    cache.mkdir()
    (cache / '2026-08-14_h30.json').write_text(json.dumps({
        'run_date': '2026-08-14', 'horizon_days': 30, 'spy_return': 0.0,
        'tickers': {
            'GRKZF': _fwd(0.001, 15523.99),
            'CFNB': _fwd(33.95, 0.0),
            'AAPL': _fwd(250.0, 0.04),
        }}), encoding='utf-8')
    snapshot = {'date': '2026-08-14', 'results': [
        {'ticker': 'GRKZF', 'price': 23.35},
        {'ticker': 'CFNB', 'price': 1697.5},
        {'ticker': 'AAPL', 'price': 251.0},
    ]}
    out = annotate_snapshot_returns(snapshot, [30], yf_client=None, cache_dir=str(cache),
                                    today=date(2026, 9, 14))
    rows = {r['ticker']: r for r in snapshot['results']}
    assert out == {30: 2}
    assert '_fwd' not in rows['GRKZF']
    assert rows['CFNB']['_fwd'][30]['ret'] == 0.0
    assert rows['AAPL']['_fwd'][30]['ret'] == 0.04
