# tests/test_splits.py
"""Split restatement of EDGAR share counts and EPS (data/splits.py).

The fixtures are shaped on real companyfacts series (output/hist on
2026-10-12): GOOGL's period-end shares, where as-filed and restated quarters
interleave around the 2022 20:1 split; CMG's zero-valued 2022-04-25 point;
PANW's pre-IPO jump that once absorbed its 3:1 split.
"""
from datetime import date

import pandas as pd
import pytest

from data.splits import (
    APPLIED_KEY,
    adjust_for_ticker,
    adjust_history,
    is_split_ratio,
    split_events_from_prices,
)

GOOGL_SPLIT = [(date(2022, 7, 18), 20.0)]


def _googl_shares():
    # Millions, as filed. 2021-12-31 and 2022-06-30 were re-reported after
    # the split; 2022-03-31 was not.
    raw = {
        '2018-12-31': 696, '2019-12-31': 688, '2020-12-31': 675,
        '2021-09-30': 665, '2021-12-31': 13242, '2022-03-31': 659,
        '2022-06-30': 13078, '2022-09-30': 12971, '2022-12-31': 12849,
        '2023-12-31': 12460, '2024-12-31': 12211, '2025-12-31': 12088,
    }
    return {k: v * 1e6 for k, v in raw.items()}


@pytest.mark.parametrize('r', [2.0, 3.0, 1.5, 1.25, 1.2, 1.3, 20.0, 50.0, 7.0,
                               0.5, 0.1, 0.05, 1 / 15, 4 / 3])
def test_split_ratios_are_recognised(r):
    assert is_split_ratio(r)


@pytest.mark.parametrize('r', [1.281, 1.253, 1.319, 1.241, 1.398, 1.555, 9.295,
                               0.807, 1.998, 0, -2, float('nan'), None, 'x'])
def test_spin_off_adjustments_are_not_splits(r):
    # GE 1.281/1.253, DHR 1.319, FDX 1.241, A 1.398: price factors, not counts.
    assert not is_split_ratio(r)


def test_interleaved_bases_are_each_restated():
    adj, applied = adjust_history({'shares_history': _googl_shares()}, GOOGL_SPLIT)
    sh = {k: round(v / 1e6) for k, v in adj['shares_history'].items()}
    assert sh['2018-12-31'] == 13920 and sh['2020-12-31'] == 13500
    assert sh['2021-12-31'] == 13242 and sh['2022-03-31'] == 13180
    assert sh['2022-06-30'] == 13078 and sh['2025-12-31'] == 12088
    assert applied == [('shares_history', 5)]
    assert adj[APPLIED_KEY] == [['2022-07-18', 20.0]]


def test_eps_is_judged_through_its_implied_count():
    # GOOGL diluted EPS: FY21 still as filed ($112.20), FY22 on the new basis.
    hist = {
        'earnings_history': {'2020-12-31': 40269e6, '2021-12-31': 76033e6,
                             '2022-12-31': 59972e6, '2023-12-31': 73795e6},
        'eps_diluted_history': {'2020-12-31': 58.61, '2021-12-31': 112.20,
                                '2022-12-31': 4.56, '2023-12-31': 5.80},
    }
    eps = adjust_history(hist, GOOGL_SPLIT)[0]['eps_diluted_history']
    assert eps == pytest.approx({'2020-12-31': 2.9305, '2021-12-31': 5.61,
                                 '2022-12-31': 4.56, '2023-12-31': 5.80})


def test_already_restated_series_is_left_alone():
    flat = {f'{y}-12-31': 1e9 * (1 - 0.01 * (y - 2018)) for y in range(2018, 2026)}
    adj, applied = adjust_history({'shares_history': flat}, GOOGL_SPLIT)
    assert applied == [] and adj['shares_history'] == flat


def test_a_restated_history_is_never_restated_again():
    adj, _ = adjust_history({'shares_history': _googl_shares()}, GOOGL_SPLIT)
    again, applied = adjust_history(adj, GOOGL_SPLIT)
    assert applied == [] and again is adj
    # A later split is still applied, and both are remembered.
    nxt, applied = adjust_history(adj, GOOGL_SPLIT + [(date(2026, 1, 5), 2.0)])
    assert nxt[APPLIED_KEY] == [['2022-07-18', 20.0], ['2026-01-05', 2.0]]


def test_a_unit_scale_spike_does_not_place_the_split():
    # CMG: 50:1 on 2024-06-26, and a 0-share point two years before it.
    raw = {'2021-10-22': 28.0, '2022-02-04': 28.0, '2022-04-25': 2.8e-5,
           '2022-07-25': 28.0, '2023-07-24': 27.6, '2024-04-22': 27.5,
           '2024-07-22': 1369.5, '2025-01-31': 1355.3}
    sh = {k: v * 1e6 for k, v in raw.items()}
    out = adjust_history({'shares_history': sh}, [(date(2024, 6, 26), 50.0)])[0]
    got = {k: round(v / 1e6) for k, v in out['shares_history'].items()}
    assert got['2021-10-22'] == 1400 and got['2024-04-22'] == 1375
    assert got['2024-07-22'] == 1370
    assert out['shares_history']['2022-04-25'] == sh['2022-04-25']   # kept as filed


def test_a_real_jump_long_before_the_split_does_not_absorb_it():
    # PANW: pre-IPO count x3.4 in 2011-12, then 3:1 (2022) and 2:1 (2024).
    raw = {'2011-07-31': 19.8, '2012-07-31': 67.9, '2014-07-31': 79.5,
           '2016-07-31': 90.5, '2018-07-31': 93.6, '2020-07-31': 96.3,
           '2021-07-31': 97.3, '2022-04-30': 99.6, '2022-07-31': 298.8,
           '2023-07-31': 308.3, '2024-04-30': 323.4, '2024-07-31': 650.2,
           '2025-07-31': 668.0}
    sh = {k: v * 1e6 for k, v in raw.items()}
    ev = [(date(2022, 9, 14), 3.0), (date(2024, 12, 16), 2.0)]
    got = {k: round(v / 1e6) for k, v in
           adjust_history({'shares_history': sh}, ev)[0]['shares_history'].items()}
    assert got['2016-07-31'] == 543 and got['2021-07-31'] == 584
    assert got['2022-07-31'] == 598 and got['2024-07-31'] == 650


def test_other_series_and_small_stock_dividends_pass_through():
    hist = {'shares_history': _googl_shares(), 'revenue_history': {'2020': 1.0}}
    adj, _ = adjust_history(hist, [(date(2010, 3, 1), 1.0077)])
    assert adj is hist
    adj, _ = adjust_history(hist, GOOGL_SPLIT)
    assert adj['revenue_history'] is hist['revenue_history']
    assert hist['shares_history']['2018-12-31'] == 696e6     # input untouched


def test_events_come_from_the_price_parquet(tmp_path):
    idx = pd.to_datetime(['2014-04-03', '2020-01-02', '2022-07-18', '2023-07-05'])
    pd.DataFrame({'Close': [1.0, 2.0, 3.0, 4.0],
                  'Stock Splits': [1.998, 0.0, 20.0, 1.281]},
                 index=idx).to_parquet(tmp_path / 'GOOGL.parquet')
    pd.DataFrame({'Close': [1.0]}, index=idx[:1]).to_parquet(tmp_path / 'ONLY.parquet')
    assert split_events_from_prices(str(tmp_path), 'GOOGL') == GOOGL_SPLIT
    assert split_events_from_prices(str(tmp_path), 'ONLY') == []
    assert split_events_from_prices(str(tmp_path), 'NONE') == []
    hist = {'shares_history': _googl_shares()}
    assert adjust_for_ticker(hist, 'GOOGL', str(tmp_path))[APPLIED_KEY]
    assert adjust_for_ticker(hist, 'NONE', str(tmp_path)) is hist
    assert adjust_for_ticker(hist, 'GOOGL', None) is hist
