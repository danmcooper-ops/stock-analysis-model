# tests/test_sector_pool.py
"""The Sector Analysis page's pool history (models/sector_pool.py).

Each rule below came from a real sector going wrong on the first pass over
the 2026-10-08 snapshot: Energy's growth window fell back to 2017-22 because
Sinopec and PetroChina stopped filing after 2022, and once that was fixed its
single-year FY2020 endpoint (a sixth of the neighbouring years' pool) made the
pool "grow" 71% a year.
"""
import math

import pytest

from models.sector_pool import (
    COMPLETE_COVERAGE_RATIO, ENDPOINT_BLOCK, panel_pools, pool_rows,
    sector_pool_history, year_series)
from scripts.scoring import _compute_pool_share_trajectory


def _row(t, rev_by, oi_by, revenue=None, oi=None, **kw):
    """A pool row whose today's revenue/operating income default to its
    latest history year."""
    ly = max(oi_by)
    return dict({
        'ticker': t, 'sector': 'Tech', 'company_name': t + ' Inc',
        'revenue': revenue if revenue is not None else rev_by.get(ly, 100.0),
        'operating_income': oi if oi is not None else oi_by[ly],
        'edgar_history': {'revenue_history': {str(y): v for y, v in rev_by.items()},
                          'operating_income_history': {str(y): v for y, v in oi_by.items()}},
    }, **kw)


def _flat(t, years, rev, margin, growth=0.0):
    # A per-ticker nudge keeps identical fixtures from reading as duplicate
    # listings of one issuer (pool_rows folds rows with equal statements).
    rev = rev + sum(map(ord, t)) * 1e-6
    rev_by = {y: rev * (1 + growth) ** (y - years[0]) for y in years}
    return _row(t, rev_by, {y: v * margin for y, v in rev_by.items()})


YEARS = list(range(2016, 2026))


def test_year_series_parses_live_and_round_tripped_keys():
    assert year_series({2024: 1, '2023': 2.5, '2022-12': 3, 'x': 4,
                        2021: None, 2020: True}) == {2024: 1.0, 2023: 2.5, 2022: 3.0}
    assert year_series(None) == {}


def test_panel_pools_counts_only_histories_holding_both_years():
    hs = [{2020: 10, 2025: 20}, {2020: -5, 2025: 5}, {2025: 100}]
    assert panel_pools(hs, 2020, 2025) == (10.0, 25.0, 2)


def test_scoring_gate_unchanged_by_the_shared_helpers():
    """The Pool Share gate now parses years and sums its panel through the
    shared helpers; the value it produces must not move."""
    rows = [_flat('A', YEARS, 100, 0.1, 0.10), _flat('B', YEARS, 100, 0.1),
            _flat('C', YEARS, 100, 0.1)]
    _compute_pool_share_trajectory(rows)
    a = rows[0]['pool_share_cagr']
    # A grows 10%/yr against two flat peers: its share goes from 1/3 to
    # 1.1^9 / (1.1^9 + 2) over the 2020->2025 window... measured on 2020/2025.
    s0 = 1.1 ** 4 / (1.1 ** 4 + 2)
    s1 = 1.1 ** 9 / (1.1 ** 9 + 2)
    assert a == pytest.approx((s1 / s0) ** (1 / 5) - 1)
    assert rows[1]['pool_share_cagr'] < 0


def test_pool_rows_drop_duplicate_listings_and_rows_without_income():
    a = _flat('FNMA', YEARS, 100, 0.2)
    rows = [a, dict(a, ticker='FNMAS'), dict(a, ticker='X', operating_income=None),
            dict(a, ticker='Y', revenue=0)]
    assert [r['ticker'] for r in pool_rows(rows)] == ['FNMA']


def test_decomposition_identity_holds_exactly():
    rows = [_flat('A', YEARS, 100, 0.10, 0.05), _flat('B', YEARS, 200, 0.20, 0.02),
            _flat('C', YEARS, 50, 0.05, 0.08)]
    rows[0]['edgar_history']['operating_income_history'] = {
        str(y): 100 * 1.05 ** (y - 2016) * (0.10 + 0.01 * (y - 2016)) for y in YEARS}
    d = sector_pool_history(rows)['decomposition']
    assert d['block'] == ENDPOINT_BLOCK and (d['y0'], d['y1']) == (2020, 2025)
    assert (1 + d['pool_cagr']) == pytest.approx(
        (1 + d['revenue_cagr']) * (1 + d['margin_cagr']), rel=1e-12)
    assert d['margin1'] > d['margin0']


def test_endpoints_average_three_years_so_a_trough_does_not_set_the_answer():
    """Energy FY2020: one collapsed year at the start of the window."""
    rows = [_flat(t, YEARS, 100, 0.15) for t in 'ABCD']
    for r in rows:
        r['edgar_history']['operating_income_history']['2020'] = 1.0
    d = sector_pool_history(rows)['decomposition']
    single_year = (15.0 / 1.0) ** (1 / 5) - 1          # ~72%/yr
    assert d['pool_cagr'] < 0.25 < single_year
    assert d['pool_cagr'] == pytest.approx((45 / 31) ** (1 / 5) - 1, rel=1e-6)


def test_short_history_falls_back_to_single_year_endpoints():
    yrs = list(range(2019, 2026))                        # 2018 missing for 3y blocks
    h = sector_pool_history([_flat(t, yrs, 100, 0.1) for t in 'ABC'])
    assert h['window'] == [2020, 2025, 1]


def test_incomplete_latest_year_is_flagged_and_never_an_endpoint():
    rows = [_flat(t, YEARS, 100, 0.1) for t in 'ABCDE']
    # Only A has filed FY2026 (a January year end).
    rows[0]['edgar_history']['revenue_history']['2026'] = 300
    rows[0]['edgar_history']['operating_income_history']['2026'] = 90
    h = sector_pool_history(rows)
    last = h['points'][-1]
    assert last['year'] == 2026 and last['complete'] is False
    assert last['coverage'] < COMPLETE_COVERAGE_RATIO
    assert h['window'][1] == 2025
    assert h['cycle']['year'] == 2025


def test_stale_filers_leave_the_series_and_are_named():
    """Sinopec/PetroChina: big, still trading, no SEC history after 2022."""
    rows = [_flat(t, YEARS, 100, 0.1) for t in 'ABC']
    stale = _row('SNPMF', {y: 1000 for y in range(2013, 2023)},
                 {y: 50 for y in range(2013, 2023)}, revenue=1000, oi=50)
    h = sector_pool_history(rows + [stale])
    assert [s['ticker'] for s in h['stale']] == ['SNPMF']
    assert h['stale'][0]['last_year'] == 2022
    assert h['n_with_history'] == 3 and h['n_rows'] == 4
    assert all(p['revenue'] == pytest.approx(300) for p in h['points'])
    assert h['window'][1] == 2025


def test_a_one_year_filing_lag_is_not_stale():
    rows = [_flat(t, YEARS, 100, 0.1) for t in 'ABC']
    rows.append(_flat('LAG', YEARS[:-1], 50, 0.1))
    h = sector_pool_history(rows)
    assert h['stale'] == [] and h['n_with_history'] == 4


def test_shifts_net_to_zero_and_rank_gainers_and_losers():
    rows = [_flat('UP', YEARS, 100, 0.1, 0.15), _flat('DOWN', YEARS, 100, 0.1, -0.05),
            _flat('FLAT', YEARS, 100, 0.1), _flat('FLAT2', YEARS, 100, 0.1)]
    sh = sector_pool_history(rows)['shifts']
    assert sh['gainers'][0]['ticker'] == 'UP'
    assert sh['losers'][0]['ticker'] == 'DOWN'
    moves = sh['gainers'] + sh['losers']
    assert sum(m['delta'] for m in moves) == pytest.approx(0, abs=1e-12)
    assert all(math.isclose(m['delta'], m['share1'] - m['share0']) for m in moves)


def test_hhi_trend_reads_concentration():
    rows = [_flat('BIG', YEARS, 100, 0.1, 0.25)] + [
        _flat(t, YEARS, 100, 0.1) for t in 'ABC']
    assert sector_pool_history(rows)['hhi_trend']['trend'] == 'concentrating'
    flat = [_flat(t, YEARS, 100, 0.1) for t in 'ABCD']
    assert sector_pool_history(flat)['hhi_trend']['trend'] == 'stable'


def test_cycle_position_against_the_sectors_own_range():
    rows = [_flat(t, YEARS, 100, 0.1) for t in 'ABC']
    for r in rows:
        oi = r['edgar_history']['operating_income_history']
        for y in YEARS:
            oi[str(y)] = 100 * (0.05 + 0.01 * (y - 2016))   # rising every year
    c = sector_pool_history(rows)['cycle']
    assert c['position'] == 'top' and c['pctile'] == 1.0
    assert c['margin'] == pytest.approx(c['high'])


def test_thin_panels_give_no_window_analysis():
    h = sector_pool_history([_flat('A', YEARS, 100, 0.1), _flat('B', YEARS, 100, 0.1)])
    assert h['points'] and h['decomposition'] is None and h['shifts'] is None


def test_rows_without_history_give_none():
    r = _flat('A', YEARS, 100, 0.1)
    r['edgar_history'] = {}
    assert sector_pool_history([r]) is None
    assert sector_pool_history([]) is None


def test_a_large_new_entrant_does_not_make_earlier_years_incomplete():
    """Packaged Foods, 2026-10-08: a company whose history starts in 2025
    held a third of today's revenue, and judged against today's revenue
    every earlier year read 43% complete — no growth window at all."""
    rows = [_flat(t, YEARS, 100, 0.1) for t in 'ABC']
    rows.append(_flat('NEW', [2025], 400, 0.1))
    h = sector_pool_history(rows)
    assert all(p['complete'] for p in h['points'])
    assert h['points'][0]['coverage'] < 0.5             # the footnote still says so
    assert h['window'] == [2020, 2025, ENDPOINT_BLOCK]
