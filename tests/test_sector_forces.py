# tests/test_sector_forces.py
"""Evidence-linked headwinds and tailwinds (models/sector_forces.py)."""
from datetime import date, timedelta

import pytest

from models.narrative import _SECTOR_THESIS_RISKS, _SECTOR_THESIS_TAILWINDS
from models.sector_forces import (
    FORCE_META, _status, evaluate_forces, market_confirmation, sector_forces,
    split_theme)
from scripts.macro_dashboard import MACRO_SERIES


def test_every_force_has_metadata_and_every_entry_a_force():
    """Metadata is keyed by theme so it cannot slide onto the wrong force;
    this pins that the two stay in step when a text changes."""
    assert set(FORCE_META) == set(_SECTOR_THESIS_RISKS) == set(_SECTOR_THESIS_TAILWINDS)
    for sector, meta in FORCE_META.items():
        themes = {split_theme(t)[0] for t in
                  _SECTOR_THESIS_RISKS[sector] + _SECTOR_THESIS_TAILWINDS[sector]}
        assert themes == set(meta), sector


def test_force_texts_are_the_narratives_word_for_word():
    """The Claude macro narrative reads the same strings; this module only
    adds to them."""
    for sector in FORCE_META:
        fs = sector_forces(sector)
        assert [f['text'] for f in fs if f['kind'] == 'headwind'] == _SECTOR_THESIS_RISKS[sector]
        assert [f['text'] for f in fs if f['kind'] == 'tailwind'] == _SECTOR_THESIS_TAILWINDS[sector]
        for f in fs:
            assert f['theme'] and f['detail']
            assert f['theme'] + ' — ' + f['detail'] == f['text']


def test_every_macro_indicator_names_a_dashboard_series():
    ids = {m['id'] for m in MACRO_SERIES}
    for sector, meta in FORCE_META.items():
        for theme, (_typ, horizon, ind) in meta.items():
            assert horizon in ('cyclical', 'secular'), theme
            if ind and ind['kind'] == 'macro':
                assert ind['series'] in ids, (sector, theme)
            if ind:
                assert ind['sign'] in (1, -1)


@pytest.mark.parametrize('pressure,move,expected', [
    (0.90, 0.00, 'active'), (0.70, -0.09, 'active'), (0.90, -0.10, 'easing'),
    (0.50, 0.10, 'building'), (0.20, 0.30, 'building'),
    (0.50, -0.10, 'easing'), (0.30, -0.20, 'dormant'), (0.50, 0.05, 'dormant'),
])
def test_status_edges(pressure, move, expected):
    assert _status(pressure, move) == expected


def _sidecar(series_id, values, fmt='pct2'):
    """A macro sidecar holding one series of monthly *values*, newest last."""
    end = date(2026, 10, 1)
    ds = [(end - timedelta(days=30 * (len(values) - 1 - k))).isoformat()
          for k in range(len(values))]
    return {'as_of': '2026-10-08', 'series': {series_id: {
        'l': 'Test series', 'fmt': fmt, 'suffix': '', 'chg_1y': 1.0, 'pctile': 0.99,
        'latest': {'d': ds[-1], 'v': values[-1]}, 'hist': {'d': ds, 'v': values}}}}


def _force(result, theme):
    return next(f for f in result['forces'] if f['theme'] == theme)


def test_rising_rates_activate_a_rate_headwind_and_silence_the_rate_cut_tailwind():
    side = _sidecar('DGS10', [3.0 + 0.02 * k for k in range(120)])
    res = evaluate_forces('Real Estate', side, {})
    hw = _force(res, 'Interest-rate sensitivity and tenant-credit risk')
    tw = _force(res, 'Rate-cut re-rating of cap rates')
    assert hw['status'] == 'active' and tw['status'] == 'dormant'
    assert hw['evidence']['reading'] == '5.38%'
    assert hw['evidence']['pctile'] == 0.99      # the series' own, as the Macro tab quotes
    assert 0 < len(hw['evidence']['spark']) <= 48


def test_falling_from_a_high_reads_easing():
    vals = [3.0 + 0.02 * k for k in range(100)] + [5.0 - 0.03 * k for k in range(15)]
    res = evaluate_forces('Real Estate', _sidecar('DGS10', vals), {})
    assert _force(res, 'Interest-rate sensitivity and tenant-credit risk')['status'] == 'easing'


def test_without_a_sidecar_macro_forces_read_no_data_and_the_rest_render():
    res = evaluate_forces('Real Estate', None, {})
    statuses = {f['theme']: f['status'] for f in res['forces']}
    assert statuses['Interest-rate sensitivity and tenant-credit risk'] == 'no_data'
    assert statuses['Remote work structural vacancy'] == 'qualitative'
    assert res['market'] is None and res['as_of'] is None
    assert len(res['forces']) == 6


def _entry(ind_cagr, sector_cagr=0.10, industry='Software - Infrastructure'):
    return {'history': {'decomposition': {'pool_cagr': sector_cagr, 'y0': 2020, 'y1': 2025,
                                          'block': 3}},
            'industries': [{'industry': industry, 'pool_cagr': ind_cagr, 'pool_share': 0.2}]}


@pytest.mark.parametrize('ind_cagr,expected', [
    (0.14, 'active'), (0.115, 'building'), (0.105, 'dormant'), (0.02, 'dormant')])
def test_industry_evidence_against_the_sector(ind_cagr, expected):
    res = evaluate_forces('Technology', None, _entry(ind_cagr))
    f = _force(res, 'Software dollar share keeps rising')
    assert f['status'] == expected
    assert f['evidence']['diff'] == pytest.approx(ind_cagr - 0.10)


def test_a_headwind_reads_the_same_industry_the_other_way():
    """App software outgrowing the sector is evidence for digitisation and
    against AI eating its moats."""
    e = _entry(0.20, industry='Software - Application')
    res = evaluate_forces('Technology', None, e)
    assert _force(res, 'Durable enterprise digitisation')['status'] == 'active'
    assert _force(res, 'AI disruption of existing software moats')['status'] == 'dormant'


def test_missing_industry_is_no_data_with_the_reason():
    res = evaluate_forces('Technology', None, _entry(0.2, industry='Other'))
    f = _force(res, 'Software dollar share keeps rising')
    assert f['status'] == 'no_data' and 'fewer than three' in f['evidence']['note']


def test_margin_cycle_evidence():
    pts = [{'year': 2016 + k, 'complete': True, 'margin': m}
           for k, m in enumerate([0.10, 0.11, 0.12, 0.13, 0.14, 0.09])]
    entry = {'history': {'points': pts, 'cycle': {'low': 0.09, 'high': 0.14, 'years': 6}}}
    res = evaluate_forces('Consumer Defensive', None, entry)
    # Margins just fell to the bottom of their range: input-cost pressure is
    # on, pricing power is not.
    assert _force(res, 'Input cost inflation from agricultural commodities')['status'] == 'active'
    assert _force(res, 'Pricing power durability')['status'] == 'dormant'


def test_market_confirmation_reads_the_sector_etf():
    side = {'as_of': '2026-10-08',
            'sector_data': {'Technology': {'etf': 'XLK', 'rs_3m': 0.04, 'rs_6m': 0.25}}}
    m = market_confirmation('Technology', side)
    assert m['etf'] == 'XLK' and m['rs_6m'] == 0.25
    assert market_confirmation('Energy', side) is None


def test_move_is_ranked_on_an_even_weekly_grid():
    """hist is daily for its last year and weekly before; ranked as shipped,
    the last year counted ~3.5x. A flat-high final year sat at the top of
    the even grid but well inside the dense one."""
    from models.sector_forces import _even_weekly
    start = date(2016, 10, 1)
    weekly = [(start + timedelta(days=7 * k), 3.0 + 0.004 * k) for k in range(470)]
    last = weekly[-1][0]
    daily = [(last + timedelta(days=k), 5.0) for k in range(1, 366)]
    even = _even_weekly(weekly + daily)
    assert len(even) == 470 + 52 or len(even) == 470 + 53
    assert sum(1 for d, _ in even if d > last) <= 53

# --- exposure and balance (step 4) ------------------------------------------

from models.sector_forces import exposure, force_balance  # noqa: E402


def _re_rows():
    """Real Estate rows: six levered, six not, plus an industry."""
    rows = []
    for k in range(6):
        rows.append({'ticker': 'LEV%d' % k, 'sector': 'Real Estate', 'industry': 'REIT - Office',
                     'revenue': 100 + k, 'operating_income': 30 + k, 'nd_ebitda': 9.0 - k * 0.1,
                     'rating': 'PASS'})
        rows.append({'ticker': 'CASH%d' % k, 'sector': 'Real Estate', 'industry': 'REIT - Specialty',
                     'revenue': 200 + k, 'operating_income': 60 + k, 'nd_ebitda': -1.0 - k,
                     'rating': 'BUY'})
    return rows


def test_metric_exposure_ranks_and_measures_reach():
    rows = _re_rows()
    e = exposure({'type': 'rates'}, None, rows)
    assert e['basis'] == 'metric' and e['metric'] == 'nd_ebitda'
    assert [m['ticker'] for m in e['most']] == ['LEV0', 'LEV1', 'LEV2']
    assert e['least'][0]['ticker'] == 'CASH5'
    pool = sum(r['operating_income'] for r in rows)
    assert e['pool_share'] == pytest.approx(sum(30 + k for k in range(4)) / pool)


def test_industry_exposure_is_the_industrys_companies():
    e = exposure({'type': 'demand'}, {'kind': 'industry', 'industry': 'REIT - Specialty'}, _re_rows())
    assert e['basis'] == 'industry' and e['n'] == 6
    assert e['most'][0]['ticker'] == 'CASH5' and e['least'] == []
    assert e['pool_share'] == pytest.approx(sum(60 + k for k in range(6)) / (
        sum(30 + k for k in range(6)) + sum(60 + k for k in range(6))))


def test_exposure_lists_name_material_companies_only():
    rows = _re_rows() + [{'ticker': 'TINY', 'sector': 'Real Estate', 'revenue': 1,
                          'operating_income': 0.01, 'nd_ebitda': 50.0}]
    e = exposure({'type': 'rates'}, None, rows)
    assert e['most'][0]['ticker'] == 'LEV0'          # TINY ranks first, holds ~0%


def test_exposure_needs_a_metric_or_an_industry():
    assert exposure({'type': 'regulation'}, None, _re_rows()) is None
    assert exposure({'type': 'rates'}, None, _re_rows()[:4]) is None   # too few rows


def test_balance_weights_live_forces_by_reach():
    forces = [{'kind': 'tailwind', 'status': 'active', 'exposure': {'pool_share': 0.4}},
              {'kind': 'tailwind', 'status': 'building', 'exposure': {'pool_share': 0.2}},
              {'kind': 'tailwind', 'status': 'dormant', 'exposure': {'pool_share': 0.9}},
              {'kind': 'headwind', 'status': 'active', 'exposure': None}]
    b = force_balance(forces)
    assert b['tailwind'] == pytest.approx(0.5)
    assert b['headwind'] == pytest.approx(1 / 3) and b['estimated'] == 1
    assert b['net'] == pytest.approx(0.5 - 1 / 3)


def test_evaluate_forces_carries_exposure_and_balance():
    side = _sidecar('DGS10', [3.0 + 0.02 * k for k in range(120)])
    res = evaluate_forces('Real Estate', side, {}, _re_rows())
    hw = _force(res, 'Interest-rate sensitivity and tenant-credit risk')
    assert hw['exposure']['most'][0]['ticker'] == 'LEV0'
    assert res['balance']['headwind'] == pytest.approx(hw['exposure']['pool_share'])
    assert res['balance']['tailwind'] == 0


def test_industry_and_sector_growth_compare_the_same_years():
    """Independent Power Producers falls back to FY2022-25 (Constellation
    spun off in 2022); the sector's growth must be measured over those years
    too, not quoted from its own 2018-20 -> 2023-25 window."""
    from models.sector_pool import growth_over, industry_pools, sector_pool_history
    from tests.test_sector_pool import YEARS, _flat
    rows = [dict(_flat(t, YEARS, 100 + i, 0.2, 0.02), industry='Utilities - Regulated Electric')
            for i, t in enumerate(('DUK', 'SO', 'NEE'))]
    rows += [dict(_flat('NRG', YEARS, 50, 0.1, 0.10), industry='Utilities - Independent Power Producers'),
             dict(_flat('HNRG', YEARS, 20, 0.1, 0.10), industry='Utilities - Independent Power Producers'),
             dict(_flat('CEG', list(range(2022, 2026)), 80, 0.1, 0.30),
                  industry='Utilities - Independent Power Producers')]
    entry = {'history': sector_pool_history(rows), 'industries': industry_pools(rows)}
    assert entry['history']['window'] == [2020, 2025, 3]
    f = _force(evaluate_forces('Utilities', None, entry, rows), 'AI data-centre electricity demand')
    ev = f['evidence']
    assert ev['window'] == [2022, 2025, 1]
    assert ev['sector_cagr'] == pytest.approx(growth_over(rows, [2022, 2025, 1])['pool_cagr'])
    assert f['status'] == 'active'
