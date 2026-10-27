"""Bonds vs stocks (models/bond_relative.py) and its render-time payload."""
import json
from datetime import date

import pytest

from models.bond_relative import (HY_CREDIT_LOSS, SECTOR_MIN_COMPANIES, bond_lines,
                                  bonds_vs_stocks, model_read, stock_aggregate)

TENORS = ['1M', '3M', '6M', '1Y', '2Y', '3Y', '5Y', '7Y', '10Y', '20Y', '30Y']


def _entry(v, d='2026-10-23', chg=None, pctile=None, win=None):
    return {'latest': {'d': d, 'v': v}, 'chg_1y': chg, 'pctile': pctile, 'pct_win': win}


def _sidecar(series=None, now=None):
    now = now or [4.1, 4.2, 4.3, 4.4, 4.7, 4.75, 4.8, 4.9, 5.0, 5.4, 5.3]
    return {'curve': {'tenors': TENORS, 'now': {'d': '2026-10-23', 'v': now},
                      'y1': {'d': '2025-10-24', 'v': [v - 1 for v in now]}},
            'series': series if series is not None else {
                'DGS10': _entry(5.0, chg=1.0, pctile=0.99, win='20y'),
                'T10YIE': _entry(2.4),
                'BAMLC0A0CM': _entry(0.9), 'BAMLH0A0HYM2': _entry(3.4)}}


def _row(t, mcap, pe=None, ni=None, fcf_y=None, sh_y=None, sector='Technology', **kw):
    return {'ticker': t, 'mcap': mcap, 'pe': pe, 'net_income': ni, 'fcf_yield': fcf_y,
            'shareholder_yield': sh_y, 'sector': sector, **kw}


def test_yields_convert_and_fall_back_to_derived_lines():
    by = {b['id']: b for b in bond_lines(_sidecar())}
    assert by['ust10Y']['y'] == pytest.approx(0.05)
    assert by['ust10Y']['chg_1y'] == pytest.approx(0.01)
    assert by['ust10Y']['pctile'] == 0.99
    # No DFII10: real = nominal - breakeven, flagged.
    assert by['tips10']['y'] == pytest.approx(0.026)
    assert 'breakeven' in by['tips10']['derived']
    # No effective yields: matched-duration Treasury + OAS, flagged.
    assert by['ig']['y'] == pytest.approx(0.049 + 0.009)
    assert by['hy']['gross'] == pytest.approx(0.048 + 0.034)


def test_direct_series_win_over_derived():
    s = _sidecar()
    s['series'].update({'DFII10': _entry(2.5), 'BAMLC0A0CMEY': _entry(5.6, pctile=0.8, win='3y'),
                        'BAMLH0A0HYM2EY': _entry(7.9)})
    by = {b['id']: b for b in bond_lines(s)}
    assert by['tips10']['y'] == pytest.approx(0.025) and 'derived' not in by['tips10']
    assert by['ig']['y'] == pytest.approx(0.056) and by['ig']['pct_win'] == '3y'
    assert 'derived' not in by['hy']


def test_hy_haircut_is_visible():
    hy = {b['id']: b for b in bond_lines(_sidecar())}['hy']
    assert hy['y'] == pytest.approx(hy['gross'] - HY_CREDIT_LOSS)
    assert hy['loss'] == HY_CREDIT_LOSS


def test_missing_series_drops_only_its_line():
    s = _sidecar(series={'DGS10': _entry(5.0)})
    ids = [b['id'] for b in bond_lines(s)]
    assert ids == ['ust3M', 'ust2Y', 'ust10Y', 'ust30Y']


def test_none_without_curve():
    assert bonds_vs_stocks([_row('A', 1e9, pe=10)], {'series': {}}) is None
    assert bonds_vs_stocks([_row('A', 1e9, pe=10)], None) is None


def test_loss_makers_count_against_earnings():
    rows = [_row('A', 100.0, pe=10.0), _row('B', 100.0, pe=None, ni=-5.0)]
    agg = stock_aggregate(rows)
    # (10 - 5) / 200, not 10 / 100.
    assert agg['earnings_yield'] == pytest.approx(0.025)
    assert agg['coverage'] == pytest.approx(1.0)


def test_implausible_flow_is_left_out_and_lender_fcf_ignored():
    rows = [_row('A', 100.0, pe=0.001, fcf_y=0.05),
            _row('BANK', 100.0, pe=10.0, fcf_y=0.9, epv_bridge='equity')]
    agg = stock_aggregate(rows)
    assert agg['earnings_yield'] == pytest.approx(0.1)   # A's 100,000 earnings dropped
    assert agg['coverage'] == pytest.approx(0.5)
    assert agg['fcf_yield'] == pytest.approx(0.05)        # the bank's FCF dropped


def test_sector_floor_and_order():
    rows = [_row(f'T{i}', 100.0, pe=20.0, fcf_y=0.03) for i in range(SECTOR_MIN_COMPANIES)]
    rows += [_row(f'E{i}', 100.0, pe=10.0, sector='Energy') for i in range(SECTOR_MIN_COMPANIES)]
    rows += [_row('U0', 100.0, pe=15.0, sector='Utilities')]
    rows += [_row(f'H{i}', 100.0, pe=(25.0 if i == 0 else None), sector='Healthcare')
             for i in range(SECTOR_MIN_COMPANIES)]
    out = bonds_vs_stocks(rows, _sidecar())
    sec = {e['sector']: e for e in out['sectors']}
    assert [e['sector'] for e in out['sectors']][:2] == ['Energy', 'Technology']
    assert sec['Utilities']['status'] == 'insufficient'
    assert sec['Healthcare']['status'] == 'insufficient'   # 20% of cap has earnings
    assert sec['Energy']['vs_real'] == pytest.approx(0.1 - 0.026)
    assert out['premia']['vs_real'] == pytest.approx(out['stocks']['earnings_yield'] - 0.026)


def test_model_read_prefers_spy_then_universe():
    rows = [_row('A', 100.0, _fv_effective=20.0, price=10.0),
            _row('B', 100.0, _fv_effective=None, price=10.0)]
    assert model_read(rows, {'funds': {'SPY': {'status': 'ok', 'mos': -0.3}}}) == \
        {'source': 'SPY', 'mos': -0.3}
    m = model_read(rows, None)
    assert m['source'] == 'universe'
    assert m['mos'] == pytest.approx(1 - 1 / 1.5)   # V = (2 + 1) / 2


def test_payload_written_and_degraded_suppressed(tmp_path):
    from scripts import report_html
    rows = [_row(f'T{i}', 100.0, pe=20.0) for i in range(6)]
    p = report_html._build_bond_payload(rows, str(tmp_path), date(2026, 10, 26), _sidecar())
    assert p['stocks']['n'] == 6
    saved = json.loads((tmp_path / 'bonds_2026-10-26.json').read_text(encoding='utf-8'))
    assert saved['date'] == '2026-10-26' and saved['premia'] == pytest.approx(p['premia'])
    assert report_html._build_bond_payload(
        rows, str(tmp_path), date(2026, 10, 26), _sidecar(),
        run_provenance={'coverage': {'degraded': True}}) is None
