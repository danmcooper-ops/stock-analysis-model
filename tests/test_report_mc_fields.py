"""DCF Monte Carlo diagnostics on the company page (engine #597 fields):
the report's row context carries them, rounded, and the heavy ones ride in
the detail parts and app detail shards rather than the inline DATA blob."""
import math

from scripts.report_html import _extract_details_payload, _round_list, _round_map, _row_context

ROW = {
    'ticker': 'TEST', 'price': 120.0, 'dcf_fv': 130.0,
    'mc_price_pctile': 0.6512, 'mc_prob_above_price': 0.3488, 'implied_growth_z': 1.2345,
    'mc_driver_top': 'growth',
    'mc_drivers': {'growth': 0.61234, 'discount_rate': 0.28766, 'terminal_growth': 0.03, 'exit_multiple': 0.07},
    'mc_quantiles': [50.123 + i for i in range(21)],
}


def _ctx(row):
    return _row_context(row, {'gates': []}, set(), {}, {})


def test_row_context_carries_the_diagnostics_rounded():
    c = _ctx(ROW)
    assert c['mc_price_pctile'] == 0.6512 and c['mc_prob_above_price'] == 0.3488
    assert c['implied_growth_z'] == 1.2345 and c['mc_driver_top'] == 'growth'
    assert c['mc_drivers'] == {'growth': 0.612, 'discount_rate': 0.288, 'terminal_growth': 0.03,
                               'exit_multiple': 0.07}
    assert c['mc_quantiles'] == [round(50.123 + i, 2) for i in range(21)]


def test_rows_without_a_simulation_carry_none():
    c = _ctx({'ticker': 'NODCF', 'price': 10.0})
    for k in ('mc_price_pctile', 'mc_prob_above_price', 'implied_growth_z', 'mc_driver_top',
              'mc_drivers', 'mc_quantiles'):
        assert c[k] is None, k


def test_quantiles_and_drivers_move_to_the_detail_payload():
    records = [_ctx(ROW), _ctx({'ticker': 'NODCF', 'price': 10.0})]
    payload = _extract_details_payload(records)
    assert 'mc_quantiles' not in records[0] and 'mc_drivers' not in records[0]
    assert payload['TEST']['mc_quantiles'][0] == 50.12
    assert payload['TEST']['mc_drivers']['growth'] == 0.612
    # The small scalars stay inline for the page's first paint.
    assert records[0]['mc_price_pctile'] == 0.6512
    assert 'NODCF' not in payload


def test_round_helpers_refuse_partial_data():
    assert _round_list([1.234, 2.345], 1) == [1.2, 2.3]
    for bad in (None, [], [1.0, None], [1.0, math.nan], [True, 1.0], 'x'):
        assert _round_list(bad, 2) is None
    assert _round_map({'a': 0.1234, 'b': None, 'c': math.inf}, 2) == {'a': 0.12}
    assert _round_map({'b': None}, 2) is None and _round_map(None, 2) is None
