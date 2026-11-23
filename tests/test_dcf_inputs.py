"""dcf_inputs: the forward DCF records every growth-blend input, so the blend
can be re-derived or re-weighted (with or without analysts) from the row."""
import pandas as pd
import pytest

from scripts import analyze_stock as az
from tests.test_monte_carlo import _dcf_yf_data


def _with_analysts(yf, ltg=None, plus1y=0.20, rev_g=0.10, earn_g=0.15):
    yf = dict(yf)
    rows = {}
    if ltg is not None:
        rows['LTG'] = {'stockTrend': ltg}
    rows['+1y'] = {'stockTrend': plus1y}
    yf['growth_estimates'] = pd.DataFrame.from_dict(rows, orient='index')
    yf['info'] = {**yf['info'], 'revenueGrowth': rev_g, 'earningsGrowth': earn_g}
    return yf


def _rederive(d):
    """The blend recomputed from the recorded signals and weights."""
    s, w = d['signals'], d['weights']
    return sum(s[k] * w[k] for k in s) / sum(w.values())


def test_records_signals_and_rederives_growth():
    _, _, g, diag, _ = az.run_forward_dcf(_with_analysts(_dcf_yf_data()), wacc=0.10)
    d = diag['dcf_inputs']
    assert set(d['signals']) >= {'fcf_cagr', 'rev_cagr', 'analyst_st', 'analyst_lt', 'earnings_g'}
    assert d['blend'] == pytest.approx(_rederive(d))
    raw = d['blend'] + d['margin_adj'] + d['surprise_adj']
    assert d['growth'] == g == pytest.approx(min(d['growth_cap'], max(-0.15, raw)))
    # Analyst signals enter after the haircut; raw values are kept beside them.
    assert d['analyst_raw'] == {'st': 0.10, 'lt': 0.20, 'earnings': 0.15}
    assert d['signals']['analyst_lt'] == pytest.approx(0.20 * d['analyst_haircut'])
    assert d['base_fcf'] > 0 and d['wacc'] == 0.10 and d['growth_sigma'] > 0
    assert d['fcf_span'] == ['2021-12-31', '2024-12-31']


def test_long_term_horizon_says_which_row_was_used():
    _, _, _, diag, _ = az.run_forward_dcf(_with_analysts(_dcf_yf_data()), wacc=0.10)
    assert diag['dcf_inputs']['analyst_lt_horizon'] == '+1y'      # LTG missing
    _, _, _, diag, _ = az.run_forward_dcf(_with_analysts(_dcf_yf_data(), ltg=0.08), wacc=0.10)
    assert diag['dcf_inputs']['analyst_lt_horizon'] == 'LTG'
    assert diag['dcf_inputs']['analyst_raw']['lt'] == 0.08


def test_without_analysts_nothing_analyst_is_recorded():
    _, _, _, diag, _ = az.run_forward_dcf(_dcf_yf_data(), wacc=0.10)
    d = diag['dcf_inputs']
    assert not {'analyst_st', 'analyst_lt', 'earnings_g'} & set(d['signals'])
    assert d['analyst_lt_horizon'] is None


def test_wrapper_keeps_its_contract():
    yf = _with_analysts(_dcf_yf_data(), ltg=0.08)
    assert az._get_analyst_lt_growth(yf) == 0.08
    assert az._get_analyst_lt_growth({}) is None
