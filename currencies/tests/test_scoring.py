"""Scoring primitives, the N/A policy, and rating caps.

The N/A tests are the important ones. The whole model rests on treating
"structurally inapplicable" and "missing data" differently, and the two are
easy to conflate in a refactor — with opposite failure modes: pegs sweeping
the risk pillars, or pegs punished for a question that does not apply.
"""

import pytest

from scripts import scoring
from scripts.scoring import (
    Gate, RATING_RANK, _cap_rating, _gate_short, _ranked_percentiles,
    _score_linear, apply_rating_caps, compute_continuous_scores,
    gate_metadata, rating_from_composite, score_and_rate,
)


# --------------------------------------------------------------- primitives

def test_score_linear_endpoints_and_midpoint():
    assert _score_linear(0.0, 0.0, 10.0) == 0.0
    assert _score_linear(10.0, 0.0, 10.0) == 100.0
    assert _score_linear(5.0, 0.0, 10.0) == 50.0


def test_score_linear_clamps_both_tails():
    # Clamping is the model's only outlier handling; there is no winsorization.
    assert _score_linear(-99.0, 0.0, 10.0) == 0.0
    assert _score_linear(999.0, 0.0, 10.0) == 100.0


def test_score_linear_reversed_range_scores_lower_is_better():
    # Direction is encoded purely by argument order.
    assert _score_linear(0.03, 0.15, 0.03) == 100.0
    assert _score_linear(0.15, 0.15, 0.03) == 0.0
    assert _score_linear(0.09, 0.15, 0.03) == pytest.approx(50.0)


def test_score_linear_none_and_degenerate():
    assert _score_linear(None, 0.0, 1.0) is None
    assert _score_linear(5.0, 2.0, 2.0) == 50.0


def test_ranked_percentiles_ties_get_equal_scores():
    out = _ranked_percentiles([(0, 1.0), (1, 5.0), (2, 5.0), (3, 9.0)])
    assert out[1] == out[2]
    assert out[0] == 0.0
    assert out[3] == 100.0


def test_ranked_percentiles_direction():
    hi = _ranked_percentiles([(0, 1.0), (1, 2.0)], higher_better=True)
    lo = _ranked_percentiles([(0, 1.0), (1, 2.0)], higher_better=False)
    assert hi[1] == 100.0 and hi[0] == 0.0
    assert lo[1] == 0.0 and lo[0] == 100.0


def test_ranked_percentiles_singleton_is_neutral():
    assert _ranked_percentiles([(0, 3.0)]) == {0: 50.0}
    assert _ranked_percentiles([]) == {}


def test_gate_short_normalizes_separators():
    assert _gate_short('Carry: Carry/Vol') == 'carry_vol'
    assert _gate_short('Momentum: 12m-1m') == '12m_1m'
    assert _gate_short('Valuation: REER vs 10y') == 'reer_vs_10y'


# ------------------------------------------------------------- the NA policy

def _mini_gates():
    """Two gates in one pillar: one always applicable, one gated on a flag."""
    return [
        Gate('Valuation: Always', 'a',
             lambda v, r: v > 0 if v is not None else None,
             lambda v, r, p: _score_linear(v, 0.0, 10.0)),
        Gate('Valuation: Sometimes', 'b',
             lambda v, r: v > 0 if v is not None else None,
             lambda v, r, p: _score_linear(v, 0.0, 10.0),
             applicable=lambda r: r.get('ok', True)),
    ]


@pytest.fixture
def patched_gates(monkeypatch):
    monkeypatch.setattr(scoring, 'GATES', _mini_gates())


def test_inapplicable_gate_leaves_numerator_and_denominator(patched_gates):
    """A gate that cannot describe the row must not drag its pillar down."""
    rows = [{'code': 'X', 'a': 10.0, 'b': None, 'ok': False}]
    score_and_rate(rows)
    r = rows[0]
    assert r['_gp_sometimes'] is None
    assert r['_score_sometimes'] is None
    assert r['_gates_inapplicable'] == 1
    assert r['_gates_applicable'] == 1
    # Pillar renormalizes over the one applicable gate: full marks, not half.
    assert r['_score_pillar_valuation'] == 100.0


def test_missing_data_scores_zero_and_stays_in_denominator(patched_gates):
    """Same row shape, but the gate applies — sparse data stays penalized."""
    rows = [{'code': 'X', 'a': 10.0, 'b': None, 'ok': True}]
    score_and_rate(rows)
    r = rows[0]
    assert r['_gp_sometimes'] is None       # still renders N/A
    assert r['_score_sometimes'] == 0.0     # but scores worst
    assert r['_gates_inapplicable'] == 0
    assert r['_gates_applicable'] == 2
    assert r['_score_pillar_valuation'] == 50.0


def test_coverage_counts_only_applicable_gates(patched_gates):
    rows = [{'code': 'X', 'a': 10.0, 'b': None, 'ok': False}]
    score_and_rate(rows)
    # One applicable gate, one covered -> 100%, not 50%.
    assert rows[0]['_data_coverage_score'] == 100.0


def test_gate_weight_scales_within_pillar(monkeypatch):
    monkeypatch.setattr(scoring, 'GATES', [
        Gate('Valuation: Heavy', 'a', lambda v, r: True,
             lambda v, r, p: 100.0, weight=3.0),
        Gate('Valuation: Light', 'b', lambda v, r: True,
             lambda v, r, p: 0.0, weight=1.0),
    ])
    rows = [{'code': 'X', 'a': 1.0, 'b': 1.0}]
    score_and_rate(rows)
    assert rows[0]['_score_pillar_valuation'] == 75.0


def test_relative_gate_excludes_inapplicable_rows_from_pool(monkeypatch):
    monkeypatch.setattr(scoring, 'GATES', [
        Gate('Carry: Rel', 'v', lambda v, r: True, lambda v, r, p: p,
             relative_mode='global', higher_better=True,
             applicable=lambda r: r.get('ok', True)),
    ])
    rows = [{'code': 'A', 'v': 1.0, 'ok': True},
            {'code': 'B', 'v': 2.0, 'ok': True},
            {'code': 'C', 'v': 99.0, 'ok': False}]
    compute_continuous_scores(rows)
    # C is out of the pool entirely, so A and B span the full 0-100 range
    # instead of being squashed against an outlier that isn't competing.
    assert rows[0]['_score_rel'] == 0.0
    assert rows[1]['_score_rel'] == 100.0
    assert rows[2]['_score_rel'] is None


# ------------------------------------------------------------------ ratings

def test_rating_bands():
    assert rating_from_composite(90.0) == 'LONG'
    assert rating_from_composite(57.0) == 'LONG'
    assert rating_from_composite(56.9) == 'LEAN LONG'
    assert rating_from_composite(39.0) == 'LEAN LONG'
    assert rating_from_composite(25.0) == 'NEUTRAL'
    assert rating_from_composite(24.9) == 'AVOID'
    assert rating_from_composite(None) == 'UNRATED'


def test_cap_takes_ordinal_minimum():
    assert _cap_rating('LONG', 'NEUTRAL') == 'NEUTRAL'
    assert _cap_rating('AVOID', 'LONG') == 'AVOID'   # a cap never promotes
    assert _cap_rating('UNRATED', 'NEUTRAL') == 'UNRATED'
    assert RATING_RANK['LONG'] > RATING_RANK['NEUTRAL'] > RATING_RANK['AVOID']


def test_base_currency_is_capped_neutral():
    rows = [{'code': 'USD', '_composite_score': 95.0}]
    apply_rating_caps(rows)
    assert rows[0]['rating_raw'] == 'LONG'
    assert rows[0]['rating'] == 'NEUTRAL'
    assert any('Base currency' in x for x in rows[0]['_rating_cap_reasons'])


def test_peg_is_capped_neutral():
    rows = [{'code': 'HKD', '_composite_score': 95.0}]
    apply_rating_caps(rows)
    assert rows[0]['rating'] == 'NEUTRAL'


def test_hyperinflation_caps_to_avoid():
    rows = [{'code': 'TRY', '_composite_score': 95.0, 'inflation_pct': 80.0}]
    apply_rating_caps(rows)
    assert rows[0]['rating'] == 'AVOID'


def test_caps_fail_open_on_missing_metrics():
    # No inflation, no coverage, no REER history recorded -> no cap fires.
    rows = [{'code': 'JPY', '_composite_score': 80.0}]
    apply_rating_caps(rows)
    assert rows[0]['_rating_cap'] is None
    assert rows[0]['rating'] == 'LONG'


def test_low_coverage_caps_to_neutral():
    rows = [{'code': 'JPY', '_composite_score': 80.0,
             '_data_coverage_score': 10.0}]
    apply_rating_caps(rows)
    assert rows[0]['rating'] == 'NEUTRAL'


def test_short_reer_history_caps_to_neutral():
    rows = [{'code': 'JPY', '_composite_score': 80.0, 'reer_months': 12}]
    apply_rating_caps(rows)
    assert rows[0]['rating'] == 'NEUTRAL'


# ----------------------------------------------------------------- metadata

def test_purge_drops_retired_gate_fields(patched_gates):
    rows = [{'code': 'X', 'a': 5.0, 'b': 5.0,
             '_gate_retired': 1, '_gp_retired': True, '_score_retired': 50}]
    score_and_rate(rows)
    assert '_gate_retired' not in rows[0]
    assert '_gp_retired' not in rows[0]
    assert '_score_retired' not in rows[0]


def test_purge_keeps_pillar_scores(patched_gates):
    rows = [{'code': 'X', 'a': 5.0, 'b': 5.0}]
    score_and_rate(rows)
    assert rows[0]['_score_pillar_valuation'] is not None
