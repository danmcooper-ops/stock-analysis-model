"""Integrity of the real GATES list and the metadata it exports.

These are the coupling tests: the report renders its columns straight from
`gate_metadata()`, so a gate added to GATES without display metadata would
ship a column labelled with a raw field name.
"""

from data import currency_meta
from scripts.config import PILLAR_ORDER, PILLAR_WEIGHTS
from scripts.scoring import (
    GATES, _GATE_DISPLAY, _GATE_TIPS, _PILLAR_TIPS, _UI_TIPS, _gate_key,
    _gate_short, _gp_key, _score_key, gate_metadata,
)


def test_every_gate_belongs_to_a_declared_pillar():
    for g in GATES:
        assert g.category in PILLAR_ORDER, g.name


def test_every_pillar_has_at_least_one_gate():
    covered = {g.category for g in GATES}
    for pillar in PILLAR_ORDER:
        assert pillar in covered, pillar


def test_gate_names_follow_the_category_prefix_convention():
    # _gate_short() slices on ': ', so a name without it would collide.
    for g in GATES:
        assert ': ' in g.name, g.name


def test_gate_field_keys_are_unique():
    shorts = [_gate_short(g.name) for g in GATES]
    assert len(shorts) == len(set(shorts)), 'duplicate gate short key'


def test_every_gate_has_display_metadata():
    for g in GATES:
        assert g.field in _GATE_DISPLAY, (
            'gate %r has no _GATE_DISPLAY entry; the report would render a '
            'column labelled with the raw field name' % g.name)


def test_display_metadata_has_no_orphans():
    fields = {g.field for g in GATES}
    for field in _GATE_DISPLAY:
        assert field in fields, 'stale _GATE_DISPLAY entry: %s' % field


def test_every_gate_has_a_tooltip():
    for g in GATES:
        assert g.field in _GATE_TIPS, (
            'gate %r has no tooltip; the report would show a bare column with '
            'no way to find out what it measures' % g.name)


def test_tooltips_have_no_orphans():
    fields = {g.field for g in GATES}
    for field in _GATE_TIPS:
        assert field in fields, 'stale tooltip for retired gate: %s' % field


def test_every_pillar_has_a_tooltip():
    for pillar in PILLAR_ORDER:
        assert _PILLAR_TIPS.get(pillar), pillar
    for pillar in _PILLAR_TIPS:
        assert pillar in PILLAR_ORDER, 'stale pillar tooltip: %s' % pillar


def test_tooltips_say_more_than_the_label():
    """A tooltip that only restates its column heading is worse than none.

    Cheap proxy for that: real explanations run to a sentence or more.
    """
    for g in GATES:
        tip = _GATE_TIPS[g.field]
        label = _GATE_DISPLAY[g.field][0]
        assert len(tip) > 60, '%s: tooltip too short to explain anything' % g.name
        assert tip.strip().lower() != label.strip().lower(), g.name


def test_ui_tooltips_cover_the_keys_the_template_asks_for():
    """The template resolves these by name; a missing key renders an empty
    tooltip attribute and the element silently loses its explanation."""
    for key in ('composite', 'rating', 'gates', 'coverage', 'regime', 'na',
                'spark', 'policy_rate', 'inflation_pct', 'name'):
        assert _UI_TIPS.get(key), 'missing UI tooltip: %s' % key


def test_metadata_carries_tooltips_to_the_report():
    gm = gate_metadata()
    for meta in gm['gates']:
        assert meta['tip'], meta['name']
    for cat in gm['categories']:
        assert cat['tip'], cat['name']
    assert gm['ui']


def test_pillar_weights_sum_to_one():
    assert abs(sum(PILLAR_WEIGHTS.values()) - 1.0) < 1e-9


def test_metadata_shape_matches_gates():
    gm = gate_metadata()
    assert len(gm['gates']) == len(GATES)
    assert len(gm['categories']) == len(PILLAR_ORDER)
    for meta, g in zip(gm['gates'], GATES):
        assert meta['key'] == _gate_key(g.name)
        assert meta['gpKey'] == _gp_key(g.name)
        assert meta['scoreKey'] == _score_key(g.name)
        assert meta['label'] and meta['fmt']


def test_metadata_category_counts_are_right():
    gm = gate_metadata()
    for c in gm['categories']:
        expected = sum(1 for g in GATES if g.category == c['name'])
        assert c['count'] == expected, c['name']


def test_relative_gates_declare_a_direction():
    for g in GATES:
        if g.relative_mode:
            assert isinstance(g.higher_better, bool), g.name


def test_test_fn_returns_none_for_missing_values():
    """Every gate must map a missing value to None, never to False.

    Returning False would render a red 'fail' cell for data that was simply
    never fetched, which is a materially different claim.
    """
    for g in GATES:
        assert g.test_fn(None, {}) is None, g.name


def test_absolute_score_fn_returns_none_for_missing_values():
    """Absolute gates must not fabricate a score from a missing value.

    Scoped to absolute gates deliberately. A relative gate's score_fn is
    `lambda v, r, pct: pct` — it ignores the raw value by construction, since
    the percentile was computed in an earlier pass. That is not a hole:
    compute_continuous_scores short-circuits a None value to 0.0 and never
    reaches score_fn, so the pct path is unreachable with missing data.
    """
    for g in GATES:
        if g.relative_mode:
            continue
        assert g.score_fn(None, {}, None) is None, g.name


def test_applicability_predicates_run_on_a_bare_row():
    """Predicates must tolerate a row missing every optional key."""
    for g in GATES:
        if g.applicable is None:
            continue
        for code in ('USD', 'JPY', 'HKD', 'SGD'):
            g.applicable({'code': code})   # must not raise


def test_base_currency_has_no_applicable_carry_gates():
    row = {'code': currency_meta.BASE_CURRENCY, 'reer_months': 400}
    carry = [g for g in GATES if g.category == 'Carry']
    assert carry
    assert all(not g.applicable(row) for g in carry)


def test_peg_has_no_applicable_valuation_or_momentum_market_gates():
    row = {'code': 'HKD', 'reer_months': 400}
    val = [g for g in GATES if g.category == 'Valuation']
    assert val and all(not g.applicable(row) for g in val)

    mom = [g for g in GATES if g.category == 'Momentum']
    assert mom and all(not g.applicable(row) for g in mom)


def test_floating_currency_has_every_gate_applicable():
    row = {'code': 'JPY', 'reer_months': 400}
    for g in GATES:
        assert g.applicable is None or g.applicable(row), g.name
