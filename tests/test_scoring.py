# tests/test_scoring.py
import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.scoring import (
    _score_linear, compute_continuous_scores, apply_composite_rating_override,
    rating_from_composite, _mc_confidence_label, SCORING_GATES,
    SCREENING_GATES, gate_metadata, score_and_rate,
    apply_screening_matrix,
)


# ---------------------------------------------------------------------------
# _score_linear
# ---------------------------------------------------------------------------

class TestScoreLinear:
    def test_at_best_returns_100(self):
        """Value at best endpoint → 100."""
        assert _score_linear(0.40, -0.20, 0.40) == pytest.approx(100.0)

    def test_at_worst_returns_0(self):
        """Value at worst endpoint → 0."""
        assert _score_linear(-0.20, -0.20, 0.40) == pytest.approx(0.0)

    def test_midpoint_returns_50(self):
        """Value at midpoint → 50."""
        assert _score_linear(0.10, -0.20, 0.40) == pytest.approx(50.0)

    def test_below_worst_clamps_to_0(self):
        """Value below worst → clamped to 0."""
        assert _score_linear(-0.50, -0.20, 0.40) == pytest.approx(0.0)

    def test_above_best_clamps_to_100(self):
        """Value above best → clamped to 100."""
        assert _score_linear(0.80, -0.20, 0.40) == pytest.approx(100.0)

    def test_none_returns_none(self):
        """None value → None."""
        assert _score_linear(None, 0, 100) is None

    def test_inverted_scale(self):
        """When best < worst (lower is better), scoring inverts correctly."""
        # Price/FV: worst=1.5, best=0.7 → lower is better
        assert _score_linear(0.7, 1.5, 0.7) == pytest.approx(100.0)
        assert _score_linear(1.5, 1.5, 0.7) == pytest.approx(0.0)
        assert _score_linear(1.1, 1.5, 0.7) == pytest.approx(50.0)

    def test_equal_worst_best_returns_50(self):
        """When worst == best → 50."""
        assert _score_linear(5, 5, 5) == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# _mc_confidence_label
# ---------------------------------------------------------------------------

class TestMCConfidenceLabel:
    def test_low_cv_is_high(self):
        assert _mc_confidence_label(0.15) == 'HIGH (15%)'

    def test_medium_cv(self):
        assert _mc_confidence_label(0.30) == 'MEDIUM (30%)'

    def test_high_cv_is_low(self):
        assert _mc_confidence_label(0.50) == 'LOW (50%)'

    def test_boundary_020(self):
        assert _mc_confidence_label(0.20) == 'MEDIUM (20%)'

    def test_boundary_040(self):
        assert _mc_confidence_label(0.40) == 'LOW (40%)'


# ---------------------------------------------------------------------------
# compute_continuous_scores
# ---------------------------------------------------------------------------

class TestContinuousScoring:
    def _make_row(self, **kwargs):
        """Create a minimal result row with defaults."""
        defaults = {
            'ticker': 'TEST',
            'mos': 0.20,
            'ms_pfv': 1.0,
            'piotroski': 7,
            'cash_conv': 1.0,
            'accruals': 0.02,
            'spread': 0.15,
            'analyst_ltg': 0.10,
            'margin_trend': 0.02,
            'surprise_avg': 0.05,
            '_price_fv': 0.9,
            'mc_cv': 0.15,
        }
        defaults.update(kwargs)
        return defaults

    def test_produces_composite_score(self):
        """Should produce _composite_score for each row."""
        rows = [self._make_row(ticker='A'), self._make_row(ticker='B')]
        compute_continuous_scores(rows)
        for r in rows:
            assert '_composite_score' in r
            assert r['_composite_score'] is not None
            assert 0 <= r['_composite_score'] <= 100

    def test_produces_category_scores(self):
        """Should produce category scores (valuation, quality, moat, growth)."""
        rows = [self._make_row()]
        compute_continuous_scores(rows)
        r = rows[0]
        for cat in ('_score_valuation', '_score_quality', '_score_moat', '_score_growth'):
            assert cat in r
            assert 0 <= r[cat] <= 100

    def test_produces_per_gate_scores(self):
        """Should produce per-gate score fields."""
        rows = [self._make_row()]
        compute_continuous_scores(rows)
        r = rows[0]
        assert '_score_mos' in r
        assert '_score_piotroski' in r
        assert '_score_accruals' in r

    def test_higher_quality_gives_higher_score(self):
        """Better metrics → higher composite score."""
        good = self._make_row(
            ticker='GOOD', mos=0.35, piotroski=9, cash_conv=1.2,
            spread=0.25, analyst_ltg=0.15, _price_fv=0.7)
        poor = self._make_row(
            ticker='POOR', mos=-0.10, piotroski=2, cash_conv=0.3,
            spread=0.01, analyst_ltg=0.02, _price_fv=1.4)
        rows = [good, poor]
        compute_continuous_scores(rows)
        assert rows[0]['_composite_score'] > rows[1]['_composite_score']

    def test_mc_cv_penalty_applied(self):
        """High MC CV should penalize the composite score."""
        # Use 3+ rows with varied spread/analyst_ltg so percentile-ranking
        # doesn't create asymmetry between the two target rows
        low_cv = self._make_row(ticker='STABLE', mc_cv=0.15,
                                spread=0.15, analyst_ltg=0.10)
        high_cv = self._make_row(ticker='VOLATILE', mc_cv=0.50,
                                 spread=0.15, analyst_ltg=0.10)
        # Filler row to stabilize percentile rankings
        filler = self._make_row(ticker='FILLER', mc_cv=0.20,
                                spread=0.05, analyst_ltg=0.05)
        rows = [low_cv, high_cv, filler]
        compute_continuous_scores(rows)
        # STABLE and VOLATILE have same absolute metrics but different CV
        # The high-CV row should be penalized
        stable_score = rows[0]['_composite_score']
        volatile_score = rows[1]['_composite_score']
        assert stable_score > volatile_score

    def test_missing_fields_handled_gracefully(self):
        """Missing metric fields should not crash scoring."""
        row = {'ticker': 'SPARSE', 'mos': 0.10}
        rows = [row]
        compute_continuous_scores(rows)
        assert '_composite_score' in row

    def test_pctile_cleaned_up(self):
        """Temporary _pctile dict should be removed after scoring."""
        rows = [self._make_row(ticker='A'), self._make_row(ticker='B')]
        compute_continuous_scores(rows)
        for r in rows:
            assert '_pctile' not in r

    def test_tied_relative_values_receive_same_score(self):
        """Identical sector-relative raw values should not score differently."""
        rows = [
            self._make_row(ticker='A', sector='Tech', accruals=0.02,
                           gross_margin_avg_5y=0.50),
            self._make_row(ticker='B', sector='Tech', accruals=0.02,
                           gross_margin_avg_5y=0.50),
            self._make_row(ticker='C', sector='Tech', accruals=0.10,
                           gross_margin_avg_5y=0.30),
            self._make_row(ticker='D', sector='Tech', accruals=-0.02,
                           gross_margin_avg_5y=0.70),
            self._make_row(ticker='E', sector='Tech', accruals=0.05,
                           gross_margin_avg_5y=0.40),
        ]
        compute_continuous_scores(rows)
        assert rows[0]['_score_accruals'] == rows[1]['_score_accruals']
        assert rows[0]['_score_gross_margin'] == rows[1]['_score_gross_margin']


# ---------------------------------------------------------------------------
# rating_from_composite + apply_composite_rating_override
# ---------------------------------------------------------------------------

class TestRatingFromComposite:
    def test_buy_threshold(self):
        assert rating_from_composite(60) == 'BUY'
        assert rating_from_composite(75) == 'BUY'

    def test_lean_buy_threshold(self):
        assert rating_from_composite(43) == 'LEAN BUY'
        assert rating_from_composite(59.9) == 'LEAN BUY'

    def test_hold_threshold(self):
        assert rating_from_composite(29) == 'HOLD'
        assert rating_from_composite(42.9) == 'HOLD'

    def test_pass_threshold(self):
        assert rating_from_composite(0) == 'PASS'
        assert rating_from_composite(28.9) == 'PASS'

    def test_none_composite(self):
        assert rating_from_composite(None) is None

    def test_custom_thresholds_via_params(self):
        params = {'rating_threshold_buy': 70, 'rating_threshold_lean': 50,
                  'rating_threshold_pass': 25}
        assert rating_from_composite(65, params) == 'LEAN BUY'
        assert rating_from_composite(70, params) == 'BUY'
        assert rating_from_composite(20, params) == 'PASS'


class TestApplyCompositeRatingOverride:
    def test_buy_from_high_score(self):
        rows = [{'rating': None, '_composite_score': 70}]
        apply_composite_rating_override(rows)
        assert rows[0]['rating'] == 'BUY'

    def test_lean_buy_from_medium_score(self):
        rows = [{'rating': None, '_composite_score': 50}]
        apply_composite_rating_override(rows)
        assert rows[0]['rating'] == 'LEAN BUY'

    def test_hold_from_low_score(self):
        rows = [{'rating': None, '_composite_score': 35}]
        apply_composite_rating_override(rows)
        assert rows[0]['rating'] == 'HOLD'

    def test_pass_from_very_low_score(self):
        rows = [{'rating': None, '_composite_score': 20}]
        apply_composite_rating_override(rows)
        assert rows[0]['rating'] == 'PASS'

    def test_none_composite_leaves_rating_unchanged(self):
        rows = [{'rating': 'HOLD', '_composite_score': None}]
        apply_composite_rating_override(rows)
        assert rows[0]['rating'] == 'HOLD'

    def test_overwrites_existing_rating(self):
        rows = [{'rating': 'PASS', '_composite_score': 70}]
        apply_composite_rating_override(rows)
        assert rows[0]['rating'] == 'BUY'


class TestCanonicalScoreAndRate:
    def _row(self, **kwargs):
        row = {
            'ticker': 'CAP',
            'price': 120.0,
            'dcf_fv': 100.0,
            'mos': -0.20,
            'pfcf': 10,
            'int_cov': 20,
            'accruals': 0.01,
            'shareholder_yield': 0.05,
            'insider_pct': 0.10,
            'share_buyback_rate': 0.03,
            'roic_cv': 0.10,
            'spread': 0.20,
            'gross_margin_avg_5y': 0.70,
            'fundamental_growth': 0.10,
            'gross_margin_trend': 0.02,
            'roe': 0.30,
            'nd_ebitda': 0.0,
            'cash_conv': 1.2,
            'rev_cagr_10y': 0.08,
            'sbc': 0.0,
            'revenue': 100.0,
            'fcf': 25.0,
            'pb': 2.0,
            'fcf_cagr_5y': 0.10,
            'shares_cagr_5y': -0.02,
            'piotroski': 9,
            'roic_by_year': {2020: 0.10, 2024: 0.16},
            'epv_fv': 120.0,
            'rim_mos': 0.20,
            'mc_cv': 0.10,
            'rating': None,
        }
        row.update(kwargs)
        return row

    def test_preserves_score_rating_and_applies_critical_cap(self):
        rows = [self._row()]
        score_and_rate(rows)
        assert rows[0]['_rating_from_score'] == 'BUY'
        assert rows[0]['rating_raw'] == 'BUY'
        assert rows[0]['_rating_cap'] == 'PASS'
        assert rows[0]['rating'] == 'PASS'
        assert any('price/fair value' in r for r in rows[0]['_rating_cap_reasons'])

    def test_derives_fields_for_replay_style_rows(self):
        rows = [self._row(price=80.0, dcf_fv=100.0, epv_fv=110.0,
                          roic_by_year={2021: 0.08, 2023: 0.12})]
        score_and_rate(rows)
        assert rows[0]['_price_fv'] == pytest.approx(0.8)
        assert rows[0]['fcf_margin'] == pytest.approx(0.25)
        assert rows[0]['sbc_pct_rev'] == pytest.approx(0.0)
        assert rows[0]['epv_floor_ratio'] == pytest.approx(1.375)
        assert rows[0]['roic_trend_slope'] == pytest.approx(0.04)

    def test_gate_metadata_matches_screening_gate_count(self):
        meta = gate_metadata()
        assert len(meta['gates']) == len(SCREENING_GATES)
        keys = {g['key'] for g in meta['gates']}
        assert '_gate_roic_trend' in keys
        assert '_gate_epv_floor' in keys
        assert '_gate_rim_mos' in keys


# ---------------------------------------------------------------------------
# apply_screening_matrix — binary gate edge cases
# ---------------------------------------------------------------------------

class TestScreeningGateEdgeCases:
    """Regression tests for binary gates that previously mishandled negatives."""

    def _row(self, **kwargs):
        defaults = {'ticker': 'EDGE', 'price': 100.0}
        defaults.update(kwargs)
        return defaults

    def test_price_fv_gate_fails_when_dcf_fv_negative(self):
        """Negative DCF fair value must not pass the Price/FV gate.

        A distressed company with negative implied equity value previously
        passed because price / -fv < 1.0 evaluated to True.
        """
        rows = [self._row(price=50.0, dcf_fv=-25.0)]
        apply_screening_matrix(rows)
        assert rows[0]['_gp_price_fv'] is None
        assert rows[0]['_gate_price_fv'] is None

    def test_price_fv_gate_fails_when_dcf_fv_zero(self):
        """Zero fair value must short-circuit to N/A, not div-by-zero."""
        rows = [self._row(price=50.0, dcf_fv=0.0)]
        apply_screening_matrix(rows)
        assert rows[0]['_gp_price_fv'] is None

    def test_price_fv_gate_passes_when_undervalued(self):
        rows = [self._row(price=50.0, dcf_fv=100.0)]
        apply_screening_matrix(rows)
        assert rows[0]['_gp_price_fv'] is True
        assert rows[0]['_gate_price_fv'] == pytest.approx(0.5)

    def test_price_fv_gate_fails_when_overvalued(self):
        rows = [self._row(price=150.0, dcf_fv=100.0)]
        apply_screening_matrix(rows)
        assert rows[0]['_gp_price_fv'] is False
        assert rows[0]['_gate_price_fv'] == pytest.approx(1.5)

    def test_price_book_gate_fails_when_pb_negative(self):
        """Negative book value (e.g. heavy buybacks) must not pass the P/B gate.

        Previously v <= 5.0 returned True for any negative v.
        """
        rows = [self._row(pb=-1.5)]
        apply_screening_matrix(rows)
        assert rows[0]['_gp_price_book'] is False
        assert rows[0]['_gate_price_book'] == pytest.approx(-1.5)

    def test_price_book_gate_passes_when_pb_in_range(self):
        rows = [self._row(pb=2.5)]
        apply_screening_matrix(rows)
        assert rows[0]['_gp_price_book'] is True

    def test_price_book_gate_fails_when_pb_above_threshold(self):
        rows = [self._row(pb=8.0)]
        apply_screening_matrix(rows)
        assert rows[0]['_gp_price_book'] is False

    def test_negative_pb_does_not_clamp_continuous_score_to_100(self):
        """Negative P/B previously hit _score_linear(-3, 15, 0.5) → clamp 100.

        Now the score function rejects v <= 0 and returns None, which the
        aggregator treats as a worst-case 0 (consistent with other missing
        gates), so a negative-book company can no longer outscore a healthy one.
        """
        rows = [self._row(pb=-3.0), self._row(pb=2.0, ticker='OK')]
        compute_continuous_scores(rows)
        assert rows[0]['_score_price_book'] == 0.0
        assert rows[1]['_score_price_book'] > rows[0]['_score_price_book']

    def test_spread_gate_threshold_is_seven_percent(self):
        """Pin the actual threshold so the display label/tooltip stay aligned."""
        below = [self._row(spread=0.06)]
        above = [self._row(spread=0.08)]
        apply_screening_matrix(below)
        apply_screening_matrix(above)
        assert below[0]['_gp_spread_>_5%'] is False
        assert above[0]['_gp_spread_>_5%'] is True
