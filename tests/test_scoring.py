# tests/test_scoring.py
import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.scoring import (
    _score_linear, compute_continuous_scores, apply_composite_rating_override,
    rating_from_composite, _mc_confidence_label, SCORING_GATES,
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
