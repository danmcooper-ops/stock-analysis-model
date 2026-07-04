# tests/test_scoring.py
import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.scoring import (
    _score_linear, compute_continuous_scores, apply_composite_rating_override,
    rating_from_composite, _mc_confidence_label, SCORING_GATES,
    SCREENING_GATES, gate_metadata, score_and_rate,
    apply_screening_matrix, _rating_cap_for_row, apply_rating_caps,
    prepare_scoring_fields,
)


# ---------------------------------------------------------------------------
# prepare_scoring_fields — EDGAR FCF fallback (fix (a))
# ---------------------------------------------------------------------------

class TestEdgarFcfFallback:
    def test_fallback_populates_fcf_margin_and_pfcf_when_yfinance_missing(self):
        """No yfinance fcf but EDGAR-derived fcf_edgar present → fcf, fcf_margin,
        and pfcf are all populated from the EDGAR value."""
        r = {'sector': 'Technology', 'fcf': None, 'fcf_edgar': 300.0,
             'revenue': 1000.0, 'mcap': 6000.0, 'pfcf': None}
        prepare_scoring_fields([r])
        assert r['fcf'] == 300.0
        assert r['_fcf_source'] == 'edgar'
        assert r['fcf_margin'] == pytest.approx(0.30)
        assert r['pfcf'] == pytest.approx(20.0)  # 6000 / 300

    def test_existing_yfinance_fcf_not_overridden(self):
        """A yfinance fcf already present is kept; the EDGAR value is ignored."""
        r = {'sector': 'Technology', 'fcf': 250.0, 'fcf_edgar': 999.0,
             'revenue': 1000.0, 'mcap': 5000.0, 'pfcf': 20.0}
        prepare_scoring_fields([r])
        assert r['fcf'] == 250.0
        assert '_fcf_source' not in r
        assert r['fcf_margin'] == pytest.approx(0.25)
        assert r['pfcf'] == 20.0  # untouched

    def test_financial_services_excluded_from_fallback(self):
        """Banks/insurers/brokers: OCF-based FCF is not a valid proxy, so the
        fallback is skipped and FCF Margin stays N/A."""
        r = {'sector': 'Financial Services', 'fcf': None, 'fcf_edgar': 800.0,
             'revenue': 500.0, 'mcap': 4000.0, 'pfcf': None}
        prepare_scoring_fields([r])
        assert r['fcf'] is None
        assert r.get('_fcf_source') is None
        assert r['fcf_margin'] is None
        assert r['pfcf'] is None

    def test_negative_fcf_gives_negative_margin_but_no_pfcf(self):
        """Negative EDGAR FCF → a meaningful negative margin, but pfcf stays
        None (P/FCF is meaningless for negative FCF)."""
        r = {'sector': 'Industrials', 'fcf': None, 'fcf_edgar': -120.0,
             'revenue': 1000.0, 'mcap': 5000.0, 'pfcf': None}
        prepare_scoring_fields([r])
        assert r['fcf'] == -120.0
        assert r['fcf_margin'] == pytest.approx(-0.12)
        assert r['pfcf'] is None


# ---------------------------------------------------------------------------
# prepare_scoring_fields — effective fair value / blended P-FV & MoS (fix (b))
# ---------------------------------------------------------------------------

class TestEffectiveFairValue:
    def test_dcf_used_when_present(self):
        """DCF fair value present → it is the effective FV; Price/FV and MoS
        derive from it and the alt models are ignored."""
        r = {'price': 90.0, 'dcf_fv': 100.0,
             'epv_growth_fv': 50.0, 'rim_fv': 50.0, 'ddm_fv': 50.0}
        prepare_scoring_fields([r])
        assert r['_fv_source'] == 'dcf'
        assert r['_fv_effective'] == 100.0
        assert r['_price_fv'] == pytest.approx(0.90)
        assert r['mos'] == pytest.approx(0.10)

    def test_blend_median_when_dcf_absent(self):
        """No DCF but >=2 growth-inclusive models → median consensus FV drives
        Price/FV and MoS."""
        r = {'price': 120.0, 'dcf_fv': None,
             'epv_growth_fv': 100.0, 'rim_fv': 150.0, 'ddm_fv': 200.0}
        prepare_scoring_fields([r])
        assert r['_fv_source'] == 'blend'
        assert r['_fv_effective'] == pytest.approx(150.0)  # median of 100/150/200
        assert r['_price_fv'] == pytest.approx(0.80)
        assert r['mos'] == pytest.approx(0.20)

    def test_nav_and_bare_epv_are_excluded(self):
        """nav_fv (asset floor) and bare epv_fv (growth-agnostic) are NOT part
        of the consensus — a row carrying only those has no effective FV."""
        r = {'price': 50.0, 'dcf_fv': None,
             'nav_fv': 40.0, 'epv_fv': 45.0}
        prepare_scoring_fields([r])
        assert r['_fv_source'] is None
        assert r['_fv_effective'] is None
        assert r['_price_fv'] is None
        assert r['mos'] is None

    def test_single_model_is_insufficient(self):
        """A lone model is not a consensus → no effective FV (requires >=2)."""
        r = {'price': 50.0, 'dcf_fv': None, 'rim_fv': 60.0}
        prepare_scoring_fields([r])
        assert r['_fv_source'] is None
        assert r['_price_fv'] is None
        assert r['mos'] is None


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


# ---------------------------------------------------------------------------
# rating_from_composite + apply_composite_rating_override
# ---------------------------------------------------------------------------

class TestRatingFromComposite:
    def test_buy_threshold(self):
        assert rating_from_composite(57) == 'BUY'
        assert rating_from_composite(75) == 'BUY'

    def test_lean_buy_threshold(self):
        assert rating_from_composite(39) == 'LEAN BUY'
        assert rating_from_composite(56.9) == 'LEAN BUY'

    def test_hold_threshold(self):
        assert rating_from_composite(25) == 'HOLD'
        assert rating_from_composite(38.9) == 'HOLD'

    def test_pass_threshold(self):
        assert rating_from_composite(0) == 'PASS'
        assert rating_from_composite(24.9) == 'PASS'

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
            'operating_margin': 0.30,
            '_sector_median_opm': 0.15,   # margin_advantage = +15pp
            'rev_growth_vol': 0.06,       # steady top line
            'nd_ebitda': 0.0,
            'cash_conv': 1.2,
            'rev_cagr_10y': 0.08,
            'sbc': 0.0,
            'revenue': 100.0,
            'fcf': 25.0,
            'mcap': 300.0,                # fcf_yield = 8.3% > risk-free
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
        assert any('margin of safety' in r for r in rows[0]['_rating_cap_reasons'])

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
        assert '_gate_margin_advantage' in keys
        assert '_gate_epv_floor' in keys
        assert '_gate_fcf_yield' in keys
        assert '_gate_rev_volatility' in keys
        # 2026-07 rebalance additions
        assert '_gate_ebit_ev' in keys
        assert '_gate_incr_roic' in keys
        assert '_gate_margin_vs_hist' in keys
        assert '_gate_insider_buying' in keys
        # Retired gates must be gone
        assert '_gate_roic_trend' not in keys
        assert '_gate_rim_mos' not in keys
        assert '_gate_roe' not in keys
        assert '_gate_p_fcf' not in keys         # reciprocal of FCF Yield
        assert '_gate_buyback_rate' not in keys  # inside Shrhldr Yield
        assert '_gate_cash_conv' not in keys     # accruals inverted
        assert '_gate_gross_margin' not in keys  # merged into Margin Adv

    def test_gate_metadata_score_keys_are_written(self):
        """Every screening gate name must have a scoring twin — a name
        mismatch silently blanks that gate's Matrix score column."""
        scoring_names = {g.name for g in SCORING_GATES}
        for gate in SCREENING_GATES:
            assert gate.name in scoring_names, gate.name

    def test_gate_metadata_carries_weights(self):
        meta = gate_metadata()
        by_key = {g['key']: g for g in meta['gates']}
        assert by_key['_gate_mos']['weight'] == 2.0
        assert by_key['_gate_fv_dispersion']['weight'] == 1.0


# ---------------------------------------------------------------------------
# apply_screening_matrix — binary gate edge cases
# ---------------------------------------------------------------------------

class TestScreeningGateEdgeCases:
    """Regression tests for binary gates that previously mishandled negatives."""

    def _row(self, **kwargs):
        defaults = {'ticker': 'EDGE', 'price': 100.0}
        defaults.update(kwargs)
        return defaults

    def test_negative_dcf_fv_yields_no_bogus_mos(self):
        """A negative DCF fair value must not create a false bargain signal:
        with no alt models it leaves MoS / _price_fv as None, not a positive
        value from dividing by a negative FV."""
        rows = [self._row(price=50.0, dcf_fv=-25.0)]
        apply_screening_matrix(rows)
        assert rows[0]['mos'] is None
        assert rows[0]['_price_fv'] is None
        assert rows[0].get('_fv_source') is None

    def test_fv_dispersion_gate_passes_on_tight_agreement(self):
        """Tight model agreement → the FV Dispersion gate passes (MAD/median)."""
        rows = [self._row(dcf_fv=100.0, epv_growth_fv=110.0,
                          rim_fv=120.0, ddm_fv=130.0)]
        apply_screening_matrix(rows)
        # median=115; MAD=median(15,5,5,15)=10; 10/115 ≈ 0.087 ≤ 0.15
        assert rows[0]['_gate_fv_dispersion'] == pytest.approx(10 / 115.0)
        assert rows[0]['_gp_fv_dispersion'] is True

    def test_fv_dispersion_gate_fails_on_wide_spread(self):
        """Wide disagreement → the gate fails (value-trap risk: the fair value
        is not corroborated)."""
        rows = [self._row(dcf_fv=50.0, epv_growth_fv=100.0, rim_fv=200.0)]
        apply_screening_matrix(rows)
        # median=100; MAD=median(50,0,100)=50; 50/100 = 0.5 > 0.15
        assert rows[0]['_gate_fv_dispersion'] == pytest.approx(0.5)
        assert rows[0]['_gp_fv_dispersion'] is False

    def test_fv_dispersion_na_with_single_model(self):
        """A lone model can't form a spread → N/A."""
        rows = [self._row(rim_fv=100.0)]
        apply_screening_matrix(rows)
        assert rows[0]['fv_dispersion'] is None
        assert rows[0]['_gp_fv_dispersion'] is None

    def test_fv_dispersion_excludes_nav(self):
        """nav_fv is not part of the consensus, so a low NAV must not widen the
        measured dispersion."""
        rows = [self._row(dcf_fv=100.0, epv_growth_fv=105.0, nav_fv=20.0)]
        apply_screening_matrix(rows)
        # only dcf & epv_growth count: median=102.5, MAD=2.5, 2.5/102.5 ≈ 0.024
        assert rows[0]['_gate_fv_dispersion'] == pytest.approx(2.5 / 102.5)
        assert rows[0]['_gp_fv_dispersion'] is True

    def test_fv_dispersion_uses_preblend_dcf(self):
        """Dispersion must key off the pre-blend DCF so the DDM leg (blended
        into dcf_fv upstream) isn't double-counted."""
        # blended dcf_fv pulled toward ddm; preblend is the true DCF
        rows = [self._row(dcf_fv=90.0, _dcf_fv_preblend=100.0,
                          epv_growth_fv=105.0)]
        apply_screening_matrix(rows)
        # uses 100 & 105, not 90: median=102.5, MAD=2.5
        assert rows[0]['_gate_fv_dispersion'] == pytest.approx(2.5 / 102.5)

    # --- Overhaul gates: Margin Advantage / Rev Volatility / FCF Yield ---

    def test_margin_advantage_from_sector_median(self):
        """margin_advantage = operating margin − sector median; passes above
        +5pp."""
        rows = [self._row(operating_margin=0.28, _sector_median_opm=0.15)]
        apply_screening_matrix(rows)
        assert rows[0]['margin_advantage'] == pytest.approx(0.13)
        assert rows[0]['_gp_margin_advantage'] is True

    def test_margin_advantage_below_sector_fails(self):
        rows = [self._row(operating_margin=0.10, _sector_median_opm=0.15)]
        apply_screening_matrix(rows)
        assert rows[0]['margin_advantage'] == pytest.approx(-0.05)
        assert rows[0]['_gp_margin_advantage'] is False

    def test_rev_volatility_gate(self):
        """Low revenue-growth volatility passes; high fails."""
        steady = self._row(rev_growth_vol=0.06)
        lumpy = self._row(rev_growth_vol=0.30)
        apply_screening_matrix([steady, lumpy])
        assert steady['_gp_rev_volatility'] is True
        assert lumpy['_gp_rev_volatility'] is False

    def test_fcf_yield_gated_against_risk_free(self):
        """FCF yield passes only when it beats the row's risk-free rate."""
        beats = self._row(fcf=8.0, mcap=100.0, _risk_free_rate=0.045)   # 8% > 4.5%
        trails = self._row(fcf=3.0, mcap=100.0, _risk_free_rate=0.045)  # 3% < 4.5%
        apply_screening_matrix([beats, trails])
        assert beats['fcf_yield'] == pytest.approx(0.08)
        assert beats['_gp_fcf_yield'] is True
        assert trails['_gp_fcf_yield'] is False

    def test_fcf_yield_na_without_fcf(self):
        rows = [self._row(fcf=None, mcap=100.0)]
        apply_screening_matrix(rows)
        assert rows[0]['fcf_yield'] is None
        assert rows[0]['_gp_fcf_yield'] is None

    def test_p_tbv_gate_fails_when_negative(self):
        """Negative tangible book (insolvent on a tangible basis) must not pass
        the P/TBV gate. P/TBV would invert to a misleading "cheap" signal.
        """
        rows = [self._row(p_tbv=-1.5)]
        apply_screening_matrix(rows)
        assert rows[0]['_gp_p_tbv'] is False
        assert rows[0]['_gate_p_tbv'] == pytest.approx(-1.5)

    def test_p_tbv_gate_passes_when_in_range(self):
        rows = [self._row(p_tbv=1.5)]
        apply_screening_matrix(rows)
        assert rows[0]['_gp_p_tbv'] is True

    def test_p_tbv_gate_fails_when_above_threshold(self):
        rows = [self._row(p_tbv=4.0)]
        apply_screening_matrix(rows)
        assert rows[0]['_gp_p_tbv'] is False

    def test_negative_p_tbv_does_not_clamp_continuous_score_to_100(self):
        """Negative P/TBV previously hit _score_linear(-3, 5.0, 1.0) → clamp 100.

        The score function rejects v <= 0 and returns None, which the
        aggregator treats as a worst-case 0 (consistent with other missing
        gates), so a negative-tangible-book company can no longer outscore a
        healthy one.
        """
        rows = [self._row(p_tbv=-3.0), self._row(p_tbv=1.5, ticker='OK')]
        compute_continuous_scores(rows)
        assert rows[0]['_score_p_tbv'] == 0.0
        assert rows[1]['_score_p_tbv'] > rows[0]['_score_p_tbv']

    def test_spread_gate_threshold_is_seven_percent(self):
        """Pin the actual threshold so the display label/tooltip stay aligned."""
        below = [self._row(spread=0.06)]
        above = [self._row(spread=0.08)]
        apply_screening_matrix(below)
        apply_screening_matrix(above)
        assert below[0]['_gp_spread'] is False
        assert above[0]['_gp_spread'] is True


class TestThinEdgarHistoryCap:
    """The rating cap that catches foreign issuers with no EDGAR multi-year history.

    Without this cap, a 20-F/40-F filer whose IFRS taxonomy isn't being parsed
    (so `edgar_history` is None/empty) can score BUY off short-term yfinance
    signals alone. The cap forces them to HOLD until real history is wired up.
    """
    # _fv_effective is normally derived by prepare_scoring_fields; these rows
    # call _rating_cap_for_row/apply_rating_caps directly, so supply it (the
    # missing-FV cap keys on the effective FV, not on dcf_fv).
    _BASE = {'price': 100, 'dcf_fv': 120, '_fv_effective': 120,
             '_price_fv': 0.83, 'mos': 0.17}

    def test_zero_years_capped_to_hold(self):
        row = {**self._BASE, 'edgar_history': {'years_available': 0}}
        cap, reasons = _rating_cap_for_row(row)
        assert cap == 'HOLD'
        assert any('thin EDGAR history' in r for r in reasons)

    def test_missing_edgar_history_capped_to_hold(self):
        row = {**self._BASE, 'edgar_history': None}
        cap, reasons = _rating_cap_for_row(row)
        assert cap == 'HOLD'
        assert any('thin EDGAR history (0y)' in r for r in reasons)

    def test_four_years_still_capped(self):
        row = {**self._BASE, 'edgar_history': {'years_available': 4}}
        cap, reasons = _rating_cap_for_row(row)
        assert cap == 'HOLD'
        assert any('thin EDGAR history (4y)' in r for r in reasons)

    def test_five_years_not_capped(self):
        row = {**self._BASE, 'edgar_history': {'years_available': 5}}
        cap, reasons = _rating_cap_for_row(row)
        assert not any('thin EDGAR history' in r for r in reasons)

    def test_apply_caps_downgrades_buy_to_hold(self):
        rows = [{
            **self._BASE, 'ticker': 'TSMWF',
            '_composite_score': 60.5,
            'edgar_history': {'years_available': 0},
        }]
        apply_rating_caps(rows)
        assert rows[0]['rating_raw'] == 'BUY'
        assert rows[0]['rating'] == 'HOLD'
        assert 'thin EDGAR history (0y)' in rows[0]['_rating_cap_reasons']

    def test_apply_caps_leaves_us_filer_buy(self):
        rows = [{
            **self._BASE, 'ticker': 'AAPL',
            '_composite_score': 60.5,
            'edgar_history': {'years_available': 15},
        }]
        apply_rating_caps(rows)
        assert rows[0]['rating_raw'] == 'BUY'
        assert rows[0]['rating'] == 'BUY'


# ---------------------------------------------------------------------------
# 2026-07 rebalance: new derived fields (prepare_scoring_fields)
# ---------------------------------------------------------------------------

class TestEbitEv:
    def test_derivation(self):
        r = {'operating_income': 12e9, 'enterprise_value': 100e9}
        prepare_scoring_fields([r])
        assert r['ebit_ev'] == pytest.approx(0.12)

    def test_none_on_missing_or_bad_ev(self):
        for row in ({'operating_income': 12e9},
                    {'operating_income': 12e9, 'enterprise_value': 0},
                    {'operating_income': 12e9, 'enterprise_value': -5e9},
                    {'enterprise_value': 100e9}):
            prepare_scoring_fields([row])
            assert row['ebit_ev'] is None


class TestIncrementalRoic:
    def test_normal_case(self):
        r = {'_nopat_by_year': {'2021': 100.0, '2024': 160.0},
             '_ic_by_year':    {'2021': 1000.0, '2024': 1400.0}}
        prepare_scoring_fields([r])
        # ΔNOPAT 60 / ΔIC 400 = 15%
        assert r['incremental_roic'] == pytest.approx(0.15)
        assert r['_incr_roic_undefined'] is False

    def test_shrinking_capital_is_undefined_not_zero(self):
        """Capital-light compounder returning cash: ΔIC <= 0 must flag the
        gate inapplicable, never score it 0."""
        r = {'_nopat_by_year': {'2021': 100.0, '2024': 160.0},
             '_ic_by_year':    {'2021': 1400.0, '2024': 1000.0}}
        prepare_scoring_fields([r])
        assert r['incremental_roic'] is None
        assert r['_incr_roic_undefined'] is True

    def test_tiny_denominator_is_undefined(self):
        r = {'_nopat_by_year': {'2021': 100.0, '2024': 160.0},
             '_ic_by_year':    {'2021': 1000.0, '2024': 1000.5}}
        prepare_scoring_fields([r])
        assert r['incremental_roic'] is None
        assert r['_incr_roic_undefined'] is True

    def test_single_year_or_missing_stays_applicable_na(self):
        """Missing data (no intermediates) is sparse data → gate stays
        applicable and scores 0 via the missing-data path."""
        for row in ({}, {'_nopat_by_year': {'2024': 160.0},
                         '_ic_by_year': {'2024': 1000.0}}):
            prepare_scoring_fields([row])
            assert row['incremental_roic'] is None
            assert row['_incr_roic_undefined'] is False

    def test_clamped_to_sane_range(self):
        r = {'_nopat_by_year': {'2021': 0.0, '2024': 1000.0},
             '_ic_by_year':    {'2021': 900.0, '2024': 1000.0}}
        prepare_scoring_fields([r])
        assert r['incremental_roic'] == 1.0  # 1000/100 clamped


class TestMarginVsHist:
    def test_over_earning_positive(self):
        r = {'operating_margin': 0.25, 'op_margin_avg_10y': 0.15}
        prepare_scoring_fields([r])
        assert r['margin_vs_hist'] == pytest.approx(0.10)

    def test_below_history_negative(self):
        r = {'operating_margin': 0.10, 'op_margin_avg_10y': 0.15}
        prepare_scoring_fields([r])
        assert r['margin_vs_hist'] == pytest.approx(-0.05)

    def test_none_when_either_missing(self):
        for row in ({'operating_margin': 0.2}, {'op_margin_avg_10y': 0.15}, {}):
            prepare_scoring_fields([row])
            assert row['margin_vs_hist'] is None


# ---------------------------------------------------------------------------
# 2026-07 rebalance: applicability mask + per-gate weights
# ---------------------------------------------------------------------------

def _full_row(**overrides):
    """Row with every scoring field populated at mid-range values."""
    row = {
        'ticker': 'FULL', 'sector': 'Technology', 'price': 100.0,
        'dcf_fv': 130.0, 'mos': 0.23, 'fv_dispersion': 0.30,
        'operating_income': 10e9, 'enterprise_value': 100e9,
        'p_tbv': 2.0, 'tangible_book_ps': 50.0,
        'epv_fv': 110.0, 'fcf': 5e9, 'mcap': 90e9, 'revenue': 40e9,
        'int_cov': 10.0, 'accruals': 0.02, 'rev_growth_vol': 0.08,
        'nd_ebitda': 0.5, 'piotroski': 7,
        'operating_margin': 0.25, 'op_margin_avg_10y': 0.24,
        'op_margin_hist_years': 10,
        'roic_cv': 0.15, 'spread': 0.10, 'margin_advantage': 0.06,
        '_nopat_by_year': {'2021': 100.0, '2024': 150.0},
        '_ic_by_year': {'2021': 800.0, '2024': 1000.0},
        'fundamental_growth': 0.05, 'gross_margin_trend': 0.01,
        'rev_cagr_10y': 0.06, 'fcf_cagr_5y': 0.08,
        'shareholder_yield': 0.04, 'insider_pct': 0.08,
        'sbc_pct_rev': 0.01, 'shares_cagr_5y': -0.01,
        'insider_buy_ratio': 0.7, 'insider_buy_count_365d': 5,
        'insider_sell_count_365d': 3,
        'mc_cv': 0.10, '_risk_free_rate': 0.045,
    }
    row.update(overrides)
    return row


class TestApplicabilityMask:
    def test_financial_row_excludes_fcf_gates_from_denominator(self):
        """A bank's FCF-flavored and EV gates go inapplicable: excluded from
        _gates_passed denominator and stored as None, not failed."""
        bank = _full_row(ticker='BANK', sector='Financial Services')
        generic = _full_row(ticker='TECH')
        apply_screening_matrix([bank, generic])
        # ebit_ev, fcf_yield, fcf_margin, fcf_cagr_5y masked for financials
        assert bank['_gates_inapplicable'] == 4
        assert generic['_gates_inapplicable'] == 0
        bank_denom = int(bank['_gates_passed'].split('/')[1])
        gen_denom = int(generic['_gates_passed'].split('/')[1])
        assert gen_denom - bank_denom == 4
        assert bank['_gp_ebit_ev'] is None
        assert bank['_gp_fcf_yield'] is None

    def test_inapplicable_scores_are_none_and_category_renormalizes(self):
        """Scores render N/A (None) for inapplicable gates, and the category
        average is taken over applicable weight only."""
        bank = _full_row(ticker='BANK', sector='Financial Services')
        generic = _full_row(ticker='TECH')
        compute_continuous_scores([bank, generic])
        assert bank['_score_ebit_ev'] is None
        assert bank['_score_fcf_yield'] is None
        # Valuation average must not be dragged to 0 by the masked gates:
        # both rows share identical applicable-gate inputs, so the bank's
        # category scores stay in a sane band rather than collapsing.
        assert bank['_score_valuation'] is not None
        assert bank['_score_valuation'] > 0
        assert bank['_score_moat'] is not None

    def test_negative_tbv_masks_ptbv(self):
        row = _full_row(ticker='NEGTBV', tangible_book_ps=-5.0, p_tbv=None)
        compute_continuous_scores([row])
        assert row['_score_p_tbv'] is None
        # Missing balance sheet (no tangible_book_ps at all) stays applicable
        missing = _full_row(ticker='NOBOOK', p_tbv=None)
        missing.pop('tangible_book_ps')
        compute_continuous_scores([missing])
        assert missing['_score_p_tbv'] == 0.0

    def test_quiet_insiders_masked_not_zeroed(self):
        quiet = _full_row(ticker='QUIET', insider_buy_count_365d=1,
                          insider_sell_count_365d=1, insider_buy_ratio=None)
        no_data = _full_row(ticker='NODATA', insider_buy_count_365d=None,
                            insider_sell_count_365d=None,
                            insider_buy_ratio=None)
        active = _full_row(ticker='ACTIVE')
        compute_continuous_scores([quiet, no_data, active])
        assert quiet['_score_insider_buying'] is None
        assert no_data['_score_insider_buying'] is None
        assert active['_score_insider_buying'] == pytest.approx(70.0)

    def test_thin_margin_history_masks_guard_gate(self):
        thin = _full_row(ticker='THIN', op_margin_hist_years=3)
        compute_continuous_scores([thin])
        assert thin['_score_margin_vs_hist'] is None

    def test_data_coverage_over_applicable_gates(self):
        """A fully-covered bank must not read as low-coverage because of
        gates that cannot describe it."""
        bank = _full_row(ticker='BANK', sector='Financial Services')
        apply_screening_matrix([bank])
        compute_continuous_scores([bank])
        assert bank['_data_coverage_score'] is not None
        assert bank['_data_coverage_score'] > 80


class TestGateWeights:
    def test_valuation_average_is_weighted(self):
        """The Valuation category average must equal the per-gate weighted
        mean with MoS counted at weight 2.0."""
        from scripts.scoring import _score_key
        row = _full_row()
        compute_continuous_scores([row])
        num = den = 0.0
        for g in SCORING_GATES:
            if g.category != 'Valuation':
                continue
            s = row[_score_key(g.name)]
            assert s is not None  # _full_row keeps every valuation gate live
            num += s * g.weight
            den += g.weight
        assert den == pytest.approx(7.0)  # 6 gates, MoS double-weighted
        assert row['_score_valuation'] == pytest.approx(num / den, abs=0.1)

    def test_mos_weight_is_two(self):
        mos_gate = next(g for g in SCORING_GATES
                        if g.name == 'Valuation: MoS')
        assert mos_gate.weight == 2.0
        others = [g.weight for g in SCORING_GATES
                  if g.name != 'Valuation: MoS']
        assert all(w == 1.0 for w in others)


class TestRatingCapUsesFvEffective:
    def test_consensus_fallback_row_not_capped_for_missing_fv(self):
        """A DCF-less row whose MoS came from the multi-model consensus
        must NOT be capped at HOLD for 'missing fair value'."""
        row = {'ticker': 'NODCF', 'price': 100.0, 'dcf_fv': None,
               '_fv_effective': 125.0, 'mos': 0.20,
               'edgar_history': {'years_available': 10}}
        cap, reasons = _rating_cap_for_row(row)
        assert not any('missing price or fair value' in r for r in reasons)

    def test_no_fv_at_all_still_capped(self):
        row = {'ticker': 'NOFV', 'price': 100.0, 'dcf_fv': None,
               '_fv_effective': None, 'mos': None,
               'edgar_history': {'years_available': 10}}
        cap, reasons = _rating_cap_for_row(row)
        assert cap == 'HOLD'
        assert any('missing price or fair value' in r for r in reasons)
