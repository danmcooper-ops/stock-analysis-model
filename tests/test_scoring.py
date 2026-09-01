# tests/test_scoring.py
import pytest


from scripts.scoring import (
    _score_linear, compute_continuous_scores, apply_composite_rating_override,
    rating_from_composite, _mc_confidence_label, GATES,
    gate_metadata, score_and_rate,
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


class TestEdgarIntCovFallback:
    def test_fallback_populates_int_cov_when_yfinance_missing(self):
        """No yfinance income statement (int_cov None) but EDGAR-derived
        int_cov_edgar present → int_cov is populated and tagged."""
        r = {'sector': 'Technology', 'int_cov': None, 'int_cov_edgar': 29.06}
        prepare_scoring_fields([r])
        assert r['int_cov'] == pytest.approx(29.06)
        assert r['_int_cov_source'] == 'edgar'

    def test_existing_yfinance_int_cov_not_overridden(self):
        """yfinance stays the primary source when it resolved."""
        r = {'sector': 'Industrials', 'int_cov': 6.05, 'int_cov_edgar': 8.22}
        prepare_scoring_fields([r])
        assert r['int_cov'] == pytest.approx(6.05)
        assert '_int_cov_source' not in r

    def test_financial_services_excluded_from_fallback(self):
        """Interest is a bank's cost of goods, not a fixed charge to cover —
        the gate masks the sector, so the fallback must not populate it."""
        r = {'sector': 'Financial Services', 'int_cov': None,
             'int_cov_edgar': 1.4}
        prepare_scoring_fields([r])
        assert r['int_cov'] is None
        assert r.get('_int_cov_source') is None

    def test_negative_coverage_propagates(self):
        """Negative EBIT → negative coverage: a real 'cannot cover interest'
        read the gate should fail, not suppress into N/A."""
        r = {'sector': 'Healthcare', 'int_cov': None, 'int_cov_edgar': -2.5}
        prepare_scoring_fields([r])
        assert r['int_cov'] == pytest.approx(-2.5)
        assert r['_int_cov_source'] == 'edgar'


class TestSbcXbrlPreference:
    def test_xbrl_value_preferred_over_yfinance(self):
        r = {'sector': 'Technology', 'sbc': 10.0, 'revenue': 1000.0,
             'sbc_pct_rev_xbrl': 0.045}
        prepare_scoring_fields([r])
        assert r['sbc_pct_rev'] == pytest.approx(0.045)  # not 10/1000

    def test_yfinance_fallback_when_xbrl_missing(self):
        r = {'sector': 'Technology', 'sbc': 10.0, 'revenue': 1000.0}
        prepare_scoring_fields([r])
        assert r['sbc_pct_rev'] == pytest.approx(0.01)

    def test_none_when_both_missing(self):
        r = {'sector': 'Technology', 'revenue': 1000.0}
        prepare_scoring_fields([r])
        assert r['sbc_pct_rev'] is None

    def test_xbrl_zero_is_a_value_not_missing(self):
        """A company reporting zero SBC via XBRL must keep the 0.0 (best
        score), not fall through to the yfinance path."""
        r = {'sector': 'Technology', 'sbc': 10.0, 'revenue': 1000.0,
             'sbc_pct_rev_xbrl': 0.0}
        prepare_scoring_fields([r])
        assert r['sbc_pct_rev'] == 0.0


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


class TestMcConfidenceLabelConstrained:
    """Constraint diagnostics downgrade the label one notch and tag it."""

    def test_high_becomes_medium_when_clipped(self):
        assert _mc_confidence_label(0.15, clip_rate=0.35) == 'MEDIUM (15%, constrained)'

    def test_medium_becomes_low_when_wipeouts_are_common(self):
        assert _mc_confidence_label(0.30, clip_rate=0.0, invalid_rate=0.25) == 'LOW (30%, constrained)'

    def test_low_stays_low_but_is_tagged(self):
        assert _mc_confidence_label(0.55, clip_rate=0.9) == 'LOW (55%, constrained)'

    def test_thresholds_are_exclusive(self):
        from scripts.config import MC_CLIP_RATE_DOWNGRADE, MC_INVALID_RATE_DOWNGRADE
        assert _mc_confidence_label(0.15, clip_rate=MC_CLIP_RATE_DOWNGRADE,
                                    invalid_rate=MC_INVALID_RATE_DOWNGRADE) == 'HIGH (15%)'
        assert _mc_confidence_label(0.15, clip_rate=MC_CLIP_RATE_DOWNGRADE + 1e-9) \
            == 'MEDIUM (15%, constrained)'

    def test_missing_diagnostics_leave_label_unchanged(self):
        assert _mc_confidence_label(0.15) == 'HIGH (15%)'
        assert _mc_confidence_label(0.15, clip_rate=None, invalid_rate=None) == 'HIGH (15%)'

    def test_none_cv_still_none(self):
        assert _mc_confidence_label(None, clip_rate=0.9, invalid_rate=0.9) is None


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
            # Incr ROIC computes at 25% (ΔNOPAT 50 / ΔIC 200) — without
            # these the gate scores 0 on missing data, which FCF Margin's
            # near-perfect score used to mask before the Pool Share swap.
            '_nopat_by_year': {'2021': 100.0, '2024': 150.0},
            '_ic_by_year': {'2021': 800.0, '2024': 1000.0},
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
        assert rows[0]['roic_trend_slope'] == pytest.approx(0.04)

    def test_gate_metadata_matches_gate_count(self):
        meta = gate_metadata()
        assert len(meta['gates']) == len(GATES)
        keys = {g['key'] for g in meta['gates']}
        assert '_gate_margin_advantage' in keys
        assert '_gate_mult_vs_hist' in keys   # time-series cheapness (replaced EPV Floor)
        assert '_gate_fcf_yield' in keys
        assert '_gate_rev_volatility' in keys
        # 2026-07 rebalance additions
        assert '_gate_ebit_ev' in keys
        assert '_gate_incr_roic' in keys
        assert '_gate_margin_vs_hist' in keys
        assert '_gate_insider_buying' in keys
        assert '_gate_pool_share' in keys
        # Retired gates must be gone
        assert '_gate_roic_trend' not in keys
        assert '_gate_rim_mos' not in keys
        assert '_gate_roe' not in keys
        assert '_gate_p_fcf' not in keys         # reciprocal of FCF Yield
        assert '_gate_buyback_rate' not in keys  # inside Shrhldr Yield
        assert '_gate_cash_conv' not in keys     # accruals inverted
        assert '_gate_gross_margin' not in keys  # merged into Margin Adv
        assert '_gate_epv_floor' not in keys     # collinear with MoS (r=0.68)
        assert '_gate_fcf_margin' not in keys    # margin level = Margin Adv's
        # axis; FCF votes via FCF Yield / FCF Durability / Accruals

    def test_every_gate_has_test_and_score_fns(self):
        """The unified spec must carry BOTH halves for every gate — a gate
        with a missing fn would blank its Matrix cell or score column."""
        for gate in GATES:
            assert callable(gate.test_fn), gate.name
            assert callable(gate.score_fn), gate.name

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


def _pp_row(ticker, oi_by_year, sector='Widgets'):
    """Row with only what the pool-share trajectory pass consumes."""
    return {'ticker': ticker, 'sector': sector,
            'edgar_history': {'operating_income_history': oi_by_year}}


class TestPoolShareTrajectory:
    def _base_universe(self):
        """Three Widgets tickers with a full 2019/2024 panel. Pools:
        2019 = 20+50+30 = 100, 2024 = 45+70+35 = 150.
        A's share: 0.20 → 0.30."""
        return [
            _pp_row('A', {2019: 20.0, 2024: 45.0}),
            _pp_row('B', {2019: 50.0, 2024: 70.0}),
            _pp_row('C', {2019: 30.0, 2024: 35.0}),
        ]

    def test_full_5y_window_hand_checked(self):
        rows = self._base_universe()
        prepare_scoring_fields(rows)
        a = rows[0]
        assert a['pool_share_cagr'] == pytest.approx((0.30 / 0.20) ** (1 / 5) - 1)
        assert a['_pool_share_undefined'] is False

    def test_string_keys_match_int_keys(self):
        """Snapshot JSON round-trip turns year keys into strings — values
        must be identical to the int-keyed live run."""
        int_rows = self._base_universe()
        str_rows = [
            _pp_row('A', {'2019': 20.0, '2024': 45.0}),
            _pp_row('B', {'2019': 50.0, '2024': 70.0}),
            _pp_row('C', {'2019': 30.0, '2024': 35.0}),
        ]
        prepare_scoring_fields(int_rows)
        prepare_scoring_fields(str_rows)
        assert str_rows[0]['pool_share_cagr'] == pytest.approx(
            int_rows[0]['pool_share_cagr'])

    def test_span_fallback_with_3y_floor(self):
        """No latest−5 year → fall back to oldest, annualized over the
        actual span; below 3 years the gate is undefined."""
        rows = [
            _pp_row('A', {2021: 20.0, 2024: 45.0}),   # span 3: computes
            _pp_row('B', {2021: 50.0, 2024: 70.0}),
            _pp_row('C', {2021: 30.0, 2024: 35.0}),
        ]
        prepare_scoring_fields(rows)
        a = rows[0]
        assert a['pool_share_cagr'] == pytest.approx((0.30 / 0.20) ** (1 / 3) - 1)
        assert a['_pool_share_undefined'] is False

        short = _pp_row('D', {2022: 10.0, 2024: 12.0})  # span 2: undefined
        rows2 = self._base_universe() + [short]
        prepare_scoring_fields(rows2)
        assert short['pool_share_cagr'] is None
        assert short['_pool_share_undefined'] is True

    def test_consistent_panel_excludes_partial_peers(self):
        """A peer reporting only in the end year must not enter either
        endpoint pool — subject's CAGR is unchanged by its arrival."""
        baseline = self._base_universe()
        prepare_scoring_fields(baseline)
        expected = baseline[0]['pool_share_cagr']

        newcomer = _pp_row('NEW', {2024: 999.0})   # IPO: end year only
        rows = self._base_universe() + [newcomer]
        prepare_scoring_fields(rows)
        assert rows[0]['pool_share_cagr'] == pytest.approx(expected)
        assert newcomer['pool_share_cagr'] is None
        assert newcomer['_pool_share_undefined'] is True

    def test_panel_below_min_sector_stocks_is_undefined(self):
        rows = [
            _pp_row('A', {2019: 20.0, 2024: 45.0}),
            _pp_row('B', {2019: 50.0, 2024: 70.0}),
        ]
        prepare_scoring_fields(rows)
        for r in rows:
            assert r['pool_share_cagr'] is None
            assert r['_pool_share_undefined'] is True

    def test_negative_endpoints_clamp_in_pool_and_undefine_subject(self):
        """Peer's negative year contributes 0 to the pool; a subject with a
        loss-making endpoint has share 0 → undefined, never −100%."""
        rows = [
            _pp_row('A', {2019: 20.0, 2024: 45.0}),
            _pp_row('B', {2019: 50.0, 2024: 70.0}),
            _pp_row('C', {2019: -10.0, 2024: 35.0}),  # clamped to 0 in 2019 pool
        ]
        prepare_scoring_fields(rows)
        a, c = rows[0], rows[2]
        # Pools: 2019 = 20+50+0 = 70, 2024 = 45+70+35 = 150
        assert a['pool_share_cagr'] == pytest.approx(
            ((45 / 150) / (20 / 70)) ** (1 / 5) - 1)
        assert c['pool_share_cagr'] is None
        assert c['_pool_share_undefined'] is True

    def test_structural_na_degradation(self):
        """No edgar_history / empty history / missing sector → None + flag;
        the screening gate renders N/A and leaves the denominator."""
        rows = [
            {'ticker': 'X'},                                        # no history
            {'ticker': 'Y', 'sector': 'Widgets',
             'edgar_history': {'operating_income_history': {}}},    # empty
            _pp_row('Z', {2019: 20.0, 2024: 45.0}, sector=None),    # no sector
        ]
        prepare_scoring_fields(rows)
        for r in rows:
            assert r['pool_share_cagr'] is None
            assert r['_pool_share_undefined'] is True

        solo = {'ticker': 'X'}
        apply_screening_matrix([solo])
        assert solo['_gp_pool_share'] is None
        assert solo['_gates_inapplicable'] >= 1

    def test_near_zero_start_share_is_clamped(self):
        rows = [
            _pp_row('A', {2019: 0.001, 2024: 50.0}),
            _pp_row('B', {2019: 100.0, 2024: 100.0}),
            _pp_row('C', {2019: 100.0, 2024: 100.0}),
        ]
        prepare_scoring_fields(rows)
        assert rows[0]['pool_share_cagr'] == 1.0

    def test_stale_snapshot_values_are_overwritten(self):
        """Rescoring a row whose prior run computed a value must recompute
        from scratch — a row that can no longer compute goes back to N/A."""
        r = {'ticker': 'X', 'pool_share_cagr': 0.42,
             '_pool_share_undefined': False}
        prepare_scoring_fields([r])
        assert r['pool_share_cagr'] is None
        assert r['_pool_share_undefined'] is True


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
        'mult_vs_hist': -0.15, 'mult_hist_years': 10,
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
        # ebit_ev, fcf_yield, fcf_cagr_5y, int_coverage, net_debt_ebitda and
        # margins masked for financials; pool_share is inapplicable for BOTH
        # rows (no edgar_history in the fixture)
        assert bank['_gates_inapplicable'] == 7
        assert generic['_gates_inapplicable'] == 1
        bank_denom = int(bank['_gates_passed'].split('/')[1])
        gen_denom = int(generic['_gates_passed'].split('/')[1])
        assert gen_denom - bank_denom == 6
        assert bank['_gp_ebit_ev'] is None
        assert bank['_gp_fcf_yield'] is None
        assert bank['_gp_int_coverage'] is None
        assert bank['_gp_net_debt_ebitda'] is None
        # Gross margin does not exist for banks/insurers — the gate must be
        # masked, not silently failed against the denominator. The fixture
        # supplies a passing gross_margin_trend, so a mask (rather than a
        # missing-data N/A) is the only thing that can null this out.
        assert generic['_gp_margins'] is True
        assert bank['_gp_margins'] is None
        assert bank['_gate_margins'] is None

    def test_inapplicable_scores_are_none_and_category_renormalizes(self):
        """Scores render N/A (None) for inapplicable gates, and the category
        average is taken over applicable weight only."""
        bank = _full_row(ticker='BANK', sector='Financial Services')
        generic = _full_row(ticker='TECH')
        compute_continuous_scores([bank, generic])
        assert bank['_score_ebit_ev'] is None
        assert bank['_score_fcf_yield'] is None
        assert bank['_score_int_coverage'] is None
        assert bank['_score_net_debt_ebitda'] is None
        # Category averages must not be dragged to 0 by the masked gates:
        # both rows share identical applicable-gate inputs, so the bank's
        # category scores stay in a sane band rather than collapsing.
        assert bank['_score_valuation'] is not None
        assert bank['_score_valuation'] > 0
        assert bank['_score_quality'] is not None
        assert bank['_score_quality'] > 0
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
        for g in GATES:
            if g.category != 'Valuation':
                continue
            s = row[_score_key(g.name)]
            assert s is not None  # _full_row keeps every valuation gate live
            num += s * g.weight
            den += g.weight
        assert den == pytest.approx(7.0)  # 6 gates, MoS double-weighted
        assert row['_score_valuation'] == pytest.approx(num / den, abs=0.1)

    def test_mos_weight_is_two(self):
        mos_gate = next(g for g in GATES
                        if g.name == 'Valuation: MoS')
        assert mos_gate.weight == 2.0
        others = [g.weight for g in GATES
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


class TestFxFetchFailedCap:
    def test_fx_failure_caps_to_hold(self):
        row = {'ticker': 'ADR', 'price': 100.0, '_fv_effective': 130.0,
               'mos': 0.23, 'fx_fetch_failed': True,
               'edgar_history': {'years_available': 10}}
        cap, reasons = _rating_cap_for_row(row)
        assert cap == 'HOLD'
        assert any('FX conversion failed' in r for r in reasons)

    def test_no_cap_when_fx_ok(self):
        row = {'ticker': 'OK', 'price': 100.0, '_fv_effective': 130.0,
               'mos': 0.23, 'fx_fetch_failed': False,
               'edgar_history': {'years_available': 10}}
        cap, reasons = _rating_cap_for_row(row)
        assert not any('FX conversion failed' in r for r in reasons)


class TestMultipleVsHistory:
    """compute_multiple_vs_history — time-series cheapness (replaced EPV Floor)."""

    def _inputs(self, n_years=8, price=10.0, oi=100.0, end=None):
        import pandas as pd
        from datetime import date as _d
        end = end or _d.today().isoformat()
        idx = pd.date_range('2016-01-01', end, freq='B')
        close = pd.Series(price, index=idx)  # flat adjusted price
        eh = {
            'operating_income_history': {y: oi for y in range(2016, 2016 + n_years)},
        }
        return close, eh

    def test_flat_history_at_own_median(self):
        """Same multiple today as always → metric ≈ 0 (at own median)."""
        from scripts.analyze_stock import compute_multiple_vs_history
        close, eh = self._inputs()
        # historical multiple = 10/100; current = 10/100
        v, yrs = compute_multiple_vs_history(close, eh, operating_income=100.0)
        assert yrs == 8
        assert v == pytest.approx(0.0, abs=1e-9)

    def test_cheap_vs_own_history(self):
        """Current EBIT double the historical at the same price → −50%."""
        from scripts.analyze_stock import compute_multiple_vs_history
        close, eh = self._inputs()
        v, _ = compute_multiple_vs_history(close, eh, operating_income=200.0)
        assert v == pytest.approx(-0.50)

    def test_split_invariance(self):
        """A 20:1 split (halved-and-halved-again adjusted history) must NOT
        read as expensive: the same adjusted series feeds history and today,
        so the share basis cancels. Regression for the GOOGL/NVDA zeroing —
        the old price×as-reported-shares construction read splitters at
        3-4× their own median."""
        from scripts.analyze_stock import compute_multiple_vs_history
        close, eh = self._inputs()
        # Split-adjusted series: same economic value, 20× smaller prices.
        v, _ = compute_multiple_vs_history(close / 20.0, eh,
                                           operating_income=100.0)
        assert v == pytest.approx(0.0, abs=1e-9)

    def test_insufficient_history_returns_none(self):
        from scripts.analyze_stock import compute_multiple_vs_history
        close, eh = self._inputs(n_years=3)
        v, yrs = compute_multiple_vs_history(close, eh, operating_income=100.0)
        assert v is None and yrs < 5

    def test_negative_ebit_years_excluded(self):
        from scripts.analyze_stock import compute_multiple_vs_history
        close, eh = self._inputs(n_years=8)
        eh['operating_income_history'][2018] = -50.0  # loss year drops out
        v, yrs = compute_multiple_vs_history(close, eh, operating_income=100.0)
        assert yrs == 7
        assert v == pytest.approx(0.0, abs=1e-9)

    def test_none_without_prices_or_current_inputs(self):
        from scripts.analyze_stock import compute_multiple_vs_history
        close, eh = self._inputs()
        assert compute_multiple_vs_history(None, eh, 100.0) == (None, 0)
        assert compute_multiple_vs_history(close, eh, None) == (None, 0)
        assert compute_multiple_vs_history(close, eh, -5.0) == (None, 0)

    def test_stale_price_tail_returns_none(self):
        """A parquet whose last bar is months old (delisting, failed refresh)
        must not masquerade as today's price."""
        from scripts.analyze_stock import compute_multiple_vs_history
        close, eh = self._inputs(end='2025-12-31')
        v, yrs = compute_multiple_vs_history(close, eh, operating_income=100.0)
        assert v is None and yrs >= 5

    def test_mask_requires_five_years(self):
        row = _full_row(mult_vs_hist=None, mult_hist_years=3)
        compute_continuous_scores([row])
        assert row['_score_mult_vs_hist'] is None  # inapplicable, not zeroed
        ok = _full_row()  # 10 years in the fixture
        compute_continuous_scores([ok])
        assert ok['_score_mult_vs_hist'] is not None


class TestTrapSignals:
    """Value-trap overlay: fail-open nulls, axis skipping, coverage floor."""

    def _full_trap_row(self):
        # Levered secular decliner with an uncovered dividend, derated
        # multiple, deep drawdown and a crowded short — every axis hot.
        return {
            'rev_down_years': 3, 'rev_cagr_5y': -0.06, 'rev_cagr_10y': -0.02,
            'gross_margin_trend': -0.012, 'fcf_neg_years_5y': 3,
            'nd_ebitda': 4.5, 'int_cov': 1.5, 'altman_z_zone': 'grey',
            'net_debt_slope_3y': 0.06,
            'spread': -0.06, 'roic_trend_slope': -0.12,
            'incremental_roic': -0.05, 'wacc': 0.09,
            '_incr_roic_undefined': False, 'pool_share_cagr': -0.12,
            'mult_vs_hist': -0.55, 'fv_dispersion': 0.55,
            'div_fcf_ratio_3y': 1.8,
            'momentum_12_1': -0.55, 'short_pct_float': 0.28,
        }

    def _compounder_row(self):
        return {
            'rev_down_years': 0, 'rev_cagr_5y': 0.12, 'rev_cagr_10y': 0.10,
            'gross_margin_trend': 0.004, 'fcf_neg_years_5y': 0,
            'nd_ebitda': 0.5, 'int_cov': 15.0, 'altman_z_zone': 'safe',
            'net_debt_slope_3y': -0.02,
            'spread': 0.15, 'roic_trend_slope': 0.05,
            'incremental_roic': 0.30, 'wacc': 0.09,
            '_incr_roic_undefined': False, 'pool_share_cagr': 0.05,
            'mult_vs_hist': 0.10, 'fv_dispersion': 0.10,
            'div_fcf_ratio_3y': 0.3,
            'momentum_12_1': 0.20, 'short_pct_float': 0.02,
        }

    def test_full_trap_scores_high(self):
        from scripts.scoring import compute_trap_signals
        r = self._full_trap_row()
        compute_trap_signals([r])
        assert r['trap_score'] >= 75
        assert r['trap_flag'] is True
        assert len(r['trap_reasons']) >= 4
        assert set(r['_trap_components']) == {
            'decline', 'balance_sheet', 'value_destruction',
            'derating', 'payout', 'market'}

    def test_compounder_scores_low(self):
        from scripts.scoring import compute_trap_signals
        r = self._compounder_row()
        compute_trap_signals([r])
        assert r['trap_score'] <= 15
        assert r['trap_flag'] is False

    def test_all_none_fails_open(self):
        from scripts.scoring import compute_trap_signals
        r = {}
        compute_trap_signals([r])
        assert r['trap_score'] is None
        assert r['trap_flag'] is None          # never True on thin data
        assert r['trap_reasons'] == []

    def test_coverage_floor_three_axes_none(self):
        from scripts.scoring import compute_trap_signals
        # Only decline + balance_sheet + derating resolve: 3 axes < floor(4).
        r = {'rev_down_years': 3, 'nd_ebitda': 4.0,
             'mult_vs_hist': -0.4, 'fv_dispersion': 0.4}
        compute_trap_signals([r])
        assert r['trap_score'] is None
        assert r['trap_flag'] is None

    def test_coverage_floor_four_axes_scores(self):
        from scripts.scoring import compute_trap_signals
        # decline(0.25)+balance_sheet(0.20)+value_destruction(0.20)+
        # derating(0.15) = 4 axes, 0.80 weight → passes both floors.
        r = {'rev_down_years': 3, 'nd_ebitda': 4.0, 'spread': -0.06,
             'mult_vs_hist': -0.4}
        compute_trap_signals([r])
        assert r['trap_score'] is not None

    def test_incr_roic_undefined_skipped(self):
        from scripts.scoring import compute_trap_signals
        base = {'spread': 0.10, 'roic_trend_slope': 0.0,
                'incremental_roic': None, 'wacc': 0.09,
                '_incr_roic_undefined': True,
                'rev_down_years': 0, 'nd_ebitda': 1.0,
                'mult_vs_hist': 0.0, 'div_fcf_ratio_3y': 0.0}
        r1 = dict(base)
        compute_trap_signals([r1])
        # A shrinking capital base must not read as value destruction: the
        # sub-score list for C excludes c3 entirely.
        r2 = dict(base, _incr_roic_undefined=False, incremental_roic=-0.20)
        compute_trap_signals([r2])
        assert r2['trap_score'] > r1['trap_score']

    def test_net_debt_slope_only_counts_when_levered(self):
        from scripts.scoring import compute_trap_signals
        base = {'rev_down_years': 2, 'spread': -0.02, 'mult_vs_hist': -0.3,
                'div_fcf_ratio_3y': 0.0, 'int_cov': 10.0}
        lo = dict(base, nd_ebitda=0.5, net_debt_slope_3y=0.10)
        hi = dict(base, nd_ebitda=3.0, net_debt_slope_3y=0.10)
        compute_trap_signals([lo, hi])
        lo_b = lo['_trap_components']['balance_sheet']['score']
        hi_b = hi['_trap_components']['balance_sheet']['score']
        # Unlevered: slope skipped, B = mean(nde-ramp 0, int_cov 0) = 0.
        assert lo_b == 0.0
        assert hi_b > lo_b

    def test_nonpayer_payout_axis_present_at_zero(self):
        from scripts.scoring import compute_trap_signals
        r = dict(self._compounder_row(), div_fcf_ratio_3y=0.0)
        compute_trap_signals([r])
        assert r['_trap_components']['payout']['score'] == 0.0

    def test_reasons_sorted_by_contribution(self):
        from scripts.scoring import compute_trap_signals
        r = self._full_trap_row()
        compute_trap_signals([r])
        # decline (0.25 weight, saturated) must outrank market (0.10).
        reasons = r['trap_reasons']
        assert reasons.index('Structural revenue/margin decline') \
            < reasons.index('Heavy short interest / falling knife')

    def test_purge_preserves_trap_fields(self):
        from scripts.scoring import compute_trap_signals, _purge_stale_gate_fields
        r = self._full_trap_row()
        compute_trap_signals([r])
        _purge_stale_gate_fields([r])
        assert 'trap_score' in r and '_trap_components' in r

    def test_display_only_rating_untouched(self):
        from scripts.scoring import score_and_rate
        # End-to-end through the canonical entry: the trap fields appear and
        # the rating is identical to what the row's fundamentals imply —
        # i.e., no trap-driven cap exists yet.
        r = self._full_trap_row()
        r.update({'ticker': 'TRAP', 'price': 10.0, 'mcap': 1e9})
        score_and_rate([r])
        assert r.get('trap_score') is not None
        assert '_rating_cap_reasons' not in r or not any(
            'trap' in s.lower() for s in (r.get('_rating_cap_reasons') or []))
