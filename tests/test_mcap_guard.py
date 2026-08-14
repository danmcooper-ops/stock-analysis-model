# tests/test_mcap_guard.py
import sys
import os

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.analyze_stock import apply_mcap_integrity_guard


def _rows(n, missing=0, price=100.0):
    """n rows, the first *missing* of which have no market cap."""
    out = []
    for i in range(n):
        has = i >= missing
        out.append({
            'ticker': f'T{i}',
            'price': price,
            'mcap': 1_000_000.0 if has else None,
            'shares_out': 10_000.0 if has else None,
        })
    return out


class TestMcapIntegrityGuard:
    def test_clean_run_reports_nothing(self):
        rows = _rows(100)
        summary = apply_mcap_integrity_guard(rows)
        assert summary == {'recovered': 0, 'still_missing': 0,
                           'miss_pct': 0.0, 'alert': False,
                           'implausible_nulled': 0, 'contaminated_nulled': 0}

    def test_baseline_miss_rate_does_not_alert(self):
        """~0.1% (2 of 2250) is the normal level and must stay quiet."""
        rows = _rows(2250, missing=2)
        summary = apply_mcap_integrity_guard(rows)
        assert summary['still_missing'] == 2
        assert summary['alert'] is False

    def test_regression_scale_miss_rate_alerts(self):
        """The 2026-07-29 failure: 265 of 2244 with no prior snapshot."""
        rows = _rows(2244, missing=265)
        summary = apply_mcap_integrity_guard(rows)
        assert summary['still_missing'] == 265
        assert summary['miss_pct'] == pytest.approx(0.1181, abs=1e-3)
        assert summary['alert'] is True

    def test_prior_snapshot_reprices_yesterdays_share_count(self):
        rows = _rows(10, missing=1, price=50.0)
        prior = [{'ticker': 'T0', 'shares_out': 2_000.0, 'mcap': 90_000.0}]
        summary = apply_mcap_integrity_guard(rows, prior)
        assert rows[0]['mcap'] == pytest.approx(100_000.0)  # today's price
        assert rows[0]['shares_out'] == 2_000.0
        assert rows[0]['_mcap_source'] == 'prior_snapshot_shares'
        assert summary['recovered'] == 1
        assert summary['still_missing'] == 0

    def test_falls_back_to_prior_mcap_when_price_is_missing(self):
        rows = _rows(10, missing=1)
        rows[0]['price'] = None
        prior = [{'ticker': 'T0', 'shares_out': 2_000.0, 'mcap': 90_000.0}]
        apply_mcap_integrity_guard(rows, prior)
        assert rows[0]['mcap'] == 90_000.0
        assert rows[0]['_mcap_source'] == 'prior_snapshot_mcap'

    def test_recovery_is_always_flagged(self):
        """A silent fallback would let stale data pose as fresh."""
        rows = _rows(50, missing=10)
        prior = [{'ticker': f'T{i}', 'shares_out': 1_000.0} for i in range(10)]
        apply_mcap_integrity_guard(rows, prior)
        recovered = [r for r in rows if r.get('mcap') and r.get('_mcap_source')]
        assert len(recovered) == 10
        assert all(r['_mcap_source'].startswith('prior_snapshot') for r in recovered)
        # Untouched rows must not gain the flag.
        assert all('_mcap_source' not in r for r in rows[10:])

    def test_alert_still_fires_when_fallback_cannot_cover_everything(self):
        """Partial recovery must not silence the warning."""
        rows = _rows(1000, missing=200)
        prior = [{'ticker': f'T{i}', 'shares_out': 1_000.0} for i in range(50)]
        summary = apply_mcap_integrity_guard(rows, prior)
        assert summary['recovered'] == 50
        assert summary['still_missing'] == 150
        assert summary['alert'] is True

    def test_ticker_absent_from_prior_snapshot_is_left_alone(self):
        rows = _rows(10, missing=1)
        summary = apply_mcap_integrity_guard(rows, [{'ticker': 'OTHER',
                                                     'shares_out': 5.0}])
        assert rows[0]['mcap'] is None
        assert '_mcap_source' not in rows[0]
        assert summary['recovered'] == 0

    def test_zero_shares_in_prior_snapshot_is_not_used(self):
        rows = _rows(10, missing=1)
        prior = [{'ticker': 'T0', 'shares_out': 0, 'mcap': 0}]
        apply_mcap_integrity_guard(rows, prior)
        assert rows[0]['mcap'] is None
        assert '_mcap_source' not in rows[0]

    def test_empty_results_do_not_divide_by_zero(self):
        summary = apply_mcap_integrity_guard([], [])
        assert summary['miss_pct'] == 0.0
        assert summary['alert'] is False

    def test_threshold_is_configurable(self):
        rows = _rows(100, missing=5)
        assert apply_mcap_integrity_guard(rows, threshold=0.10)['alert'] is False
        assert apply_mcap_integrity_guard(rows, threshold=0.01)['alert'] is True


def _family(count, prices, prefix='PFD'):
    """Rows sharing one share count — each priced on its own quote."""
    return [{'ticker': f'{prefix}{i}', 'price': p, 'mcap': p * count,
             'shares_out': float(count)}
            for i, p in enumerate(prices)]


class TestContaminatedShareClusters:
    """Yahoo hands preferred OTC lines the parent's common share count
    (FNM* all 5.74B, FMC*/FRE* all 3.22B on 2026-08-12), manufacturing
    $24B-$90B phantom caps that sit BELOW MCAP_MAX_PLAUSIBLE."""

    def test_preferred_family_is_nulled(self):
        """The Freddie shape: one count, many diverging preferred quotes."""
        rows = _rows(10) + _family(3_221_329_920, [7.4, 12.08, 14.02, 15.61])
        summary = apply_mcap_integrity_guard(rows)
        assert summary['contaminated_nulled'] == 4
        for r in rows[10:]:
            assert r['mcap'] is None
            assert r['shares_out'] is None
            assert r['_shares_contaminated'] == 3_221_329_920

    def test_adr_pair_is_left_alone(self):
        """SONY/SNEJF: two listings of one security genuinely share the
        company-level count — pairs must never be treated as contamination."""
        rows = _rows(10) + _family(5_853_570_570, [23.10, 23.25], prefix='ADR')
        summary = apply_mcap_integrity_guard(rows)
        assert summary['contaminated_nulled'] == 0
        assert all(r['mcap'] for r in rows[10:])

    def test_multi_listing_trio_with_tight_prices_is_left_alone(self):
        """Clinuvel's three lines (6.68-6.95) share a real count; the tight
        price band marks them as one security, not a preferred family."""
        rows = _rows(10) + _family(50_427_898, [6.68, 6.75, 6.95], prefix='CLV')
        summary = apply_mcap_integrity_guard(rows)
        assert summary['contaminated_nulled'] == 0
        assert all(r['mcap'] for r in rows[10:])

    def test_coincidental_duplicate_pair_is_left_alone(self):
        """VET/ENTG both had exactly 152,800,000 shares — an honest
        coincidence at n=2, even though the prices diverge."""
        rows = _rows(10) + _family(152_800_000, [10.24, 87.50], prefix='CO')
        summary = apply_mcap_integrity_guard(rows)
        assert summary['contaminated_nulled'] == 0
        assert all(r['mcap'] for r in rows[10:])

    def test_near_miss_vintage_is_absorbed_into_the_cluster(self):
        """FREJO carried 3,221,309,952 — a different vintage of the parent
        count, 2e-5 away from the 3,221,329,920 cluster it belongs to."""
        rows = (_rows(10) + _family(3_221_329_920, [7.4, 12.08, 15.61])
                + _family(3_221_309_952, [15.58], prefix='FREJO'))
        summary = apply_mcap_integrity_guard(rows)
        assert summary['contaminated_nulled'] == 4
        assert rows[-1]['shares_out'] is None
        assert rows[-1]['_shares_contaminated'] == 3_221_309_952

    def test_prior_snapshot_cannot_resurrect_contaminated_rows(self):
        """The prior snapshot carries the same poisoned count at a cap below
        MCAP_MAX_PLAUSIBLE — recovery must not undo the nulling."""
        rows = _rows(10) + _family(5_738_840_064, [7.35, 14.75, 15.65])
        prior = [{'ticker': f'PFD{i}', 'shares_out': 5_738_840_064.0,
                  'mcap': 5_738_840_064.0 * p}
                 for i, p in enumerate([7.35, 14.75, 15.65])]
        summary = apply_mcap_integrity_guard(rows, prior)
        assert summary['recovered'] == 0
        for r in rows[10:]:
            assert r['mcap'] is None
            assert '_mcap_source' not in r

    def test_contaminated_rows_do_not_trip_the_miss_rate_alert(self):
        """Deliberate nulls are reported separately, not as fetch failures."""
        rows = _rows(20) + _family(5_738_840_064, [7.35, 14.75, 15.65])
        summary = apply_mcap_integrity_guard(rows, threshold=0.02)
        assert summary['contaminated_nulled'] == 3
        assert summary['still_missing'] == 0
        assert summary['alert'] is False
