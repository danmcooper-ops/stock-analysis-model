# tests/test_mcap_guard.py

import pytest


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
                           'implausible_nulled': 0}

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
