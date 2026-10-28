"""Live upstream smoke tests.

Excluded from the default run (see pytest.ini). Run explicitly with:

    pytest -m live -q

These exist to catch the failure mode unit tests cannot: an upstream changing
its response shape or its content-negotiation rules. The BIS `format=csv`
requirement is the concrete example — without it every request returns 406,
and nothing offline would ever notice.
"""

import pytest

from data import bis_client, currency_meta, ecb_client, erapi_client
from data import worldbank_client as wb

pytestmark = pytest.mark.live


def test_ecb_serves_full_history():
    per_eur = ecb_client.fetch_reference_rates()
    assert per_eur, 'ECB returned nothing'
    assert 'USD' in per_eur
    # The euro's first published reference rate. A fixed historical fact:
    # if this changes, the parse or the endpoint semantics changed.
    assert per_eur['USD']['1999-01-04'] == pytest.approx(1.1789)


def test_every_universe_currency_has_live_spot():
    per_eur = ecb_client.fetch_reference_rates()
    hist = ecb_client.build_usd_histories(per_eur)
    missing = [c for c in currency_meta.CODES if c not in hist]
    assert not missing, 'no live ECB series for: %s' % missing


def test_jpy_cross_is_usd_per_yen():
    per_eur = ecb_client.fetch_reference_rates()
    jpy = ecb_client.to_usd_series(per_eur, 'JPY')
    latest = jpy[max(jpy)]
    # A yen is worth well under a cent. Catches an inverted cross instantly.
    assert 0.001 < latest < 0.05


def test_ecb_cross_agrees_with_an_independent_provider():
    """Cross-check the ECB cross against open.er-api.com.

    This is the guard against a silently inverted cross rate. The two sources
    quote in opposite directions and fix at different times of day, so an
    exact match would be the wrong thing to require — but a sign error or an
    inversion shows up as an enormous relative difference, not a 0.5% one.
    """
    per_eur = ecb_client.fetch_reference_rates()
    hist = ecb_client.build_usd_histories(per_eur)
    rates = erapi_client.fetch_usd_rates()
    assert rates, 'independent rate source unavailable'

    compared = 0
    for code in currency_meta.CODES:
        theirs = erapi_client.usd_per_unit(rates, code)
        series = hist.get(code)
        if not theirs or not series:
            continue
        ours = series[-1][1]
        rel = abs(ours - theirs) / theirs
        assert rel < 0.03, (
            '%s: model has %.8g USD/unit, independent source has %.8g '
            '(%.1f%% apart) — check the cross-rate direction'
            % (code, ours, theirs, rel * 100))
        compared += 1

    assert compared >= 25, 'only cross-checked %d currencies' % compared


def test_bis_reer_requires_csv_format_and_covers_universe():
    reer = bis_client.fetch_reer()
    assert reer, 'BIS returned nothing (406 means the format=csv param was lost)'
    missing = [c for c in currency_meta.CODES
               if not bis_client.reer_series(reer, c)]
    assert not missing, 'no BIS REER for: %s' % missing


def test_bis_reer_has_deep_history():
    reer = bis_client.fetch_reer()
    jp = bis_client.reer_series(reer, 'JPY')
    assert len(jp) >= 380, 'expected ~390 monthly observations, got %d' % len(jp)


def test_japan_screens_cheap_in_real_terms():
    """The model's headline sanity check.

    Japan's real broad effective exchange rate has roughly halved since the
    mid-1990s. If this ever reads the other way, the REER sign convention has
    been inverted somewhere and the entire Valuation pillar is backwards.
    """
    reer = bis_client.fetch_reer()
    jp = bis_client.reer_series(reer, 'JPY')
    assert jp[-1][1] < jp[0][1] * 0.6


def test_bis_policy_rates_cover_universe_except_singapore():
    pol = bis_client.fetch_policy_rates()
    assert pol
    for code in currency_meta.CODES:
        rate = bis_client.latest_policy_rate(pol, code)
        if currency_meta.has_policy_rate(code):
            assert rate is not None, 'no policy rate for %s' % code
        else:
            assert rate is None, '%s should have no policy rate' % code


def test_worldbank_core_indicators_are_reachable():
    """World Bank is the flaky source; assert the four gated indicators only.

    Government debt and external debt are deliberately excluded — they are
    genuinely sparse (external debt is reported mainly by developing
    economies) and are displayed as context rather than scored.
    """
    macro = wb.fetch_macro()
    health = wb.indicator_health(macro)
    for field in ('current_account_pct_gdp', 'reserves_months_imports',
                  'inflation_pct', 'gdp_growth_pct'):
        h = health[field]
        assert h['covered'] >= h['total'] * 0.9, (
            '%s covered only %d/%d' % (field, h['covered'], h['total']))


def test_published_site_has_the_shape_the_check_reads():
    """scripts/check_site.py reads these keys from the live Pages site.

    Also pins the derived URL: the site moved with the repository rename
    (Currency-Arbitrage- -> Currency-Screener-), and a hard-coded old URL
    is what made the cloud check report a healthy site as down.
    """
    from scripts import check_site as cs
    site = cs.site_url(cs.parse_remote(cs.FALLBACK_REMOTE))
    status, manifest = cs.get_json(site + 'manifest.json')
    assert status == 200, 'manifest.json HTTP %s at %s' % (status, site)
    assert manifest['run_date']
    prov = manifest['provenance']
    assert 'stale_slots' in prov and 'event_counts' in prov
    _, results = cs.get_json(site + 'results.json')
    jpy = [r for r in results['results'] if r['code'] == 'JPY'][0]
    assert {'rating', '_composite_score', '_score_pillar_valuation'} <= set(jpy)
