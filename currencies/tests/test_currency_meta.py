"""Universe identifier mapping.

The euro dedupe is the test that matters here: twenty sovereign countries
share one currency and one BIS series, and a regression would silently
produce twenty EUR rows.
"""

from data import currency_meta as cm


def test_euro_resolves_to_exactly_one_row():
    assert cm.CODES.count('EUR') == 1


def test_euro_maps_to_bis_synthetic_area():
    assert cm.bis_area('EUR') == 'XM'


def test_all_eurozone_iso3s_reverse_to_eur():
    eur = cm.get('EUR')
    assert len(eur['map_iso3']) == 20
    for iso3 in eur['map_iso3']:
        assert cm.ISO3_TO_CODE[iso3] == 'EUR'


def test_bis_areas_are_deduplicated():
    areas = cm.all_bis_areas()
    assert len(areas) == len(set(areas))
    # One entry per currency, because XM already collapses the eurozone.
    assert len(areas) == len(cm.CODES)


def test_every_currency_has_complete_metadata():
    for code in cm.CODES:
        m = cm.get(code)
        assert m['bis_area'], code
        assert m['map_iso3'], code
        assert m['regime'] in ('float', 'managed', 'peg'), code
        assert m['wb_code'], code
        assert isinstance(m['has_policy_rate'], bool), code


def test_codes_are_unique():
    assert len(cm.CODES) == len(set(cm.CODES))


def test_iso3s_are_not_shared_between_currencies():
    seen = {}
    for code in cm.CODES:
        for iso3 in cm.get(code)['map_iso3']:
            assert iso3 not in seen, '%s claimed by %s and %s' % (
                iso3, seen.get(iso3), code)
            seen[iso3] = code


def test_singapore_has_no_policy_rate():
    # MAS runs exchange-rate policy, not an interest-rate target, so BIS
    # publishes no CBPOL series. This is structural, not missing data.
    assert cm.has_policy_rate('SGD') is False
    assert cm.has_policy_rate('USD') is True


def test_peg_classification():
    assert cm.is_pegged('HKD')
    assert cm.is_pegged('DKK')
    assert not cm.is_pegged('JPY')
    # A managed float is not a peg, but both count as managed.
    assert cm.is_managed('CNY')
    assert not cm.is_pegged('CNY')


def test_euro_uses_world_bank_aggregate():
    # Germany's current account is not the euro area's: intra-bloc trade nets
    # out of the aggregate but not out of any member.
    assert cm.wb_code('EUR') == 'EMU'
    assert cm.wb_code('JPY') == 'JPN'


def test_base_currency_is_in_universe():
    assert cm.BASE_CURRENCY in cm.CODES
