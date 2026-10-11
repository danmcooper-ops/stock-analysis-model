# tests/test_sector_pool_narrative.py
"""Guard: the sector profit-pool prose names real companies and real numbers,
and states each figure once.

  * `|OM| <= 100%` was the only filter on who could be named the sector's
    best or worst operator, which let a $865M company (0.02% of a $3.8T
    pool) outrank NVDA as Technology's "Margin Leader" (live 2026-09-05).
  * The `overview` paragraph is the sector page's "The Sector Today". The
    loss-maker count, concentration and margin spread are the Profit pool
    tab's tiles, so the prose no longer restates them; the strategy bullets
    that did (`insights`) and the HHI paragraph (`concentration`) are gone,
    and the one bullet that said something new — what the blended margin
    means — now follows the sentence that states it.

Neither failure is loud — the page renders either way. So pin them.
"""
import models.narrative as narrative
from models.narrative import generate_sector_profit_pool_narrative as gen


def _row(ticker, rev, oi, rev_share, profit_share, **kw):
    """One pool row. pp_multiple mirrors the pipeline: profit/revenue share,
    with the profit numerator clamped at zero, so loss-makers land on 0."""
    d = {
        'ticker': ticker,
        'company_name': ticker + ' Inc',
        'revenue': rev,
        'operating_income': oi,
        'operating_margin': oi / rev,
        'pp_revenue_share': rev_share,
        'pp_profit_share': profit_share,
        'pp_multiple': (profit_share / rev_share) if rev_share else None,
        'pp_margin_advantage': 0.0,
        'pp_sector_hhi': 0.12,
        'pp_sector_cr4': 0.40,
        'rating': 'HOLD',
    }
    d.update(kw)
    return d


def _sector(extra=()):
    """A pool whose big names carry the sector and whose tail is noise.

    Six material rows, so the thin-sector fallback (fewer than five) does not
    fire and the materiality floor is what the tests actually exercise.
    """
    rows = [
        _row('BIGA', 30_000, 11_000, 0.30, 0.55),   # material, efficient: 1.83x
        _row('BIGB', 30_000, 5_600, 0.30, 0.28),
        _row('BIGC', 20_000, 3_200, 0.20, 0.16),
        _row('BIGD', 10_000, 200, 0.10, 0.01),      # material, barely profitable
        _row('BIGE', 5_000, 400, 0.05, 0.02),
        _row('LOSSY', 9_000, -400, 0.09, 0.0),      # the big loss-maker
    ]
    rows.extend(extra)
    return rows


def test_immaterial_companies_cannot_be_named_sector_exemplars():
    """A rounding-error company must not be held up as the sector's best or
    worst operator while the mega-caps go unmentioned."""
    # 60% operating margin on 0.02% of sector revenue — sane by |OM|, absurd
    # as "the sector's fattest margin".
    speck = _row('SPECK', 20, 12, 0.0002, 0.0005)
    n = gen('Technology', _sector(extra=[speck]))
    roles = {r['role_label']: p['ticker']
             for p in n['key_players'] for r in p['roles']}
    assert roles.get('Margin Leader') != 'SPECK'
    assert roles.get('Efficiency Leader') != 'SPECK'
    assert 'SPECK' not in n['overview']


def test_a_thin_sector_keeps_its_storyline():
    """The floor must not silence a sector too small to clear it — better a
    small-cap exemplar than no ranking at all."""
    thin = [
        _row('T1', 60, 30, 0.0006, 0.0009),
        _row('T2', 40, 4, 0.0004, 0.0001),
    ]
    n = gen('Utilities', thin)
    assert n is not None
    roles = {r['role_label'] for p in n['key_players'] for r in p['roles']}
    assert 'Margin Leader' in roles, 'thin sectors still get ranked'


def test_the_materiality_floor_is_a_named_constant():
    assert hasattr(narrative, '_MIN_MATERIAL_REV_SHARE')
    # low enough to keep genuine mid-caps, high enough to drop the noise tail
    assert 0 < narrative._MIN_MATERIAL_REV_SHARE <= 0.005


def _margin_sector(margin):
    """Three equal companies at one operating margin."""
    return [_row('M%d' % i, 1_000, 1_000 * margin, 1 / 3,
                 (1 / 3) if margin > 0 else 0.0) for i in range(3)]


def test_overview_states_size_margin_and_leader_not_the_tiles():
    n = gen('Technology', _sector())
    ov = n['overview']
    assert 'blended operating margin of' in ov
    assert 'BIGA alone captures' in ov
    # the Loss-makers and Concentration tiles own these
    assert 'operating at a loss' not in ov
    assert 'HHI' not in ov
    for gone in ('insights', 'concentration'):
        assert gone not in n, gone


def test_margin_verdict_follows_the_margin_sentence():
    """The verdict says what the blended margin means, directly after the
    sentence that states it, and only at the ends of the range."""
    hi = gen('Technology', _margin_sector(0.30))['overview']
    assert 'rarefied air' in hi
    assert hi.index('blended operating margin') < hi.index('That margin') < hi.index('alone captures')
    assert 'no cushion' in gen('Energy', _margin_sector(0.05))['overview']
    assert 'destroying capital' in gen('Energy', _margin_sector(-0.05))['overview']
    mid = gen('Industrials', _margin_sector(0.12))['overview']
    for verdict in ('rarefied air', 'no cushion', 'destroying capital'):
        assert verdict not in mid
