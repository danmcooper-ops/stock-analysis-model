# tests/test_sector_pool_narrative.py
"""Guard: the sector profit-pool prose names real companies and real numbers.

Two defects motivated these, both visible on the live 2026-09-05 run:

  * the efficiency bullet printed a quotient that was not a measurement.
    `pp_profit_share` clamps its numerator at `max(opinc, 0)`, so every
    loss-making company has a multiple of exactly 0 and the bottom of the
    ranking is always one of them. The code divided by `max(worst, 0.01)`,
    making the printed figure `best * 100` — a function of that floor
    constant, not of the company it named. Technology read "262.3x more
    efficiently than SYNA"; 262.3 is 2.621 * 100, and SYNA's own multiple
    never entered the arithmetic.
  * `|OM| <= 100%` was the only filter on who could be named the sector's
    best or worst operator, which let a $865M company (0.02% of a $3.8T
    pool) outrank NVDA as Technology's "Margin Leader".

Neither fails loudly — the page renders either way, just with a fabricated
number and an unrecognisable company. So pin them.
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


def test_efficiency_bullet_never_prints_a_quotient():
    """The best/worst ratio is gone; both multiples are stated instead."""
    src = open(narrative.__file__, encoding='utf-8').read()
    assert 'max(worst_mult, 0.01)' not in src, \
        'the floor-divided quotient is back'
    n = gen('Technology', _sector())
    eff = [b for b in n['insights'] if 'share of sector revenue in profit' in b]
    assert eff, 'the efficiency bullet should fire on this pool'
    assert 'x more efficiently' not in eff[0], 'no quotient in the prose'
    # the leader's own multiple, stated directly
    assert '1.83x' in eff[0]


def test_a_loss_making_worst_is_described_not_divided_by():
    """A company at 0 has no ratio to quote; say what is actually true."""
    n = gen('Technology', _sector())
    eff = [b for b in n['insights'] if 'share of sector revenue in profit' in b][0]
    assert 'operating at a loss' in eff
    assert 'takes none of it' in eff


def test_the_loss_maker_named_is_the_largest_not_an_arbitrary_tie():
    """Every loss-maker sits at exactly 0, so the tail of the ranking is a
    tie. Name the biggest of them, deterministically — not whichever sorted
    last."""
    tiny_losers = [_row('TINY%d' % i, 200, -80, 0.002, 0.0) for i in range(4)]
    n = gen('Technology', _sector(extra=tiny_losers))
    eff = [b for b in n['insights'] if 'share of sector revenue in profit' in b][0]
    assert 'LOSSY' in eff, 'the largest loss-maker should be the one named'
    for t in ('TINY0', 'TINY1', 'TINY2', 'TINY3'):
        assert t not in eff


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
    for bullet in n['insights']:
        assert 'SPECK' not in bullet


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
