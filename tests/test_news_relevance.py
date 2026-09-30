"""Entity binding and ordering for news headlines.

The central regression test is ``test_company_item_beats_newer_sector_item``:
the shipped bug was that a merged feed sorted newest-first, and sector items
are always fresher than per-ticker ones.
"""
from data.news_relevance import (
    TIER_NAME,
    TIER_SECTOR,
    TIER_SOURCE,
    TIER_SYMBOL,
    build_news_query,
    company_aliases,
    order_headlines,
    relevance_tier,
)


def _h(title, origin='yfinance', ts=0.0):
    return {'title': title, 'origin': origin, 'timestamp': ts}


# --- alias extraction -------------------------------------------------------

def test_legal_suffix_stripped_for_core_but_kept_in_full():
    aliases, core, full = company_aliases('Apple Inc.', 'AAPL')
    assert core == 'apple'
    assert full == 'apple inc'
    assert aliases == {'apple', 'apple inc'}


def test_parenthetical_dropped_without_losing_the_suffix():
    # 'Gap, Inc. (The)' must not truncate at the comma — 'inc' is the
    # disambiguator that makes the query usable.
    _, core, full = company_aliases('Gap, Inc. (The)', 'GAP')
    assert core == 'gap'
    assert full == 'gap inc'


def test_accents_and_punctuation_normalised():
    _, core, _ = company_aliases('Anheuser-Busch InBev SA/NV', 'BUD')
    assert core.startswith('anheuser-busch')


def test_multiword_core_keeps_all_tokens():
    _, core, _ = company_aliases('Bank of America Corporation', 'BAC')
    assert core == 'bank of america'


def test_numeric_name_yields_no_aliases():
    aliases, core, _ = company_aliases('70469', 'WILCF')
    assert core == '70469'
    assert aliases == frozenset()


# --- query construction -----------------------------------------------------

def test_query_prefers_the_suffixed_name():
    assert build_news_query('Gap, Inc. (The)', 'GAP') == 'gap inc stock'
    assert build_news_query('On Holding AG', 'ONON') == 'on holding ag stock'


def test_query_adds_ticker_when_there_is_no_suffix():
    assert build_news_query('First', 'FCF') == 'first FCF stock'


def test_query_falls_back_to_ticker_for_unusable_names():
    assert build_news_query('RH', 'RH') == 'RH stock'          # initials-length
    assert build_news_query('70469', 'WILCF') == 'WILCF stock'  # numeric
    assert build_news_query('', 'MSFT') == 'MSFT stock'         # absent


def test_query_is_none_when_nothing_identifies_the_company():
    # None must mean "make no request", never "fall back to a sector search".
    assert build_news_query('', '') is None
    assert build_news_query(None, None) is None


# --- tiering ----------------------------------------------------------------

def test_tier_order_symbol_name_source_sector():
    aliases, _, _ = company_aliases('Apple Inc.', 'AAPL')
    assert relevance_tier(_h('AAPL slips on iPhone demand'), 'AAPL', aliases) == TIER_SYMBOL
    assert relevance_tier(_h('Apple slips on iPhone demand'), 'AAPL', aliases) == TIER_NAME
    assert relevance_tier(_h('Chip supply chain wobbles'), 'AAPL', aliases) == TIER_SOURCE
    assert relevance_tier(
        _h('Technology stocks rally', origin='google_news'), 'AAPL', aliases) == TIER_SECTOR


def test_ticker_match_requires_a_token_boundary():
    aliases, _, _ = company_aliases('Apple Inc.', 'AAPL')
    # 'AAPL' inside a longer token is not a mention.
    assert relevance_tier(_h('XAAPLY index rebalances'), 'AAPL', aliases) != TIER_SYMBOL


def test_short_alias_cannot_promote_sector_copy():
    """'Gap', 'On', 'Sea' are ordinary words — sector items need strong evidence."""
    aliases, _, _ = company_aliases('Gap, Inc. (The)', 'GAP')
    sector = _h('Mind the gap between growth and value stocks', origin='google_news')
    assert relevance_tier(sector, 'GAP', aliases) == TIER_SECTOR
    # The full, multi-word alias still counts.
    strong = _h('Gap Inc raises full-year outlook', origin='google_news')
    assert relevance_tier(strong, 'GAP', aliases) == TIER_NAME


def test_sector_ticker_match_requires_upper_case():
    """A symbol is written in caps; the common noun that shares it is not."""
    aliases, _, _ = company_aliases('Gap, Inc. (The)', 'GAP')
    quoted = _h('Retail movers: GAP jumps 6%', origin='google_news')
    assert relevance_tier(quoted, 'GAP', aliases) == TIER_SYMBOL
    # ...but a per-ticker source is trusted to mean the company either way.
    assert relevance_tier(_h('gap widens'), 'GAP', aliases) == TIER_SYMBOL


# --- ordering ---------------------------------------------------------------

def test_company_item_beats_newer_sector_item():
    """The shipped regression: recency used to promote generic sector copy."""
    items = [
        _h('Technology stocks rally', origin='google_news', ts=2_000_000),
        _h('Apple beats on Q3 earnings', origin='yfinance', ts=1_000_000),
    ]
    out = order_headlines(items, 'AAPL', 'Apple Inc.')
    assert out[0]['title'] == 'Apple beats on Q3 earnings'
    assert out[0]['scope'] == 'company'
    assert out[1]['scope'] == 'sector'


def test_sector_items_are_capped_and_company_items_lead():
    items = [_h(f'Sector story {i}', origin='google_news', ts=9_000_000 + i) for i in range(8)]
    items += [_h(f'Apple story {i}', origin='yfinance', ts=i) for i in range(5)]
    out = order_headlines(items, 'AAPL', 'Apple Inc.', max_total=12, max_sector=3)
    assert [h['scope'] for h in out[:5]] == ['company'] * 5
    assert sum(1 for h in out if h['scope'] == 'sector') == 3


def test_total_cap_respected():
    items = [_h(f'Apple story {i}', origin='yfinance', ts=i) for i in range(30)]
    assert len(order_headlines(items, 'AAPL', 'Apple Inc.', max_total=12)) == 12


def test_dedupe_ignores_trailing_publisher():
    items = [
        _h('Apple beats on earnings', origin='yfinance', ts=2),
        _h('Apple beats on earnings - Reuters', origin='google_news_ticker', ts=1),
    ]
    assert len(order_headlines(items, 'AAPL', 'Apple Inc.')) == 1


def test_untitled_items_dropped():
    assert order_headlines([_h(''), _h(None)], 'AAPL', 'Apple Inc.') == []


def test_tags_attached_during_ordering():
    out = order_headlines([_h('Apple raises guidance')], 'AAPL', 'Apple Inc.')
    assert 'guidance' in out[0]['tags']


def test_sector_dicts_are_not_shared_between_tickers():
    """One sector list is served to every ticker in that sector.

    Stamping tier/scope onto the fetched dict let each ticker overwrite the
    previous one's answer, and because the snapshot holds references until it
    is written, every row in a sector ended up with whatever the last ticker
    decided — visible whenever a sector headline names one company.
    """
    shared = [{'title': 'AAPL leads chip stocks higher',
               'origin': 'google_news', 'timestamp': 100}]
    a = order_headlines(list(shared), 'AAPL', 'Apple Inc.')
    m = order_headlines(list(shared), 'MSFT', 'Microsoft Corporation')
    assert a[0]['scope'] == 'company' and a[0]['tier'] == TIER_SYMBOL
    assert m[0]['scope'] == 'sector' and m[0]['tier'] == TIER_SECTOR
    assert a[0] is not m[0]
    # The source dict must come back untouched, or the next caller inherits it.
    assert 'tier' not in shared[0] and 'scope' not in shared[0]


def test_ordering_is_threadsafe_over_a_shared_sector_list():
    """The Phase-2 prefetch pool runs this from 4 threads over one list."""
    import concurrent.futures as cf
    shared = [{'title': f'Sector story {i}', 'origin': 'google_news',
               'timestamp': i} for i in range(8)]
    tickers = [f'T{i}' for i in range(40)]
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        out = list(ex.map(
            lambda t: order_headlines(list(shared), t, f'Company {t} Inc.'), tickers))
    assert all(h['scope'] == 'sector' for rows in out for h in rows)
    assert all('tier' not in h for h in shared)
