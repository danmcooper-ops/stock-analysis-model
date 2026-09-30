"""Bind a headline to the company whose page it appears on, and order by that.

Pure functions, no I/O — ``data/news_client.py`` stays the fetch/parse layer so
this can be tested without network.

The problem this solves, measured on the 2026-09-14 snapshot: the per-stock
news list merged per-ticker headlines with a Google News search for
``"{sector} stocks"`` and sorted the result newest-first. Sector items are
hours old and per-ticker items span 30 days, so recency ordering promoted the
generic feed — 70% of all headlines were sector copy, the first item shown was
sector for 91% of rows, and only 0.1% of those sector items mentioned the
company at all. Ordering is therefore by *relevance tier first*, recency
second, and sector items are capped and marked as context.

Query construction keeps the legal suffix (``Gap, Inc. (The)`` -> ``Gap Inc``)
because the suffix is what disambiguates a company whose name is an ordinary
word — ``Gap Inc stock`` finds the retailer where ``gap stock`` finds nothing
useful. The suffix is stripped only to build the aliases used for *matching*,
where ``Gap`` is what a headline actually says. Terms are left unquoted so
Google can rank rather than require an exact phrase.
"""

import re
import unicodedata

from data.news_tags import tag_headline

# Stripped to form the matching alias, retained in the query.
_LEGAL_SUFFIXES = frozenset({
    'incorporated', 'inc', 'corporation', 'corp', 'company', 'co',
    'holdings', 'holding', 'group', 'plc', 'ltd', 'limited', 'llc', 'lp',
    'nv', 'sa', 'ag', 'ab', 'se', 'oyj', 'as', 'trust', 'reit', 'adr',
    'ads', 'class', 'common', 'stock', 'sab', 'de', 'cv', 'spa', 'asa',
    'bhd', 'kgaa', 'pcl',
})

# Relevance tiers — lower sorts first.
TIER_SYMBOL = 0   # the ticker appears as a standalone token
TIER_NAME = 1     # an alias appears in the title
TIER_SOURCE = 2   # came from a per-ticker source, no textual match
TIER_SECTOR = 3   # sector feed — context only

_PUBLISHER_SUFFIX = re.compile(r'\s+-\s+[^-]{1,40}$')
_NON_NAME = re.compile(r'[^a-z0-9&\- ]+')

# Origins that are per-ticker by construction, so an unmatched title is still
# plausibly about the company. 'google_news' (sector) is deliberately absent.
TICKER_ORIGINS = frozenset({'tiingo', 'yfinance', 'google_news_ticker'})


def _clean(name):
    """Casefold, strip accents and punctuation, collapse whitespace."""
    if not name:
        return ''
    # Drop parentheticals ('Gap, Inc. (The)') but never truncate at the
    # comma — that would strip the ', Inc.' the query relies on.
    s = re.sub(r'\([^)]*\)', ' ', name)
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = _NON_NAME.sub(' ', s.lower())
    return ' '.join(s.split())


def company_aliases(company_name, ticker):
    """Return (aliases, core, full) for matching and query construction.

    Args:
        company_name: Yahoo shortName/longName, may be empty.
        ticker: Stock ticker symbol.

    Returns:
        (frozenset[str], str, str) — aliases to match in headline text, the
        suffix-stripped core, and the cleaned full name (suffix retained).
    """
    full = _clean(company_name)
    toks = full.split()
    core_toks = list(toks)
    while core_toks and core_toks[-1] in _LEGAL_SUFFIXES:
        core_toks.pop()
    if core_toks and core_toks[0] == 'the':
        core_toks.pop(0)
    core = ' '.join(core_toks)
    aliases = {a for a in (core, full) if a and not a.isdigit()}
    return frozenset(aliases), core, full


def build_news_query(company_name, ticker):
    """Return the Google News query for a company, or None if unusable.

    Returning None matters: a company with no usable identity must make no
    request at all rather than silently fall back to a sector search, which
    is the failure mode this module exists to remove.
    """
    _, core, full = company_aliases(company_name, ticker)
    ticker = (ticker or '').strip()
    if full and full != core:
        # The suffix is the disambiguator — prefer it over the ticker.
        return f'{full} stock'
    # No suffix to lean on: a name that is numeric, initials-length or just
    # the ticker again carries no more signal than the ticker itself.
    weak = (not core) or core.isdigit() or len(core) <= 2 or core == ticker.lower()
    if not weak:
        return f'{core} {ticker} stock' if ticker else f'{core} stock'
    if ticker:
        return f'{ticker} stock'
    return None


def relevance_tier(headline, ticker, aliases):
    """Classify one headline's relevance to a ticker. Lower is more relevant.

    A sector item needs *strong* evidence to be treated as company news,
    because many companies are ordinary words — 'Gap', 'On', 'Sea', 'News',
    'Block'. Letting a 3-letter alias promote sector copy would rebuild the
    problem this module removes, so a sector item only counts as a match on a
    multi-word alias or one of at least 5 characters.
    """
    origin = headline.get('origin') or ''
    raw = headline.get('title') or ''
    title = raw.lower()
    from_ticker_source = origin in TICKER_ORIGINS
    tick = (ticker or '').strip()

    if tick:
        if from_ticker_source:
            # The source already vouches for the association, so match loosely.
            hit = re.search(rf'(?<![a-z0-9]){re.escape(tick.lower())}(?![a-z0-9])', title)
        else:
            # A sector item has no such vouching, and plenty of tickers are
            # ordinary words (GAP, ON, KEY, ALL). Headlines write a symbol in
            # caps — 'GAP', '$GAP', '(NYSE:GAP)' — so require that, which
            # 'Mind the gap between growth and value' cannot satisfy.
            hit = re.search(rf'(?<![A-Za-z0-9]){re.escape(tick.upper())}(?![A-Za-z0-9])', raw)
        if hit:
            return TIER_SYMBOL
    names = aliases if from_ticker_source else {
        a for a in aliases if ' ' in a or len(a) >= 5
    }
    if any(a in title for a in names):
        return TIER_NAME
    if from_ticker_source:
        return TIER_SOURCE
    return TIER_SECTOR


def _dedupe_key(title):
    """Normalize a title so 'Foo beats' and 'Foo beats - Reuters' collapse."""
    t = ' '.join((title or '').lower().split())
    return _PUBLISHER_SUFFIX.sub('', t).strip()


def order_headlines(items, ticker, company_name, max_total=12, max_sector=3):
    """Dedupe, tag, classify and order headlines for one ticker.

    Company items lead; sector items are capped at *max_sector* and marked
    ``scope='sector'`` so the report can group them as context rather than
    interleaving them by date.

    Returns:
        list[dict] — the same dicts, each gaining 'tier', 'scope' and 'tags'.
    """
    aliases, _, _ = company_aliases(company_name, ticker)
    seen = set()
    company, sector = [], []
    for h in items or []:
        key = _dedupe_key(h.get('title'))
        if not key or key in seen:
            continue
        seen.add(key)
        # Copy before stamping. One sector list is served to every ticker in
        # that sector (594 Financial Services rows share 8 dicts), so writing
        # tier/scope onto the fetched dict would let each ticker overwrite the
        # last one's answer — and the snapshot serializes these references at
        # the end of the run, so every row would end up with whatever the last
        # ticker in its sector decided. Under the Phase-2 prefetch pool it is
        # also a data race. tags are title-derived and identical either way;
        # tier and scope are not.
        h = dict(h)
        tier = relevance_tier(h, ticker, aliases)
        h['tier'] = tier
        h['scope'] = 'sector' if tier == TIER_SECTOR else 'company'
        h['tags'] = tag_headline(h.get('title'))
        (sector if tier == TIER_SECTOR else company).append(h)

    def _key(h):
        return (h['tier'], -(h.get('timestamp') or 0))

    company.sort(key=_key)
    sector.sort(key=_key)
    sector = sector[:max_sector]
    return company[:max(max_total - len(sector), 0)] + sector
