"""Deterministic event tags for news headlines.

One keyword table for every headline source, applied at a single point
(:func:`data.news_relevance.order_headlines`) so Tiingo, yfinance and both
Google News queries are tagged identically.

Matching is plain case-folded substring containment — the idiom already used
by the layoff/culture pass this module replaces, and deliberately *not*
regex: ``redundan``, ``downsiz`` and ``restructur`` are prefixes chosen to
catch a whole family of words, and a word-boundary rewrite would silently
change what they match.

``layoffs`` and ``culture_award`` are copied verbatim from the sets that used
to live in ``scripts/analyze_stock.py``. They derive ``layoff_news_signal``
(a typed column in ``core.results``) and ``culture_award_signal``, which feed
``models/narrative.py`` and ``models/data_tab_narrative.py`` — so an edit to
those two tuples is a schema-behaviour change, not a wording change.
Measured on the 2026-09-14 corpus (26,904 headlines): the table tags 32.8% of
company headlines and 16.7% of sector ones.
"""

# The two legacy sets, named so the equivalence test can address them
# directly. Do not reorder or reword.
LAYOFF_KEYWORDS = (
    'layoff', 'lay off', 'laid off', 'job cut', 'workforce reduction',
    'redundan', 'downsiz', 'restructur', 'reorg',
)
CULTURE_POS_KEYWORDS = (
    'best place', 'top employer', 'great place to work',
    'best company', 'culture award',
)

# Declaration order is display order: the report shows at most the first two
# chips per headline, so the tags that change a thesis come first.
TAG_KEYWORDS = {
    'earnings': (
        'earnings', 'q1 ', 'q2 ', 'q3 ', 'q4 ', 'quarter', 'beats', 'misses',
        'eps', 'revenue', 'results',
    ),
    'guidance': (
        'guidance', 'outlook', 'forecast', 'raises', 'cuts view', 'warns',
    ),
    'm_and_a': (
        'acquir', 'merger', 'merges', 'takeover', 'buyout', 'to buy',
        'stake in', 'divest', 'spin-off', 'spinoff',
    ),
    'legal': (
        'lawsuit', 'sues', 'settlement', 'probe', 'investigat',
        'class action', 'subpoena', 'fraud',
    ),
    'regulatory': (
        'fda', 'sec charges', 'antitrust', 'regulat', 'approval', 'ruling',
        'tariff', 'doj',
    ),
    'analyst_action': (
        'upgrade', 'downgrade', 'price target', 'initiated', 'buy rating',
        'sell rating', 'overweight', 'neutral',
    ),
    'layoffs': LAYOFF_KEYWORDS,
    'leadership': (
        'ceo', 'cfo', 'steps down', 'resign', 'appoints', 'names new',
    ),
    'dividend': ('dividend', 'buyback', 'repurchase', 'split'),
    'product': (
        'launch', 'unveil', 'announces new', 'partnership', 'contract',
        'deal with', 'rollout',
    ),
    'culture_award': CULTURE_POS_KEYWORDS,
}

# Short enough to sit inline on a headline row without wrapping it.
TAG_LABELS = {
    'earnings': 'Earnings',
    'guidance': 'Guidance',
    'm_and_a': 'M&A',
    'legal': 'Legal',
    'regulatory': 'Regulatory',
    'analyst_action': 'Analyst',
    'layoffs': 'Layoffs',
    'leadership': 'Leadership',
    'dividend': 'Dividend',
    'product': 'Product',
    'culture_award': 'Culture',
}


def tag_headline(title):
    """Return the event tags for a headline title, in display order.

    Args:
        title: Headline text (may be None or empty).

    Returns:
        tuple[str, ...] — keys of :data:`TAG_KEYWORDS`, in declaration order.
        Empty when nothing matches.
    """
    if not title:
        return ()
    low = title.lower()
    return tuple(tag for tag, words in TAG_KEYWORDS.items()
                 if any(w in low for w in words))
