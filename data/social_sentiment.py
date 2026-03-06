# data/social_sentiment.py
"""
Social media sentiment from two free, no-auth-required sources:

1. StockTwits public API  — explicit user-labeled Bullish/Bearish messages
   Endpoint: https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json

2. Reddit public JSON API — post titles from r/stocks, r/wallstreetbets,
   r/investing scored with VADER

Combines into a composite social sentiment dict.
"""
import time
import requests
from functools import lru_cache

# VADER — lazy-loaded (shared with data/sentiment.py if both imported)
_sid = None

def _get_analyzer():
    global _sid
    if _sid is not None:
        return _sid
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
    except ImportError:
        return None
    import nltk
    try:
        _sid = SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
        _sid = SentimentIntensityAnalyzer()
    return _sid


# ---------------------------------------------------------------------------
# StockTwits
# ---------------------------------------------------------------------------

STOCKTWITS_URL = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
_ST_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; stock-research-bot/1.0)"}


def fetch_stocktwits(ticker: str, timeout: int = 8) -> dict:
    """
    Pull up to 30 recent StockTwits messages for ticker.
    Users explicitly label messages Bullish, Bearish, or unlabeled.

    Returns:
      bull_count, bear_count, neutral_count, total_labeled,
      bull_pct, bear_pct, message_count
    """
    url = STOCKTWITS_URL.format(ticker=ticker)
    try:
        resp = requests.get(url, headers=_ST_HEADERS, timeout=timeout)
        if resp.status_code == 429:
            time.sleep(2)
            resp = requests.get(url, headers=_ST_HEADERS, timeout=timeout)
        if resp.status_code != 200:
            return _empty_st()
        data = resp.json()
        messages = (data.get('messages') or [])
    except Exception:
        return _empty_st()

    bull, bear, neutral = 0, 0, 0
    for msg in messages:
        entities = msg.get('entities') or {}
        sentiment = entities.get('sentiment')
        if sentiment is None:
            neutral += 1
        elif sentiment.get('basic') == 'Bullish':
            bull += 1
        elif sentiment.get('basic') == 'Bearish':
            bear += 1
        else:
            neutral += 1

    labeled = bull + bear
    total = bull + bear + neutral
    return {
        'st_bull': bull,
        'st_bear': bear,
        'st_neutral': neutral,
        'st_total': total,
        'st_bull_pct': round(bull / labeled, 4) if labeled > 0 else None,
        'st_bear_pct': round(bear / labeled, 4) if labeled > 0 else None,
        'st_labeled': labeled,
    }


def _empty_st():
    return {
        'st_bull': 0, 'st_bear': 0, 'st_neutral': 0,
        'st_total': 0, 'st_bull_pct': None, 'st_bear_pct': None,
        'st_labeled': 0,
    }


# ---------------------------------------------------------------------------
# Reddit
# ---------------------------------------------------------------------------

REDDIT_SEARCH = "https://www.reddit.com/search.json"
SUBREDDIT_SEARCH = "https://www.reddit.com/r/{sub}/search.json"
_REDDIT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; stock-research-bot/1.0)",
    "Accept": "application/json",
}
_SUBREDDITS = ["stocks", "wallstreetbets", "investing"]


def fetch_reddit(ticker: str, max_posts: int = 15, timeout: int = 8) -> dict:
    """
    Search r/stocks, r/wallstreetbets, r/investing for the ticker symbol.
    Scores post titles with VADER.

    Returns:
      reddit_score, reddit_label, reddit_posts, reddit_bull_pct, reddit_bear_pct
    """
    sid = _get_analyzer()
    titles = []

    for sub in _SUBREDDITS:
        try:
            resp = requests.get(
                SUBREDDIT_SEARCH.format(sub=sub),
                params={'q': ticker, 'restrict_sr': 1, 'sort': 'new',
                        'limit': max_posts, 't': 'week'},
                headers=_REDDIT_HEADERS,
                timeout=timeout,
            )
            if resp.status_code == 429:
                time.sleep(3)
                continue
            if resp.status_code != 200:
                continue
            data = resp.json()
            children = (data.get('data') or {}).get('children') or []
            for child in children:
                title = (child.get('data') or {}).get('title') or ''
                if title and ticker.upper() in title.upper():
                    titles.append(title)
        except Exception:
            continue

    if not titles or sid is None:
        return _empty_reddit()

    scores = [sid.polarity_scores(t)['compound'] for t in titles]
    avg = sum(scores) / len(scores)
    bull_pct = sum(1 for s in scores if s > 0.05) / len(scores)
    bear_pct = sum(1 for s in scores if s < -0.05) / len(scores)

    if avg > 0.05:
        label = 'Positive'
    elif avg < -0.05:
        label = 'Negative'
    else:
        label = 'Neutral'

    return {
        'reddit_score': round(avg, 4),
        'reddit_label': label,
        'reddit_posts': len(scores),
        'reddit_bull_pct': round(bull_pct, 4),
        'reddit_bear_pct': round(bear_pct, 4),
    }


def _empty_reddit():
    return {
        'reddit_score': None, 'reddit_label': None,
        'reddit_posts': 0, 'reddit_bull_pct': None, 'reddit_bear_pct': None,
    }


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------

def fetch_social_sentiment(ticker: str) -> dict:
    """
    Aggregate StockTwits + Reddit into a single social sentiment result.

    Composite score:
      - If both available: weighted avg (StockTwits 60%, Reddit 40%)
        where StockTwits score = st_bull_pct - st_bear_pct (net bull ratio)
      - If only one: use that source
      - Label: Positive > 0.05, Negative < -0.05, else Neutral

    Returns the raw sub-scores plus composite_score and composite_label.
    """
    st = fetch_stocktwits(ticker)
    rd = fetch_reddit(ticker)

    # Convert StockTwits net bull ratio to [-1, 1] scale
    st_score = None
    if st['st_bull_pct'] is not None and st['st_bear_pct'] is not None:
        st_score = st['st_bull_pct'] - st['st_bear_pct']

    rd_score = rd.get('reddit_score')

    if st_score is not None and rd_score is not None:
        composite = st_score * 0.6 + rd_score * 0.4
    elif st_score is not None:
        composite = st_score
    elif rd_score is not None:
        composite = rd_score
    else:
        composite = None

    if composite is not None:
        if composite > 0.05:
            composite_label = 'Positive'
        elif composite < -0.05:
            composite_label = 'Negative'
        else:
            composite_label = 'Neutral'
    else:
        composite_label = None

    return {
        **st,
        **rd,
        'social_score': round(composite, 4) if composite is not None else None,
        'social_label': composite_label,
    }
