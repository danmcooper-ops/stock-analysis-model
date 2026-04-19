# data/sentiment.py
"""
Consumer sentiment analysis using yfinance recent news + VADER lexicon.

VADER (Valence Aware Dictionary and sEntiment Reasoner) is tuned for
short, informal text like news headlines and social media.

Requires: nltk (pip install nltk)
First run downloads VADER lexicon (~1MB, cached locally).
"""
import yfinance as yf

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    _vader_available = True
except ImportError:
    _vader_available = False

_sid = None  # lazy-loaded analyzer


def _get_analyzer():
    global _sid
    if _sid is not None:
        return _sid
    if not _vader_available:
        raise ImportError("nltk not installed. Run: pip install nltk")
    import nltk
    try:
        _sid = SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
        _sid = SentimentIntensityAnalyzer()
    return _sid


def fetch_sentiment(ticker: str, max_articles: int = 20) -> dict:
    """
    Fetch recent news headlines for `ticker` via yfinance and score them
    with VADER.

    Returns a dict with:
      - score       : float, avg compound score in [-1, 1]
      - label       : 'Positive' | 'Neutral' | 'Negative'
      - article_count : int, number of headlines scored
      - bullish_pct : float, fraction of positive articles
      - bearish_pct : float, fraction of negative articles
    """
    try:
        sid = _get_analyzer()
    except ImportError:
        return _empty_result()

    try:
        stock = yf.Ticker(ticker)
        news = stock.news or []
    except Exception:
        return _empty_result()

    scores = []
    for item in news[:max_articles]:
        # yfinance news items have 'content' dict with 'title'
        content = item.get('content') or {}
        title = content.get('title') or item.get('title') or ''
        if not title:
            continue
        compound = sid.polarity_scores(title)['compound']
        scores.append(compound)

    if not scores:
        return _empty_result()

    avg = sum(scores) / len(scores)
    bullish = sum(1 for s in scores if s > 0.05) / len(scores)
    bearish = sum(1 for s in scores if s < -0.05) / len(scores)

    if avg > 0.05:
        label = 'Positive'
    elif avg < -0.05:
        label = 'Negative'
    else:
        label = 'Neutral'

    return {
        'score': round(avg, 4),
        'label': label,
        'article_count': len(scores),
        'bullish_pct': round(bullish, 4),
        'bearish_pct': round(bearish, 4),
    }


def _empty_result():
    return {
        'score': None,
        'label': None,
        'article_count': 0,
        'bullish_pct': None,
        'bearish_pct': None,
    }
