"""Tiingo data client for news and EOD price history.

Provides two capabilities:
  1. Per-ticker news — aggregated from thousands of sources, with optional
     sentiment scores. Used as the primary news source, replacing the
     patchier yfinance news feed.
  2. EOD price history — clean split/dividend-adjusted closes. Used as a
     fallback when yfinance returns insufficient price data for beta
     calculation.

Requires TIINGO_API_KEY environment variable (or pass via constructor).
Falls back gracefully when unavailable. Uses only stdlib + pandas.
"""

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

import pandas as pd

# VADER — lazy-loaded so import cost is zero when sentiment unused
_vader_sid = None

def _vader_score(text):
    """Return VADER compound score [-1, 1] for text, or None on failure."""
    global _vader_sid
    if not text:
        return None
    try:
        if _vader_sid is None:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            import nltk
            try:
                _vader_sid = SentimentIntensityAnalyzer()
            except LookupError:
                nltk.download('vader_lexicon', quiet=True)
                _vader_sid = SentimentIntensityAnalyzer()
        return _vader_sid.polarity_scores(text)['compound']
    except Exception:
        return None


class TiingoClient:
    _BASE_URL = 'https://api.tiingo.com'

    def __init__(self, api_key=None, request_delay=0.5):
        self._api_key = api_key or os.environ.get('TIINGO_API_KEY', '')
        self._delay = request_delay
        self._last_req = 0
        self._news_cache = {}    # ticker -> list[dict]
        self._price_cache = {}   # ticker -> pd.Series

    @property
    def available(self):
        return bool(self._api_key)

    def _throttle(self):
        elapsed = time.time() - self._last_req
        if elapsed < self._delay:
            time.sleep(self._delay - elapsed)
        self._last_req = time.time()

    def _get(self, path, params=None, timeout=10):
        """GET request to Tiingo REST API. Returns parsed JSON or None."""
        if not self._api_key:
            return None
        self._throttle()
        query = dict(params or {})
        query['token'] = self._api_key
        url = f'{self._BASE_URL}{path}?{urllib.parse.urlencode(query)}'
        try:
            req = urllib.request.Request(url, headers={
                'Content-Type': 'application/json',
                'User-Agent': 'StockAnalyzer/1.0',
            })
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 401:
                print(f'Tiingo: invalid API key (401) — disabling.')
                self._api_key = ''
            return None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # News
    # ------------------------------------------------------------------

    def fetch_ticker_news(self, ticker, max_age_days=30, max_items=10):
        """Fetch recent news for a ticker from Tiingo News API.

        Args:
            ticker: Stock ticker symbol.
            max_age_days: Ignore articles older than this.
            max_items: Maximum articles to return.

        Returns:
            list[dict] with keys: title, source, link, date, timestamp, origin.
            Empty list if unavailable.
        """
        if ticker in self._news_cache:
            return self._news_cache[ticker]

        if not self._api_key:
            self._news_cache[ticker] = []
            return []

        start_date = (datetime.now(timezone.utc) - timedelta(days=max_age_days)
                      ).strftime('%Y-%m-%d')
        data = self._get('/tiingo/news', params={
            'tickers': ticker,
            'startDate': start_date,
            'limit': max_items * 2,
        })

        if not data or not isinstance(data, list):
            self._news_cache[ticker] = []
            return []

        cutoff = time.time() - (max_age_days * 86400)
        headlines = []
        for item in data:
            pub = item.get('publishedDate', '')
            ts = 0
            date_str = ''
            if pub:
                try:
                    dt = datetime.fromisoformat(pub.replace('Z', '+00:00'))
                    ts = dt.timestamp()
                    date_str = dt.strftime('%Y-%m-%d')
                except Exception:
                    pass
            if ts < cutoff:
                continue
            title = item.get('title', '')
            description = item.get('description') or ''
            # Score title + description; fall back to title-only if desc empty
            score_text = f'{title}. {description}'.strip('. ') if description else title
            sentiment = _vader_score(score_text)
            headlines.append({
                'title': title,
                'source': item.get('source', ''),
                'link': item.get('url', ''),
                'date': date_str,
                'timestamp': ts,
                'origin': 'tiingo',
                'sentiment': round(sentiment, 3) if sentiment is not None else None,
            })

        headlines.sort(key=lambda h: h.get('timestamp', 0), reverse=True)
        result = headlines[:max_items]
        self._news_cache[ticker] = result
        return result

    def fetch_ticker_sentiment(self, ticker, max_age_days=30, max_items=10):
        """Return aggregate VADER sentiment over recent Tiingo headlines.

        Args:
            ticker: Stock ticker symbol.
            max_age_days: Lookback window for articles.
            max_items: Maximum articles to score.

        Returns:
            dict with keys:
                score (float|None): Mean compound score in [-1, 1].
                label (str|None): 'Positive', 'Neutral', or 'Negative'.
                bullish_pct (float|None): Fraction of articles with score > 0.05.
                bearish_pct (float|None): Fraction of articles with score < -0.05.
                article_count (int): Number of articles scored.
        """
        headlines = self.fetch_ticker_news(ticker, max_age_days=max_age_days,
                                           max_items=max_items)
        scores = [h['sentiment'] for h in headlines if h.get('sentiment') is not None]
        if not scores:
            return {'score': None, 'label': None, 'bullish_pct': None,
                    'bearish_pct': None, 'article_count': 0}

        avg = sum(scores) / len(scores)
        bullish = sum(1 for s in scores if s > 0.05) / len(scores)
        bearish = sum(1 for s in scores if s < -0.05) / len(scores)
        label = 'Positive' if avg > 0.05 else ('Negative' if avg < -0.05 else 'Neutral')
        return {
            'score': round(avg, 3),
            'label': label,
            'bullish_pct': round(bullish, 3),
            'bearish_pct': round(bearish, 3),
            'article_count': len(scores),
        }

    # ------------------------------------------------------------------
    # EOD price history
    # ------------------------------------------------------------------

    def fetch_history(self, ticker, period='5y'):
        """Fetch adjusted EOD close prices from Tiingo.

        Args:
            ticker: Stock ticker symbol (or index like 'SPY').
            period: Lookback period string — '1y', '2y', '5y', '10y'.

        Returns:
            pd.Series indexed by date with adjusted close prices,
            or None if unavailable.
        """
        cache_key = (ticker, period)
        if cache_key in self._price_cache:
            return self._price_cache[cache_key]

        if not self._api_key:
            return None

        years = {'1y': 1, '2y': 2, '5y': 5, '10y': 10}.get(period, 5)
        start = (datetime.now(timezone.utc) - timedelta(days=years * 365)
                 ).strftime('%Y-%m-%d')

        data = self._get(
            f'/tiingo/daily/{ticker.lower()}/prices',
            params={'startDate': start, 'resampleFreq': 'daily'},
        )

        if not data or not isinstance(data, list):
            self._price_cache[cache_key] = None
            return None

        try:
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
            df = df.set_index('date').sort_index()
            col = 'adjClose' if 'adjClose' in df.columns else 'close'
            series = df[col].dropna()
            if len(series) < 60:
                self._price_cache[cache_key] = None
                return None
            self._price_cache[cache_key] = series
            return series
        except Exception:
            self._price_cache[cache_key] = None
            return None
