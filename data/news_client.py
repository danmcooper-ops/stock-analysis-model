"""News headline fetcher combining yfinance and Google News RSS.

Provides three headline sources:
  1. yfinance Ticker.news — per-company headlines (8-10 items)
  2. Google News RSS, per company — the stock's own feed
  3. Google News RSS, per sector — context, capped and demoted

Ordering and entity binding live in :mod:`data.news_relevance`; this module is
the fetch/parse layer. The split is what lets the ordering be tested without
network, and it is where the fix for the shipped defect lives: the merged feed
used to be sorted newest-first, and because sector items are hours old while
per-ticker items span 30 days, generic copy won the top slot on 91% of rows.

All functions are resilient: failures return empty lists, never raise.
No new pip dependencies — uses only stdlib + yfinance (already in pipeline).
"""

import logging
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from data.news_relevance import build_news_query, order_headlines
from data.throttle import Throttle
from data.yf_session import make_yf_session

logger = logging.getLogger(__name__)

_YF_SESSION = None


def _yf_session():
    """Shared curl_cffi session so every Yahoo call has a hard 15s timeout."""
    global _YF_SESSION
    if _YF_SESSION is None:
        _YF_SESSION = make_yf_session()
    return _YF_SESSION


class NewsClient:
    """Fetch news headlines from yfinance and Google News RSS."""

    def __init__(self, request_delay=1.0, max_age_days=30, delay_max=None,
                 penalty=1.5, relax_step=0.98, breaker_after=5):
        self._throttle = Throttle(request_delay)
        self._ticker_cache = {}    # ticker -> list[dict]  (yfinance)
        self._company_cache = {}   # ticker -> list[dict]  (per-company RSS)
        self._sector_cache = {}    # sector -> list[dict]
        self._max_age_days = max_age_days
        self._delay_max = delay_max
        self._penalty = penalty
        self._relax_step = relax_step
        # Google publishes no rate limit. A 429 streak stops the per-ticker
        # queries only: sector and yfinance news keep working, so the report
        # degrades to roughly its previous content rather than emptying.
        self._breaker_after = breaker_after
        self._rate_limit_streak = 0
        self.company_rss_enabled = True

    # ------------------------------------------------------------------
    # yfinance news (per-ticker)
    # ------------------------------------------------------------------

    def fetch_ticker_news(self, ticker, yf_ticker_obj=None):
        """Fetch news for a single ticker from yfinance.

        Args:
            ticker: Stock ticker symbol.
            yf_ticker_obj: Optional pre-existing yf.Ticker instance.

        Returns:
            list[dict] with keys: title, source, link, date, timestamp, origin.
        """
        if ticker in self._ticker_cache:
            return self._ticker_cache[ticker]

        try:
            if yf_ticker_obj is None:
                import yfinance as yf
                yf_ticker_obj = yf.Ticker(ticker, session=_yf_session())

            raw_news = yf_ticker_obj.news or []
            headlines = []
            cutoff = time.time() - (self._max_age_days * 86400)

            for item in raw_news:
                # yfinance may nest data under 'content' key
                content = item.get('content', item)
                # Handle both ISO string and unix timestamp for pubDate
                pub = content.get('pubDate') or item.get('providerPublishTime', '')
                ts = 0
                date_str = ''
                if isinstance(pub, str) and pub:
                    try:
                        dt = datetime.fromisoformat(pub.replace('Z', '+00:00'))
                        ts = dt.timestamp()
                        date_str = dt.strftime('%Y-%m-%d')
                    except Exception as e:
                        logger.debug(f'news: pubDate parse failed for {ticker}: {e}')
                elif isinstance(pub, (int, float)) and pub > 0:
                    ts = float(pub)
                    date_str = datetime.fromtimestamp(
                        ts, tz=timezone.utc
                    ).strftime('%Y-%m-%d')

                if ts < cutoff:
                    continue
                title = content.get('title') or item.get('title', '')
                source = (content.get('provider', {}).get('displayName', '')
                          if isinstance(content.get('provider'), dict)
                          else item.get('publisher', ''))
                link = content.get('canonicalUrl', {}).get('url', '') if isinstance(content.get('canonicalUrl'), dict) else item.get('link', '')
                headlines.append({
                    'title': title,
                    'source': source,
                    'link': link,
                    'date': date_str,
                    'timestamp': ts,
                    'origin': 'yfinance',
                })
        except Exception as e:
            # Fetch failed — return empty but DON'T cache, so a transient
            # failure doesn't read as "no news" for the rest of the run.
            logger.warning(f'news: yfinance news fetch failed for {ticker}: {e}')
            return []

        headlines.sort(key=lambda h: h.get('timestamp', 0), reverse=True)
        self._ticker_cache[ticker] = headlines[:10]
        return self._ticker_cache[ticker]

    # ------------------------------------------------------------------
    # Google News RSS
    # ------------------------------------------------------------------

    def fetch_company_news(self, ticker, company_name, max_items=8):
        """Fetch Google News RSS for one company.

        Cached on *ticker* alone, so the Phase-2 prefetch pool and the main
        loop cannot disagree about the company name and pay for the query
        twice — the second call must be a cache hit, on the critical path.

        Returns:
            list[dict] — may be empty; never raises.
        """
        if ticker in self._company_cache:
            return self._company_cache[ticker]
        if not self.company_rss_enabled:
            return []

        query = build_news_query(company_name, ticker)
        if not query:
            # Nothing identifies this company. Make no request rather than
            # fall back to a sector search — that fallback is the defect.
            self._company_cache[ticker] = []
            return []

        headlines = self._fetch_google_rss(query, max_items,
                                           origin='google_news_ticker')
        self._company_cache[ticker] = headlines
        return headlines

    def fetch_sector_news(self, sector, max_items=8):
        """Fetch sector-level news from Google News RSS.

        Args:
            sector: GICS sector name (e.g. 'Technology').
            max_items: Maximum headlines to return.

        Returns:
            list[dict] with keys: title, source, link, date, timestamp, origin.
        """
        if sector in self._sector_cache:
            return self._sector_cache[sector]

        query = f'{sector} stocks'
        headlines = self._fetch_google_rss(query, max_items)
        self._sector_cache[sector] = headlines
        return headlines

    def _fetch_google_rss(self, query, max_items=8, origin='google_news'):
        """Fetch and parse Google News RSS for a query string."""
        self._throttle()
        try:
            encoded_q = urllib.request.quote(query)
            url = (f'https://news.google.com/rss/search?q={encoded_q}'
                   '&hl=en-US&gl=US&ceid=US:en')
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0',
            })
            with urllib.request.urlopen(req, timeout=10) as resp:
                xml_data = resp.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):
                self._on_rate_limited(e.code)
            else:
                logger.warning(f'news: Google News RSS HTTP {e.code} for {query!r}')
            return []
        except Exception as e:
            logger.warning(f'news: Google News RSS fetch failed for {query!r}: {e}')
            return []

        # A healthy response walks a penalised interval back toward the base.
        self._rate_limit_streak = 0
        self._throttle.relax(self._relax_step)

        try:
            root = ET.fromstring(xml_data)
        except Exception as e:
            logger.warning(f'news: Google News RSS parse failed for {query!r}: {e}')
            return []

        items = root.findall('.//item')
        cutoff = time.time() - (self._max_age_days * 86400)
        headlines = []

        for item in items[:max_items * 2]:  # parse extra, filter later
            title = (item.findtext('title') or '').strip()
            link = (item.findtext('link') or '').strip()
            pub_date = item.findtext('pubDate') or ''
            source_el = item.find('source')
            source = source_el.text if source_el is not None else ''

            # Parse RFC 2822 date
            ts = 0
            date_str = ''
            if pub_date:
                try:
                    dt = parsedate_to_datetime(pub_date)
                    ts = dt.timestamp()
                    date_str = dt.strftime('%Y-%m-%d')
                except Exception as e:
                    logger.debug(f'news: RSS pubDate parse failed for {query!r}: {e}')

            if ts < cutoff or not title:
                continue

            headlines.append({
                'title': title,
                'source': source,
                'link': link,
                'date': date_str,
                'timestamp': ts,
                'origin': origin,
            })

            if len(headlines) >= max_items:
                break

        headlines.sort(key=lambda h: h.get('timestamp', 0), reverse=True)
        return headlines

    def _on_rate_limited(self, code):
        """Widen the interval, and stop per-company queries if it keeps up."""
        self._rate_limit_streak += 1
        delay = self._throttle.penalize(self._penalty, cap=self._delay_max)
        if self._rate_limit_streak == 1:
            logger.warning('news: Google News rate limited (%s) — interval now %.2fs',
                           code, delay)
        if (self._breaker_after and self.company_rss_enabled
                and self._rate_limit_streak >= self._breaker_after):
            self.company_rss_enabled = False
            logger.warning(
                'news: %d consecutive rate limits — disabling per-company Google '
                'News queries for this run; yfinance and sector news continue.',
                self._rate_limit_streak)

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def prefetch_all_sectors(self, sectors):
        """Prefetch news for all sectors (call once before the ticker loop).

        Args:
            sectors: iterable of sector names.
        """
        unique = set(s for s in sectors if s)
        logger.info(f'Fetching sector news for {len(unique)} sectors...')
        for sector in sorted(unique):
            self.fetch_sector_news(sector)

    def get_combined_news(self, ticker, sector, yf_ticker_obj=None,
                          company_name=None, max_total=12, max_sector=3):
        """Get the stock's news: its own headlines first, sector as context.

        Returns:
            list[dict] — company items ordered by relevance then recency,
            followed by at most *max_sector* sector items. Each dict gains
            'tier', 'scope' and 'tags'.
        """
        merged = self.fetch_ticker_news(ticker, yf_ticker_obj)
        merged = merged + self.fetch_company_news(ticker, company_name)
        if sector:
            merged = merged + self.fetch_sector_news(sector)
        return order_headlines(merged, ticker, company_name,
                               max_total=max_total, max_sector=max_sector)
