"""News headline fetcher combining yfinance and Google News RSS.

Provides two headline sources:
  1. yfinance Ticker.news — per-company headlines (8-10 items)
  2. Google News RSS — sector-level headlines via stdlib XML parsing

All functions are resilient: failures return empty lists, never raise.
No new pip dependencies — uses only stdlib + yfinance (already in pipeline).
"""

import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime


class NewsClient:
    """Fetch news headlines from yfinance and Google News RSS."""

    def __init__(self, request_delay=1.0, max_age_days=30):
        self._delay = request_delay
        self._last_req = 0
        self._ticker_cache = {}   # ticker -> list[dict]
        self._sector_cache = {}   # sector -> list[dict]
        self._max_age_days = max_age_days

    def _throttle(self):
        elapsed = time.time() - self._last_req
        if elapsed < self._delay:
            time.sleep(self._delay - elapsed)
        self._last_req = time.time()

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
                yf_ticker_obj = yf.Ticker(ticker)

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
                    except Exception:
                        pass
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
        except Exception:
            headlines = []

        headlines.sort(key=lambda h: h.get('timestamp', 0), reverse=True)
        self._ticker_cache[ticker] = headlines[:10]
        return self._ticker_cache[ticker]

    # ------------------------------------------------------------------
    # Google News RSS (per-sector)
    # ------------------------------------------------------------------

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

    def _fetch_google_rss(self, query, max_items=8):
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

            root = ET.fromstring(xml_data)
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
                    except Exception:
                        pass

                if ts < cutoff or not title:
                    continue

                headlines.append({
                    'title': title,
                    'source': source,
                    'link': link,
                    'date': date_str,
                    'timestamp': ts,
                    'origin': 'google_news',
                })

                if len(headlines) >= max_items:
                    break

            headlines.sort(key=lambda h: h.get('timestamp', 0), reverse=True)
            return headlines

        except Exception:
            return []

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def prefetch_all_sectors(self, sectors):
        """Prefetch news for all sectors (call once before the ticker loop).

        Args:
            sectors: iterable of sector names.
        """
        unique = set(s for s in sectors if s)
        print(f'Fetching sector news for {len(unique)} sectors...')
        for sector in sorted(unique):
            self.fetch_sector_news(sector)

    def get_combined_news(self, ticker, sector, yf_ticker_obj=None,
                          max_total=12):
        """Get merged ticker + sector news, deduplicated and sorted.

        Returns:
            list[dict] — newest first, capped at max_total.
        """
        ticker_news = self.fetch_ticker_news(ticker, yf_ticker_obj)
        sector_news = self.fetch_sector_news(sector) if sector else []

        # Deduplicate by exact title match
        seen_titles = set()
        combined = []
        for h in ticker_news + sector_news:
            t_lower = h['title'].lower().strip()
            if t_lower not in seen_titles:
                seen_titles.add(t_lower)
                combined.append(h)

        combined.sort(key=lambda h: h.get('timestamp', 0), reverse=True)
        return combined[:max_total]
