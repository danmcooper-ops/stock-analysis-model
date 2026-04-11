"""SEC EDGAR 10-K supply chain extraction.

Downloads each company's latest 10-K filing from SEC EDGAR, extracts
text sections mentioning suppliers/customers, and matches company names
against the CIK company name registry to get ticker symbols.

Free API — no authentication required, just User-Agent with email.
Uses only stdlib (urllib + json + re + html.parser).
"""

import json
import os
import re
import time
import urllib.error
import urllib.request
from html.parser import HTMLParser


class _HTMLStripper(HTMLParser):
    """Strip HTML tags, return plain text."""

    def __init__(self):
        super().__init__()
        self._parts = []

    def handle_data(self, data):
        self._parts.append(data)

    def get_text(self):
        return ' '.join(self._parts)


class SECSupplyClient:
    """Extract supplier/customer relationships from SEC 10-K filings."""

    _SUBMISSIONS_URL = 'https://data.sec.gov/submissions/CIK{cik}.json'

    # Corporate suffixes to strip during name normalization
    _SUFFIXES = frozenset({
        'inc', 'corp', 'corporation', 'co', 'company', 'ltd', 'limited',
        'llc', 'plc', 'sa', 'nv', 'se', 'ag', 'group', 'holdings',
        'international', 'intl', 'enterprises', 'technologies', 'technology',
    })

    # Names too short/ambiguous to match reliably
    _AMBIGUOUS = frozenset({
        'general', 'national', 'american', 'united', 'first', 'new',
        'global', 'western', 'eastern', 'southern', 'northern',
        'pacific', 'atlantic', 'central', 'standard', 'premier',
        'advanced', 'applied', 'digital', 'electronic', 'industrial',
        'financial', 'capital', 'energy', 'power', 'health', 'medical',
        'select', 'core', 'pure', 'open', 'net', 'one', 'two',
        'alpha', 'delta', 'omega', 'flex', 'bio', 'air', 'us',
    })

    # Common words in 10-K filings that cause false matches when part of
    # a company name.  If EVERY word in a normalized name is in this set,
    # the name is skipped.
    _COMMON_WORDS = frozenset({
        'a', 'an', 'the', 'and', 'or', 'of', 'in', 'on', 'at', 'to',
        'for', 'by', 'with', 'from', 'as', 'is', 'it', 'its', 'are',
        'be', 'we', 'our', 'all', 'any', 'no', 'not', 'may', 'can',
        'will', 'has', 'had', 'have', 'been', 'was', 'were', 'this',
        'that', 'these', 'those', 'such', 'each', 'other', 'more',
        # Business / financial terms common in filings
        'financial', 'institutions', 'services', 'service', 'solutions',
        'systems', 'management', 'resources', 'partners', 'partners',
        'consulting', 'advisors', 'investments', 'properties',
        'target', 'group', 'senior', 'billion', 'million', 'annual',
        'quality', 'industrial', 'strategic', 'strategy', 'emerging',
        'intelligent', 'magnitude', 'frequency', 'reliability',
        'interface', 'insight', 'bandwidth', 'array', 'strive',
        'freight', 'premier', 'summit', 'pinnacle', 'progress',
        'performance', 'preferred', 'principal', 'prime', 'precision',
        'creative', 'innovative', 'dynamic', 'essential', 'efficient',
        'independent', 'integrated', 'interactive', 'universal',
        'regional', 'local', 'direct', 'express', 'rapid', 'swift',
        'clear', 'bright', 'smart', 'wise', 'true', 'real', 'ideal',
        'classic', 'modern', 'new', 'next', 'first', 'primary',
    })

    # Keywords signalling supplier mentions
    _SUPPLIER_KW = [
        'supplier', 'suppliers', 'vendor', 'vendors',
        'sourced from', 'supply agreement', 'supply chain',
        'sole source', 'single source', 'raw material',
        'contract manufacturer', 'foundry', 'fabrication',
        'procurement', 'procure from',
    ]

    # Keywords signalling customer mentions
    _CUSTOMER_KW = [
        'customer', 'customers', 'client', 'clients',
        'major customer', 'significant customer', 'largest customer',
        'accounted for', 'revenue from', 'sales to',
        'end user', 'distribution partner',
    ]

    def __init__(self, cik_map, name_map,
                 email='stockanalysis@example.com', request_delay=1.0):
        """
        Args:
            cik_map: dict {ticker: zero-padded CIK} from SECLegalClient.
            name_map: dict {ticker: company name} from SECLegalClient.
            email: Contact email for SEC User-Agent header.
            request_delay: Seconds between requests.
        """
        self._ua = f'StockAnalyzer/1.0 ({email})'
        self._delay = request_delay
        self._last_req = 0
        self._cache = {}              # ticker -> result dict
        self._cik_map = cik_map
        self._name_map = name_map
        self._reverse_map = None      # normalized_name -> ticker (built lazily)

    # ------------------------------------------------------------------
    # Throttling & requests
    # ------------------------------------------------------------------

    def _throttle(self):
        elapsed = time.time() - self._last_req
        if elapsed < self._delay:
            time.sleep(self._delay - elapsed)
        self._last_req = time.time()

    def _request_json(self, url, timeout=15):
        """GET request returning parsed JSON, or None on failure."""
        self._throttle()
        try:
            req = urllib.request.Request(url, headers={'User-Agent': self._ua})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(5)
            return None
        except Exception:
            return None

    def _request_text(self, url, max_bytes=512_000, timeout=30):
        """GET request returning decoded text (capped), or '' on failure."""
        self._throttle()
        try:
            req = urllib.request.Request(url, headers={'User-Agent': self._ua})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read(max_bytes)
            return raw.decode('utf-8', errors='ignore')
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(5)
            return ''
        except Exception:
            return ''

    # ------------------------------------------------------------------
    # Name normalization & reverse map
    # ------------------------------------------------------------------

    def _normalize(self, name):
        """Lowercase, strip punctuation and corporate suffixes."""
        name = name.lower()
        name = re.sub(r'[.,/\\\-\'"()&!@#$%^*+=:;]', ' ', name)
        words = name.split()
        words = [w for w in words if w not in self._SUFFIXES]
        return ' '.join(words).strip()

    def _build_reverse_map(self):
        """Build {normalized_name -> compiled_regex} lookup from the CIK name map.

        Only stores full normalized names (no short-name fragments) to avoid
        false positives from common English words.  Uses word-boundary regex
        for matching.
        """
        if self._reverse_map is not None:
            return
        rmap = {}   # normalized_name -> (ticker, compiled_regex)
        for ticker, full_name in self._name_map.items():
            norm = self._normalize(full_name)
            # Require meaningful length
            if not norm or len(norm) < 6:
                continue
            words = norm.split()
            # Skip single common/ambiguous words
            if len(words) == 1 and words[0] in self._AMBIGUOUS:
                continue
            # Skip names where ALL words are common English — too many
            # false positives (e.g. "financial institutions", "target group")
            if all(w in self._COMMON_WORDS for w in words):
                continue
            try:
                pat = re.compile(r'\b' + re.escape(norm) + r'\b')
                rmap[norm] = (ticker, pat)
            except re.error:
                continue
        self._reverse_map = rmap
        print(f'SEC supply chain: built reverse name map ({len(rmap)} entries)')

    # ------------------------------------------------------------------
    # Filing discovery & download
    # ------------------------------------------------------------------

    def _find_latest_10k(self, cik):
        """Find latest 10-K (or 20-F) filing for a CIK.

        Returns:
            (accession_number, primary_document_filename) or (None, None).
        """
        url = self._SUBMISSIONS_URL.format(cik=cik)
        data = self._request_json(url)
        if not data:
            return None, None

        recent = data.get('filings', {}).get('recent', {})
        forms = recent.get('form', [])
        accessions = recent.get('accessionNumber', [])
        primary_docs = recent.get('primaryDocument', [])

        for i, form in enumerate(forms):
            if form in ('10-K', '20-F'):
                if i < len(accessions) and i < len(primary_docs):
                    return accessions[i], primary_docs[i]
        return None, None

    def _download_filing_text(self, cik, accession, filename):
        """Download 10-K HTML and return plain text (capped at 500KB)."""
        cik_num = cik.lstrip('0') or '0'
        accession_clean = accession.replace('-', '')
        url = (f'https://www.sec.gov/Archives/edgar/data/'
               f'{cik_num}/{accession_clean}/{filename}')
        html = self._request_text(url)
        if not html:
            return ''
        stripper = _HTMLStripper()
        try:
            stripper.feed(html)
        except Exception:
            pass
        return stripper.get_text()

    # ------------------------------------------------------------------
    # Relationship extraction
    # ------------------------------------------------------------------

    def _find_names_in_window(self, window):
        """Scan a text window for known company names using word-boundary regex.

        Uses a two-stage approach for performance:
          1. Fast plain-string pre-filter — skips ~99% of candidates instantly.
          2. Word-boundary regex — only runs on names that passed stage 1,
             ensuring precision (no false positives from partial matches).

        Returns:
            set of ticker symbols found.
        """
        self._build_reverse_map()
        window_norm = self._normalize(window)
        found = set()
        for name_key, (ticker, pat) in self._reverse_map.items():
            # Stage 1: cheap substring check before running the regex
            if name_key not in window_norm:
                continue
            # Stage 2: word-boundary regex for precision
            if pat.search(window_norm):
                found.add(ticker)
        return found

    def _extract_relationships(self, text, self_ticker=None):
        """Extract supplier and customer tickers from 10-K text.

        Args:
            text: Plain text of the 10-K filing.
            self_ticker: Ticker of the filing company (to exclude).

        Returns:
            dict with 'suppliers' and 'customers' (sets of tickers).
        """
        text_lower = text.lower()
        supplier_tickers = set()
        customer_tickers = set()

        for kw in self._SUPPLIER_KW:
            for m in re.finditer(re.escape(kw), text_lower):
                start = max(0, m.start() - 300)
                end = min(len(text), m.end() + 500)
                window = text[start:end]
                supplier_tickers |= self._find_names_in_window(window)

        for kw in self._CUSTOMER_KW:
            for m in re.finditer(re.escape(kw), text_lower):
                start = max(0, m.start() - 300)
                end = min(len(text), m.end() + 500)
                window = text[start:end]
                customer_tickers |= self._find_names_in_window(window)

        # Remove self-references
        if self_ticker:
            supplier_tickers.discard(self_ticker)
            customer_tickers.discard(self_ticker)

        return {
            'suppliers': supplier_tickers,
            'customers': customer_tickers,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_supply_chain(self, ticker, max_suppliers=10, max_customers=10):
        """Fetch supplier/customer data from SEC 10-K filing.

        Interface matches FinnhubSupplyClient.fetch_supply_chain().

        Returns:
            dict with keys:
                suppliers (list[dict]): Each has 'symbol' and 'name'.
                customers (list[dict]): Same structure.
                available (bool): Whether extraction succeeded.
        """
        if ticker in self._cache:
            return self._cache[ticker]

        empty = {'suppliers': [], 'customers': [], 'available': False}

        cik = self._cik_map.get(ticker)
        if not cik:
            self._cache[ticker] = empty
            return empty

        # Step 1: Find latest 10-K
        accession, primary_doc = self._find_latest_10k(cik)
        if not accession or not primary_doc:
            self._cache[ticker] = empty
            return empty

        # Step 2: Download and extract text
        text = self._download_filing_text(cik, accession, primary_doc)
        if not text or len(text) < 1000:
            self._cache[ticker] = empty
            return empty

        # Step 3: Extract relationships
        raw = self._extract_relationships(text, self_ticker=ticker)

        # Step 4: Format as [{symbol, name}]
        suppliers = [
            {'symbol': t, 'name': self._name_map.get(t, t)}
            for t in sorted(raw['suppliers'])
        ][:max_suppliers]

        customers = [
            {'symbol': t, 'name': self._name_map.get(t, t)}
            for t in sorted(raw['customers'])
        ][:max_customers]

        result = {
            'suppliers': suppliers,
            'customers': customers,
            'available': True,
        }
        self._cache[ticker] = result
        return result
