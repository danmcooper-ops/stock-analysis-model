"""SEC EDGAR Form 4 insider transaction parser.

Fetches and parses Form 4 filings to track insider buying/selling activity.
Provides transaction-level detail and aggregated metrics (buy ratio, net value).

Only counts open-market purchases (code 'P') and sales (code 'S') for
buy/sell metrics.  Excludes awards ('A') and option exercises ('M') which
are compensation-driven, not conviction signals.

Free API — no authentication required, just User-Agent with email.
Uses only stdlib (urllib + json + xml.etree).
"""

import json
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta


class SECInsiderClient:
    """Parse Form 4 insider transactions from SEC EDGAR."""

    _SUBMISSIONS_URL = 'https://data.sec.gov/submissions/CIK{cik}.json'

    # Transaction codes that represent genuine conviction signals
    _BUY_CODES = frozenset({'P'})     # Open-market purchase
    _SELL_CODES = frozenset({'S'})    # Open-market sale
    # Excluded: A (award/grant), M (exercise), G (gift), C (conversion),
    #           F (tax withholding), J (other), W (will/inheritance)

    # Human-readable transaction type mapping
    _CODE_LABELS = {
        'P': 'buy', 'S': 'sell', 'A': 'award', 'M': 'exercise',
        'G': 'gift', 'C': 'conversion', 'F': 'tax', 'J': 'other',
        'W': 'inheritance', 'D': 'disposition', 'I': 'discretionary',
    }

    def __init__(self, cik_map, name_map,
                 email='stockanalysis@example.com', request_delay=1.0,
                 max_form4_files=20):
        """
        Args:
            cik_map: dict {ticker: zero-padded CIK} from SECLegalClient.
            name_map: dict {ticker: company name} from SECLegalClient.
            email: Contact email for SEC User-Agent header.
            request_delay: Seconds between requests.
            max_form4_files: Max Form 4 XMLs to download per ticker.
        """
        self._ua = f'StockAnalyzer/1.0 ({email})'
        self._delay = request_delay
        self._last_req = 0
        self._cache = {}          # ticker -> result dict
        self._cik_map = cik_map
        self._name_map = name_map
        self._max_files = max_form4_files

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

    def _request_text(self, url, max_bytes=256_000, timeout=15):
        """GET request returning decoded text, or '' on failure."""
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
    # Filing discovery
    # ------------------------------------------------------------------

    def _find_form4_filings(self, cik, days_back=365):
        """Find recent Form 4 filings from SEC submissions API.

        Returns:
            list of (accession_number, primary_document, filing_date) tuples,
            newest first, capped at max_form4_files.
        """
        url = self._SUBMISSIONS_URL.format(cik=cik)
        data = self._request_json(url)
        if not data:
            return []

        recent = data.get('filings', {}).get('recent', {})
        forms = recent.get('form', [])
        accessions = recent.get('accessionNumber', [])
        primary_docs = recent.get('primaryDocument', [])
        filing_dates = recent.get('filingDate', [])

        cutoff = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        results = []

        for i, form in enumerate(forms):
            if form != '4':
                continue
            if i >= len(accessions) or i >= len(primary_docs) or i >= len(filing_dates):
                continue
            filing_date = filing_dates[i]
            if filing_date < cutoff:
                break  # Filings are sorted newest-first
            results.append((accessions[i], primary_docs[i], filing_date))
            if len(results) >= self._max_files:
                break

        return results

    # ------------------------------------------------------------------
    # Form 4 XML parsing
    # ------------------------------------------------------------------

    def _parse_form4_xml(self, cik, accession, filename):
        """Download and parse a Form 4 XML filing.

        Returns:
            list of transaction dicts, or empty list on failure.
        """
        cik_num = cik.lstrip('0') or '0'
        accession_clean = accession.replace('-', '')
        # The primaryDocument may point to an XSLT-rendered version
        # (e.g. xslF345X05/wk-form4_xxx.xml) which returns HTML.
        # Strip any XSLT prefix to get the raw XML filename.
        raw_filename = filename
        if '/' in filename:
            raw_filename = filename.split('/')[-1]
        url = (f'https://www.sec.gov/Archives/edgar/data/'
               f'{cik_num}/{accession_clean}/{raw_filename}')

        xml_text = self._request_text(url)
        if not xml_text:
            return []

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return []

        # Extract reporting owner info
        # Handle both namespaced and non-namespaced XML
        # Try without namespace first, then with common namespace
        owner_name = ''
        owner_title = ''
        is_officer = False
        is_director = False

        for owner_el in self._find_elements(root, 'reportingOwner'):
            # Owner ID
            for oid in self._find_elements(owner_el, 'reportingOwnerId'):
                name_el = self._find_element(oid, 'rptOwnerName')
                if name_el is not None and name_el.text:
                    owner_name = name_el.text.strip()

            # Owner relationship
            for rel in self._find_elements(owner_el, 'reportingOwnerRelationship'):
                title_el = self._find_element(rel, 'officerTitle')
                if title_el is not None and title_el.text:
                    owner_title = title_el.text.strip()
                officer_el = self._find_element(rel, 'isOfficer')
                if officer_el is not None and officer_el.text:
                    is_officer = officer_el.text.strip() in ('1', 'true', 'True')
                director_el = self._find_element(rel, 'isDirector')
                if director_el is not None and director_el.text:
                    is_director = director_el.text.strip() in ('1', 'true', 'True')
            break  # Use first reporting owner

        # Extract non-derivative transactions
        transactions = []
        for table in self._find_elements(root, 'nonDerivativeTable'):
            for txn_el in self._find_elements(table, 'nonDerivativeTransaction'):
                txn = self._parse_single_transaction(
                    txn_el, owner_name, owner_title, is_officer, is_director)
                if txn:
                    transactions.append(txn)

        return transactions

    def _parse_single_transaction(self, txn_el, owner_name, owner_title,
                                   is_officer, is_director):
        """Parse a single <nonDerivativeTransaction> element.

        Returns:
            dict with transaction details, or None if unparseable.
        """
        try:
            # Transaction date
            date_str = None
            for d in self._find_elements(txn_el, 'transactionDate'):
                val = self._find_element(d, 'value')
                if val is not None and val.text:
                    date_str = val.text.strip()
            if not date_str:
                return None

            # Transaction code
            code = None
            for tc in self._find_elements(txn_el, 'transactionCoding'):
                code_el = self._find_element(tc, 'transactionCode')
                if code_el is not None and code_el.text:
                    code = code_el.text.strip()
            if not code:
                return None

            # Transaction amounts
            shares = 0.0
            price = 0.0
            for ta in self._find_elements(txn_el, 'transactionAmounts'):
                shares_el = self._find_element(ta, 'transactionShares')
                if shares_el is not None:
                    v = self._find_element(shares_el, 'value')
                    if v is not None and v.text:
                        try:
                            shares = float(v.text.strip())
                        except ValueError:
                            pass

                price_el = self._find_element(ta, 'transactionPricePerShare')
                if price_el is not None:
                    v = self._find_element(price_el, 'value')
                    if v is not None and v.text:
                        try:
                            price = float(v.text.strip())
                        except ValueError:
                            pass

            # Shares owned after transaction
            shares_after = None
            for po in self._find_elements(txn_el, 'postTransactionAmounts'):
                soa = self._find_element(po, 'sharesOwnedFollowingTransaction')
                if soa is not None:
                    v = self._find_element(soa, 'value')
                    if v is not None and v.text:
                        try:
                            shares_after = float(v.text.strip())
                        except ValueError:
                            pass

            return {
                'date': date_str,
                'insider_name': owner_name,
                'title': owner_title,
                'is_officer': is_officer,
                'is_director': is_director,
                'transaction_code': code,
                'transaction_type': self._CODE_LABELS.get(code, 'other'),
                'shares': shares,
                'price_per_share': price,
                'dollar_value': round(shares * price, 2) if price else 0.0,
                'shares_after': shares_after,
            }

        except Exception:
            return None

    # ------------------------------------------------------------------
    # XML element helpers (handle optional namespaces)
    # ------------------------------------------------------------------

    @staticmethod
    def _find_elements(parent, tag):
        """Find child elements by local name, ignoring namespace."""
        results = parent.findall(tag)
        if results:
            return results
        # Fallback: iterate children and match local name
        return [c for c in parent if c.tag.split('}')[-1] == tag]

    @staticmethod
    def _find_element(parent, tag):
        """Find first child element by local name, ignoring namespace."""
        el = parent.find(tag)
        if el is not None:
            return el
        # Fallback: iterate children and match local name
        for c in parent:
            if c.tag.split('}')[-1] == tag:
                return c
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_insider_activity(self, ticker, days_back=365):
        """Fetch insider transaction activity from SEC Form 4 filings.

        Returns:
            dict with transaction details and aggregated metrics:
                available (bool): Whether any Form 4 data was found.
                transactions (list): Recent transactions (newest first).
                buy_count_90d, sell_count_90d, buy_ratio_90d: 90-day metrics.
                buy_count_365d, sell_count_365d, buy_ratio_365d: Full-year.
                net_shares_365d (float): Net shares bought - sold.
                net_value_365d (float): Net dollar value of buys - sells.
                insider_buy_ratio (float): Primary signal (= buy_ratio_365d).
        """
        if ticker in self._cache:
            return self._cache[ticker]

        empty = {
            'available': False,
            'transactions': [],
            'buy_count_90d': 0, 'sell_count_90d': 0, 'buy_ratio_90d': None,
            'buy_count_365d': 0, 'sell_count_365d': 0, 'buy_ratio_365d': None,
            'net_shares_365d': 0, 'net_value_365d': 0.0,
            'insider_buy_ratio': None,
        }

        cik = self._cik_map.get(ticker)
        if not cik:
            self._cache[ticker] = empty
            return empty

        # Step 1: Find recent Form 4 filings
        filings = self._find_form4_filings(cik, days_back=days_back)
        if not filings:
            self._cache[ticker] = empty
            return empty

        # Step 2: Parse each Form 4 XML
        all_transactions = []
        for accession, primary_doc, filing_date in filings:
            txns = self._parse_form4_xml(cik, accession, primary_doc)
            all_transactions.extend(txns)

        if not all_transactions:
            self._cache[ticker] = empty
            return empty

        # Sort by date descending
        all_transactions.sort(key=lambda t: t['date'], reverse=True)

        # Step 3: Aggregate metrics
        cutoff_90d = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        cutoff_365d = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

        buy_90 = sell_90 = 0
        buy_365 = sell_365 = 0
        net_shares = 0.0
        net_value = 0.0

        for t in all_transactions:
            code = t.get('transaction_code', '')
            txn_date = t.get('date', '')

            if txn_date < cutoff_365d:
                continue

            is_buy = code in self._BUY_CODES
            is_sell = code in self._SELL_CODES

            if is_buy:
                buy_365 += 1
                net_shares += t.get('shares', 0)
                net_value += t.get('dollar_value', 0)
                if txn_date >= cutoff_90d:
                    buy_90 += 1
            elif is_sell:
                sell_365 += 1
                net_shares -= t.get('shares', 0)
                net_value -= t.get('dollar_value', 0)
                if txn_date >= cutoff_90d:
                    sell_90 += 1

        # Buy ratios (None if no open-market transactions)
        total_90 = buy_90 + sell_90
        total_365 = buy_365 + sell_365
        buy_ratio_90d = buy_90 / total_90 if total_90 > 0 else None
        buy_ratio_365d = buy_365 / total_365 if total_365 > 0 else None

        # Cap transaction list for display
        display_txns = [t for t in all_transactions if t['date'] >= cutoff_365d]
        display_txns = display_txns[:20]

        result = {
            'available': True,
            'transactions': display_txns,
            'buy_count_90d': buy_90,
            'sell_count_90d': sell_90,
            'buy_ratio_90d': round(buy_ratio_90d, 3) if buy_ratio_90d is not None else None,
            'buy_count_365d': buy_365,
            'sell_count_365d': sell_365,
            'buy_ratio_365d': round(buy_ratio_365d, 3) if buy_ratio_365d is not None else None,
            'net_shares_365d': round(net_shares, 0),
            'net_value_365d': round(net_value, 2),
            'insider_buy_ratio': round(buy_ratio_365d, 3) if buy_ratio_365d is not None else None,
        }

        self._cache[ticker] = result
        return result
