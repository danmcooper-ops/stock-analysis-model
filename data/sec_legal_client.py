"""SEC EDGAR legal proceedings fetcher.

Searches the EDGAR EFTS (full-text search) API for mentions of
"legal proceedings" in 10-K, 10-Q, and 8-K filings for each company.

Free API — no authentication required, just a User-Agent header with
contact email (SEC requirement). Uses only stdlib (urllib + json).
"""

import json
import time
import urllib.request
from datetime import datetime, timedelta


class SECLegalClient:
    """Fetch legal proceedings filings from SEC EDGAR EFTS."""

    _EFTS_URL = 'https://efts.sec.gov/LATEST/search-index'
    _CIK_URL = 'https://www.sec.gov/files/company_tickers.json'

    def __init__(self, email='stockanalysis@example.com', request_delay=1.0):
        self._ua = f'StockAnalyzer/1.0 ({email})'
        self._delay = request_delay
        self._last_req = 0
        self._cache = {}           # ticker -> result dict
        self._cik_map = None       # ticker -> zero-padded CIK string
        self._name_map = None      # ticker -> company name (from SEC)

    def _throttle(self):
        elapsed = time.time() - self._last_req
        if elapsed < self._delay:
            time.sleep(self._delay - elapsed)
        self._last_req = time.time()

    def _request(self, url, timeout=15):
        """Make a GET request with User-Agent header. Returns parsed JSON or None."""
        self._throttle()
        try:
            req = urllib.request.Request(url, headers={'User-Agent': self._ua})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Ticker → CIK + company name mapping
    # ------------------------------------------------------------------

    def _load_cik_map(self):
        """Fetch the SEC company_tickers.json and build lookup dicts."""
        if self._cik_map is not None:
            return
        data = self._request(self._CIK_URL)
        if not data:
            self._cik_map = {}
            self._name_map = {}
            return
        cik_map = {}
        name_map = {}
        for entry in data.values():
            ticker = entry.get('ticker', '')
            cik_map[ticker] = str(entry.get('cik_str', '')).zfill(10)
            name_map[ticker] = entry.get('title', '')
        self._cik_map = cik_map
        self._name_map = name_map
        print(f'SEC EDGAR: loaded CIK map ({len(cik_map)} tickers)')

    def get_cik(self, ticker):
        """Return zero-padded CIK for a ticker, or None."""
        self._load_cik_map()
        return self._cik_map.get(ticker)

    def get_company_name(self, ticker):
        """Return SEC-registered company name for a ticker, or None."""
        self._load_cik_map()
        return self._name_map.get(ticker)

    # ------------------------------------------------------------------
    # Human-readable filing summary builder
    # ------------------------------------------------------------------

    _FORM_LABELS = {
        '10-K': 'Annual Report',
        '10-K/A': 'Annual Report (Amended)',
        '10-Q': 'Quarterly Report',
        '10-Q/A': 'Quarterly Report (Amended)',
        '8-K': 'Current Report',
        '8-K/A': 'Current Report (Amended)',
        '20-F': 'Annual Report (Foreign)',
        '6-K': 'Foreign Issuer Report',
    }

    @staticmethod
    def _format_period(period_ending):
        """Convert '2024-09-28' to 'Sep 2024'."""
        if not period_ending or len(period_ending) < 7:
            return ''
        try:
            from datetime import datetime as _dt
            dt = _dt.strptime(period_ending[:10], '%Y-%m-%d')
            return dt.strftime('%b %Y')
        except (ValueError, TypeError):
            return ''

    def _build_summary(self, form_type, file_type, description, period_ending):
        """Build a concise summary line for a legal filing.

        Combines form type label, exhibit info, and fiscal period into a
        short description like:
          'Annual Report — FY ending Sep 2024'
          'Exhibit 99.1: Certain Litigation Matters (Q ending Mar 2024)'
          'Current Report — material event disclosure'
        """
        desc_upper = (description or '').upper().strip()

        # Check if this is an exhibit rather than the main filing doc
        is_exhibit = (file_type or '').upper().startswith('EX-')

        # Formatted fiscal period
        period_str = self._format_period(period_ending)

        if is_exhibit:
            # Use the file_description if it's meaningful (not just the form)
            exhibit_label = file_type.upper()
            if desc_upper and desc_upper not in (form_type.upper(),
                                                  f'FORM {form_type}'.upper(),
                                                  '', 'NONE'):
                # Title-case the description
                desc_nice = description.strip().title()
                parts = [f'{exhibit_label}: {desc_nice}']
            else:
                parts = [f'{exhibit_label} (Legal Proceedings)']
            if period_str:
                parts.append(f'Period ending {period_str}')
            return ' — '.join(parts)

        # Main document — use the form label
        label = self._FORM_LABELS.get(form_type, form_type)

        if form_type in ('10-K', '10-K/A', '20-F'):
            if period_str:
                return f'{label} — FY ending {period_str}'
            return f'{label} — legal proceedings disclosure'
        elif form_type in ('10-Q', '10-Q/A'):
            if period_str:
                return f'{label} — quarter ending {period_str}'
            return f'{label} — legal proceedings disclosure'
        elif form_type in ('8-K', '8-K/A'):
            # 8-K descriptions sometimes have useful info
            if (desc_upper and desc_upper not in (form_type.upper(),
                                                   f'FORM {form_type}'.upper(),
                                                   '', 'NONE')):
                return f'{label}: {description.strip().title()}'
            return f'{label} — material event disclosure'
        else:
            if period_str:
                return f'{label} — period ending {period_str}'
            return label

    # ------------------------------------------------------------------
    # EFTS full-text search for legal proceedings
    # ------------------------------------------------------------------

    def fetch_legal_filings(self, ticker, days_back=730):
        """Search EDGAR EFTS for legal proceedings mentions.

        Args:
            ticker: Stock ticker symbol.
            days_back: How many days back to search (default 2 years).

        Returns:
            dict with keys:
                count (int): Number of unique filings found.
                filings (list[dict]): Filing metadata sorted newest-first.
                    Each dict has: form_type, filing_date, description, url
                latest_date (str|None): Most recent filing date, or None.
        """
        if ticker in self._cache:
            return self._cache[ticker]

        self._load_cik_map()
        company_name = self._name_map.get(ticker)
        cik = self._cik_map.get(ticker)
        if not company_name:
            result = {'count': 0, 'filings': [], 'latest_date': None}
            self._cache[ticker] = result
            return result

        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=days_back)

        # Build EFTS query: "legal proceedings" + "COMPANY NAME"
        q_parts = [
            '"legal proceedings"',
            f'"{company_name}"',
        ]
        q = ' '.join(q_parts)
        encoded_q = urllib.request.quote(q)
        url = (
            f'{self._EFTS_URL}'
            f'?q={encoded_q}'
            f'&dateRange=custom'
            f'&startdt={start_dt.strftime("%Y-%m-%d")}'
            f'&enddt={end_dt.strftime("%Y-%m-%d")}'
            f'&forms=10-K,10-Q,8-K'
        )

        data = self._request(url)
        if not data:
            result = {'count': 0, 'filings': [], 'latest_date': None}
            self._cache[ticker] = result
            return result

        raw_hits = data.get('hits', {}).get('hits', [])

        # Deduplicate by accession number (adsh), keep unique filings
        seen_adsh = set()
        filings = []
        for h in raw_hits:
            src = h.get('_source', {})
            adsh = src.get('adsh', '')
            if adsh in seen_adsh:
                continue
            seen_adsh.add(adsh)

            form_type = src.get('form', src.get('file_type', ''))
            filing_date = src.get('file_date', '')
            description = src.get('file_description', form_type)
            period_ending = src.get('period_ending', '')
            file_type = src.get('file_type', '')

            # Build human-readable summary from metadata
            summary = self._build_summary(form_type, file_type,
                                          description, period_ending)

            # Build SEC filing URL from CIK + accession number
            filing_url = ''
            if cik and adsh:
                cik_num = cik.lstrip('0') or '0'
                adsh_clean = adsh.replace('-', '')
                filing_url = (
                    f'https://www.sec.gov/Archives/edgar/data/'
                    f'{cik_num}/{adsh_clean}/{adsh}-index.htm'
                )

            filings.append({
                'form_type': form_type,
                'filing_date': filing_date,
                'description': description,
                'summary': summary,
                'url': filing_url,
            })

        # Sort newest-first
        filings.sort(key=lambda f: f['filing_date'], reverse=True)

        # Cap at 20 to keep payload reasonable
        filings = filings[:20]

        latest = filings[0]['filing_date'] if filings else None
        result = {
            'count': len(filings),
            'filings': filings,
            'latest_date': latest,
        }
        self._cache[ticker] = result
        return result
