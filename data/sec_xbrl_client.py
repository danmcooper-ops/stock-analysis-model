"""SEC EDGAR XBRL Company Facts client.

Fetches structured financial data from the SEC XBRL API to:
1. Cross-validate yfinance financial statements (flag discrepancies > 5%)
2. Provide 10+ years of revenue/earnings history for extended CAGR analysis

Free API — no authentication required, just User-Agent with email.
Uses only stdlib (urllib + json + datetime).
"""

import json
import time
import urllib.error
import urllib.request
from datetime import datetime as _dt

from models.field_keys import (
    _get, REVENUE_KEYS, NET_INCOME_KEYS, TOTAL_ASSETS_KEYS,
    EQUITY_KEYS, DEBT_KEYS, OPERATING_INCOME_KEYS, CASH_KEYS,
)


class SECXBRLClient:
    """Fetch and interpret XBRL Company Facts from SEC EDGAR."""

    _COMPANY_FACTS_URL = 'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json'

    # Multiple possible US-GAAP tags per financial concept.
    # Ordered by prevalence — first match wins.
    _XBRL_TAG_MAP = {
        'revenue': [
            'Revenues',
            'RevenueFromContractWithCustomerExcludingAssessedTax',
            'RevenueFromContractWithCustomerIncludingAssessedTax',
            'SalesRevenueNet',
            'SalesRevenueGoodsNet',
            'RevenueFromContractWithCustomerExcludingAssessedTaxTransferredAtAPointInTime',
        ],
        'net_income': [
            'NetIncomeLoss',
            'NetIncomeLossAvailableToCommonStockholdersBasic',
            'ProfitLoss',
        ],
        'total_assets': [
            'Assets',
        ],
        'total_equity': [
            'StockholdersEquity',
            'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
        ],
        'total_debt': [
            'LongTermDebt',
            'LongTermDebtAndCapitalLeaseObligations',
            'LongTermDebtNoncurrent',
        ],
        'operating_income': [
            'OperatingIncomeLoss',
        ],
        'cash': [
            'CashAndCashEquivalentsAtCarryingValue',
            'CashCashEquivalentsAndShortTermInvestments',
        ],
    }

    # Map XBRL concept names to yfinance field_keys lists for cross-validation
    _YF_KEY_MAP = {
        'revenue': REVENUE_KEYS,
        'net_income': NET_INCOME_KEYS,
        'total_assets': TOTAL_ASSETS_KEYS,
        'total_equity': EQUITY_KEYS,
        'total_debt': DEBT_KEYS,
    }

    # Which yfinance statement contains each concept
    _YF_STATEMENT_MAP = {
        'revenue': 'income_statement',
        'net_income': 'income_statement',
        'total_assets': 'balance_sheet',
        'total_equity': 'balance_sheet',
        'total_debt': 'balance_sheet',
    }

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
        self._cache = {}          # ticker -> raw company facts JSON
        self._cik_map = cik_map
        self._name_map = name_map

    # ------------------------------------------------------------------
    # Throttling & requests
    # ------------------------------------------------------------------

    def _throttle(self):
        elapsed = time.time() - self._last_req
        if elapsed < self._delay:
            time.sleep(self._delay - elapsed)
        self._last_req = time.time()

    def _request_json(self, url, timeout=20):
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

    # ------------------------------------------------------------------
    # Core data extraction
    # ------------------------------------------------------------------

    def fetch_company_facts(self, ticker):
        """Fetch full XBRL company facts JSON from SEC EDGAR.

        Returns:
            dict: The raw companyfacts JSON, or None on failure.
        """
        if ticker in self._cache:
            return self._cache[ticker]

        cik = self._cik_map.get(ticker)
        if not cik:
            self._cache[ticker] = None
            return None

        url = self._COMPANY_FACTS_URL.format(cik=cik)
        data = self._request_json(url)
        self._cache[ticker] = data
        return data

    def _extract_annual_values(self, facts_json, tag_list, form_filter='10-K'):
        """Extract annual values for a concept from XBRL facts.

        Tries each tag in tag_list until one has data.  Filters for the
        specified form type.  Handles the complexity of XBRL entries which
        include segment breakdowns and prior-year restatements alongside
        total annual figures.

        Selection logic per fiscal year:
        1. Only 10-K / 20-F filings with fp='FY'
        2. Period must span ~1 full year (340-400 days)
        3. Period end year must match the fiscal year
        4. If multiple match, keep the latest filed date

        Args:
            facts_json: Raw companyfacts JSON.
            tag_list: List of US-GAAP tag names to try.
            form_filter: Filing form type ('10-K' or '20-F').

        Returns:
            dict {fiscal_year (int): value (float)} sorted by year,
            or empty dict if no data found.
        """
        if not facts_json:
            return {}

        us_gaap = facts_json.get('facts', {}).get('us-gaap', {})

        # Merge values across all matching tags. Companies often switch
        # XBRL tags over time (e.g. SalesRevenueNet → RevenueFromContract...)
        merged = {}  # {fy: (filed_date, value)}

        for tag in tag_list:
            concept = us_gaap.get(tag)
            if not concept:
                continue

            # Try USD units first
            units = concept.get('units', {})
            entries = units.get('USD', [])
            if not entries:
                continue

            # Filter for annual filings with full-year periods
            candidates = []
            for e in entries:
                form = e.get('form', '')
                if form not in (form_filter, '20-F'):
                    continue
                fy = e.get('fy')
                fp = e.get('fp', '')
                val = e.get('val')
                filed = e.get('filed', '')
                start = e.get('start', '')
                end = e.get('end', '')
                if fy is None or val is None:
                    continue
                # Only full-year figures
                if fp not in ('FY', 'CY'):
                    continue

                # Two types of XBRL entries:
                # 1. Duration (income/cash flow): has start + end
                # 2. Point-in-time (balance sheet): has end only
                #
                # For duration entries: filter by period ~1 year (340-400 days)
                # For point-in-time: filter by end year matching fiscal year
                # Both: require end year to match fy (or fy+1 for Jan FY-ends)
                try:
                    if end:
                        d_end = _dt.strptime(end, '%Y-%m-%d')
                        end_year = d_end.year
                        if end_year != fy and end_year != fy + 1:
                            continue
                    if start and end:
                        d_start = _dt.strptime(start, '%Y-%m-%d')
                        d_end = _dt.strptime(end, '%Y-%m-%d')
                        days = (d_end - d_start).days
                        if days < 340 or days > 400:
                            continue
                except (ValueError, TypeError):
                    pass  # If dates can't be parsed, keep the entry

                candidates.append((fy, filed, val))

            # Deduplicate by fiscal year — keep the latest filed date
            for fy, filed, val in candidates:
                if fy not in merged or filed > merged[fy][0]:
                    merged[fy] = (filed, val)

        if not merged:
            return {}

        return {fy: info[1] for fy, info in sorted(merged.items())}

    # ------------------------------------------------------------------
    # Public API: Validation
    # ------------------------------------------------------------------

    def validate_against_yfinance(self, ticker, yf_financials):
        """Cross-check yfinance financial data against SEC EDGAR XBRL.

        Compares the most recent annual values for key financial concepts.
        Flags discrepancies where the percentage difference exceeds 5%.

        Args:
            ticker: Stock ticker symbol.
            yf_financials: dict with 'balance_sheet', 'income_statement',
                          'cash_flow', 'info' from yfinance.

        Returns:
            dict with validation results, or None on failure.
        """
        facts = self.fetch_company_facts(ticker)
        if not facts:
            return None

        discrepancies = []
        fields_checked = 0
        fields_flagged = 0

        for concept, yf_keys in self._YF_KEY_MAP.items():
            # Get EDGAR value (most recent annual)
            xbrl_tags = self._XBRL_TAG_MAP.get(concept, [])
            annual_vals = self._extract_annual_values(facts, xbrl_tags)
            if not annual_vals:
                continue

            edgar_year = max(annual_vals.keys())
            edgar_val = annual_vals[edgar_year]

            # Get yfinance value from the corresponding statement
            stmt_key = self._YF_STATEMENT_MAP.get(concept)
            if not stmt_key:
                continue
            stmt = yf_financials.get(stmt_key)
            if stmt is None or stmt.empty:
                continue

            # Get latest year column
            try:
                latest_col = stmt.iloc[:, 0]
                yf_val = _get(latest_col, yf_keys)
            except (IndexError, KeyError):
                continue

            if yf_val is None or edgar_val is None:
                continue

            # Compute percentage difference
            max_abs = max(abs(yf_val), abs(edgar_val))
            if max_abs == 0:
                pct_diff = 0.0
            else:
                pct_diff = abs(yf_val - edgar_val) / max_abs

            flagged = pct_diff > 0.05
            fields_checked += 1
            if flagged:
                fields_flagged += 1

            discrepancies.append({
                'field': concept,
                'yfinance': yf_val,
                'edgar': edgar_val,
                'edgar_year': edgar_year,
                'pct_diff': round(pct_diff, 4),
                'flagged': flagged,
            })

        if fields_checked == 0:
            return None

        return {
            'validated': True,
            'discrepancies': discrepancies,
            'fields_checked': fields_checked,
            'fields_flagged': fields_flagged,
            'edgar_quality_score': max(0, 100 - (20 * fields_flagged)),
        }

    # ------------------------------------------------------------------
    # Public API: Historical financials
    # ------------------------------------------------------------------

    def fetch_historical_financials(self, ticker, min_years=10):
        """Fetch long-duration annual revenue and earnings from EDGAR XBRL.

        Returns up to 20+ years of history (vs yfinance's typical 4 years).

        Args:
            ticker: Stock ticker symbol.
            min_years: Minimum years desired (informational, not enforced).

        Returns:
            dict with revenue_history, earnings_history, years_available,
            or None on failure.
        """
        facts = self.fetch_company_facts(ticker)
        if not facts:
            return None

        rev_tags = self._XBRL_TAG_MAP['revenue']
        ni_tags = self._XBRL_TAG_MAP['net_income']

        revenue_history = self._extract_annual_values(facts, rev_tags)
        earnings_history = self._extract_annual_values(facts, ni_tags)

        if not revenue_history and not earnings_history:
            return None

        years_available = max(len(revenue_history), len(earnings_history))

        return {
            'revenue_history': revenue_history,
            'earnings_history': earnings_history,
            'years_available': years_available,
        }
