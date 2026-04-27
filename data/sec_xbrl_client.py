"""SEC EDGAR XBRL Company Facts client.

Fetches structured financial data from the SEC XBRL API to:
1. Cross-validate yfinance financial statements (flag discrepancies > 5%)
2. Provide 10+ years of revenue/earnings history for extended CAGR analysis

Free API — no authentication required, just User-Agent with email.
Uses only stdlib (urllib + json + datetime).
"""

import json
import ssl
import time
import urllib.error
import urllib.request
from datetime import datetime as _dt

def _ssl_context():
    """Return an SSL context using certifi certs if available, else system certs."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx

_SSL_CTX = _ssl_context()

from models.field_keys import (
    _get, REVENUE_KEYS, NET_INCOME_KEYS, TOTAL_ASSETS_KEYS,
    EQUITY_KEYS, DEBT_KEYS, OPERATING_INCOME_KEYS, CASH_KEYS,
    OPERATING_CF_KEYS, GROSS_PROFIT_KEYS, INTEREST_KEYS,
    CAPEX_KEYS, DIVIDENDS_PAID_KEYS,
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
        'operating_cash_flow': [
            'NetCashProvidedByUsedInOperatingActivities',
        ],
        'capex': [
            'PaymentsToAcquirePropertyPlantAndEquipment',
            'PaymentsForCapitalImprovements',
            'CapitalExpendituresIncurringObligation',
        ],
        'gross_profit': [
            'GrossProfit',
        ],
        'interest_expense': [
            'InterestExpense',
            'InterestAndDebtExpense',
            'InterestExpenseDebt',
        ],
        'dividends_paid': [
            'PaymentsOfDividendsCommonStock',
            'PaymentsOfDividends',
            'DividendsCommonStockCash',
        ],
        'shares_outstanding': [
            'CommonStockSharesOutstanding',
            'CommonStockSharesOutstandingBasic',
        ],
    }

    # Map XBRL concept names to yfinance field_keys lists for cross-validation
    _YF_KEY_MAP = {
        'revenue': REVENUE_KEYS,
        'net_income': NET_INCOME_KEYS,
        'total_assets': TOTAL_ASSETS_KEYS,
        'total_equity': EQUITY_KEYS,
        'total_debt': DEBT_KEYS,
        'operating_cash_flow': OPERATING_CF_KEYS,
        'gross_profit': GROSS_PROFIT_KEYS,
        'interest_expense': INTEREST_KEYS,
    }

    # Which yfinance statement contains each concept
    _YF_STATEMENT_MAP = {
        'revenue': 'income_statement',
        'net_income': 'income_statement',
        'total_assets': 'balance_sheet',
        'total_equity': 'balance_sheet',
        'total_debt': 'balance_sheet',
        'operating_cash_flow': 'cash_flow',
        'gross_profit': 'income_statement',
        'interest_expense': 'income_statement',
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
            with urllib.request.urlopen(req, context=_SSL_CTX, timeout=timeout) as resp:
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

    def _extract_annual_values(self, facts_json, tag_list, form_filter='10-K', units_key='USD'):
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

            units = concept.get('units', {})
            entries = units.get(units_key, [])
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

    def _extract_periodic_values(self, facts_json, tag_list, units_key='USD',
                                 point_in_time=False):
        """Extract per-period (quarterly + annual) values for a flow/stock concept.

        For *flow* concepts (revenue, net income, OCF, capex, etc.), accepts
        entries with start+end durations of ~1 quarter (80–100 days) or
        ~1 year (340–400 days). YTD partials (6/9-month cumulative figures
        filed in Q2/Q3 10-Qs) are dropped so plotted magnitudes are
        comparable across points.

        Then, for each annual entry at end-date X, looks for ~3 quarterly
        entries with end-dates in (X − 380d, X). If found, derives Q4 =
        FY − sum(Q1+Q2+Q3) and replaces the annual point with four
        quarterly points. Years without quarterly coverage keep the
        annual point.

        For *stock* (point-in-time) concepts like shares outstanding, set
        ``point_in_time=True``: every dated entry is accepted (deduped by
        latest filed per end date).

        Returns:
            dict {period_end (YYYY-MM-DD str): value (float)} sorted by date.
        """
        if not facts_json:
            return {}

        us_gaap = facts_json.get('facts', {}).get('us-gaap', {})

        # Stage 1: collect entries deduped by (end, kind) where kind is "Q",
        # "FY", or "PT" (point-in-time). XBRL filings include prior-period
        # comparatives — different filings restate the same period, so we
        # keep the latest filed version per (end, kind).
        merged = {}  # {(end_str, kind): (filed_str, val)}

        for tag in tag_list:
            concept = us_gaap.get(tag)
            if not concept:
                continue
            entries = concept.get('units', {}).get(units_key, [])
            if not entries:
                continue

            for e in entries:
                form = e.get('form', '')
                if form not in ('10-K', '10-Q', '20-F'):
                    continue
                val = e.get('val')
                filed = e.get('filed', '')
                start = e.get('start', '')
                end = e.get('end', '')
                if val is None or not end:
                    continue

                if point_in_time:
                    kind = 'PT'
                else:
                    if not start:
                        continue
                    try:
                        d_start = _dt.strptime(start, '%Y-%m-%d')
                        d_end = _dt.strptime(end, '%Y-%m-%d')
                    except (ValueError, TypeError):
                        continue
                    days = (d_end - d_start).days
                    if 80 <= days <= 100:
                        kind = 'Q'
                    elif 340 <= days <= 400:
                        kind = 'FY'
                    else:
                        # YTD partial (180/270d) or other — drop.
                        continue

                key = (end, kind)
                prev = merged.get(key)
                if prev is None or filed > prev[0]:
                    merged[key] = (filed, val)

        if not merged:
            return {}

        if point_in_time:
            # Collapse to {end_date: val}.
            return {end: info[1] for (end, _k), info in sorted(merged.items())}

        # Stage 2: derive Q4 for each annual entry where 3 quarterly entries
        # ending in the prior ~380 days are present. End-date keys are
        # YYYY-MM-DD strings, so chronological sort = lexicographic sort.
        quarters = sorted([end for (end, k) in merged if k == 'Q'])
        annuals = sorted([end for (end, k) in merged if k == 'FY'])

        out = {}  # {end_date_str: val}

        # Add all quarterly entries as-is.
        for q_end in quarters:
            out[q_end] = merged[(q_end, 'Q')][1]

        # For each annual: find ~3 quarterly entries within the prior year.
        # If ≥3 found, derive a Q4 point at the annual's end-date.
        # Otherwise keep the annual point as-is at its end-date.
        for fy_end in annuals:
            try:
                fy_d = _dt.strptime(fy_end, '%Y-%m-%d')
            except ValueError:
                continue
            # Take quarterlies within (fy_end − 380d, fy_end]
            in_window = []
            for q_end in quarters:
                if q_end > fy_end:
                    continue
                try:
                    q_d = _dt.strptime(q_end, '%Y-%m-%d')
                except ValueError:
                    continue
                gap = (fy_d - q_d).days
                if 0 < gap <= 380:
                    in_window.append((q_end, merged[(q_end, 'Q')][1]))

            if len(in_window) >= 3:
                # Take the 3 most recent (largest gap excluded if 4 found).
                in_window.sort(key=lambda x: x[0])
                q3 = in_window[-3:]
                q_sum = sum(v for _, v in q3)
                fy_val = merged[(fy_end, 'FY')][1]
                # Derived Q4 sits at the annual end-date.
                out[fy_end] = fy_val - q_sum
            else:
                # No quarterly coverage — keep the annual at year-end. (Mixed
                # magnitude is acceptable: pre-2009 or foreign filers will
                # only have annual points.)
                out[fy_end] = merged[(fy_end, 'FY')][1]

        return dict(sorted(out.items()))

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
        """Fetch long-duration revenue and earnings history from EDGAR XBRL.

        Returns ~15+ years of quarterly + annual history. Income/cash-flow
        series are quarterly where 10-Qs are available (with Q4 derived from
        annual − Q1−Q2−Q3) and fall back to annual-only for years without
        quarterly coverage. Shares outstanding is point-in-time at every
        period end. Each series is keyed by period-end date string
        ("YYYY-MM-DD").

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

        def _flow(concept, units_key='USD'):
            return self._extract_periodic_values(
                facts, self._XBRL_TAG_MAP[concept], units_key=units_key)

        def _stock(concept, units_key='USD'):
            return self._extract_periodic_values(
                facts, self._XBRL_TAG_MAP[concept], units_key=units_key,
                point_in_time=True)

        revenue_history          = _flow('revenue')
        earnings_history         = _flow('net_income')
        operating_cf_history     = _flow('operating_cash_flow')
        capex_history            = _flow('capex')
        gross_profit_history     = _flow('gross_profit')
        interest_expense_history = _flow('interest_expense')
        dividends_paid_history   = _flow('dividends_paid')
        shares_history           = _stock('shares_outstanding', units_key='shares')

        if not revenue_history and not earnings_history:
            return None

        all_series = [
            revenue_history, earnings_history, operating_cf_history,
            capex_history, gross_profit_history, interest_expense_history,
            dividends_paid_history, shares_history,
        ]
        years_available = max((len(s) for s in all_series if s), default=0)

        return {
            'revenue_history':          revenue_history,
            'earnings_history':         earnings_history,
            'operating_cf_history':     operating_cf_history,
            'capex_history':            capex_history,
            'gross_profit_history':     gross_profit_history,
            'interest_expense_history': interest_expense_history,
            'dividends_paid_history':   dividends_paid_history,
            'shares_history':           shares_history,
            'years_available':          years_available,
        }
