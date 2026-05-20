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

# Module-level FX cache. Keyed by ISO currency code; value is {fiscal_year: rate_to_USD}.
# Built lazily on first request per currency so a 50-foreign-ticker batch hits
# yfinance at most ~10 times instead of once per ticker.
_FX_CACHE = {}


def _get_fx_rates_to_usd(currency):
    """Return {fiscal_year: ccy→USD rate} using yfinance year-end closes.

    Falls back to {} if yfinance is unavailable or the pair has no data,
    in which case downstream conversion silently passes through values
    unchanged. That's acceptable for HKD (pegged ~7.78) and small CCY
    drift but introduces error for volatile currencies — Phase 2 logs
    `reporting_currency` so consumers can audit when this happens.
    """
    if currency in _FX_CACHE:
        return _FX_CACHE[currency]
    if currency == 'USD':
        _FX_CACHE[currency] = {}
        return {}
    try:
        import yfinance as yf
        pair = f'{currency}USD=X'
        df = yf.download(pair, period='max', interval='1d',
                         progress=False, auto_adjust=False)
        if df is None or df.empty:
            _FX_CACHE[currency] = {}
            return {}
        # yfinance returns a MultiIndex-columned DataFrame even for a single
        # ticker; pluck the Close column and squeeze to a Series.
        if 'Close' in df.columns.get_level_values(0):
            close = df['Close']
        else:
            close = df.iloc[:, [0]]
        if hasattr(close, 'squeeze'):
            close = close.squeeze('columns') if close.ndim == 2 else close
        rates = {}
        for ts, val in close.items():
            try:
                v = float(val)
            except (TypeError, ValueError):
                continue
            # Iteration is chronological, so later in-year quotes overwrite
            # earlier ones — year-end close wins.
            rates[ts.year] = v
        _FX_CACHE[currency] = rates
        return rates
    except Exception:
        _FX_CACHE[currency] = {}
        return {}


def _apply_fx_annual(values, fx_rates):
    """Multiply each {year: value} by the matching FX rate.

    Years missing from fx_rates are passed through unchanged — pre-2003 EUR
    or other thin-history pairs will retain native-currency magnitudes,
    which is at least directionally correct for trend signal and never
    worse than dropping the year.
    """
    if not values or not fx_rates:
        return values
    out = {}
    for y, v in values.items():
        try:
            y_int = int(y)
        except (TypeError, ValueError):
            out[y] = v
            continue
        rate = fx_rates.get(y_int)
        out[y] = v * rate if (rate is not None and v is not None) else v
    return out


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
        # Needed by the build_yfinance_shape adapter so calculate_roic /
        # calculate_wacc can derive a real (not default 21%) tax rate.
        'tax_provision': [
            'IncomeTaxExpenseBenefit',
        ],
        'pretax_income': [
            'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest',
            'IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments',
            'IncomeLossFromContinuingOperationsBeforeIncomeTaxesAndMinorityInterest',
        ],
    }

    # IFRS taxonomy tag names. 20-F / 40-F filers (foreign issuers) report under
    # facts['ifrs-full'] instead of facts['us-gaap']. Keys mirror _XBRL_TAG_MAP
    # so the same concept lookups work across both taxonomies.
    _XBRL_TAG_MAP_IFRS = {
        'revenue': [
            'Revenue',
            'RevenueFromContractsWithCustomers',
        ],
        'net_income': [
            'ProfitLoss',
            'ProfitLossAttributableToOwnersOfParent',
        ],
        'total_assets': [
            'Assets',
        ],
        'total_equity': [
            'Equity',
            'EquityAttributableToOwnersOfParent',
        ],
        'total_debt': [
            'NoncurrentBorrowings',
            'BorrowingsNoncurrent',
            'LongtermBorrowings',
        ],
        'operating_income': [
            'ProfitLossFromOperatingActivities',
            'OperatingProfitLoss',
        ],
        'cash': [
            'CashAndCashEquivalents',
        ],
        'operating_cash_flow': [
            'CashFlowsFromUsedInOperatingActivities',
        ],
        'capex': [
            'PurchaseOfPropertyPlantAndEquipment',
            'PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities',
            # SAP / SONY / many EU filers report CapEx as additions on PPE
            'AdditionsOtherThanThroughBusinessCombinationsPropertyPlantAndEquipment',
            'PaymentsForPropertyPlantAndEquipment',
            # SAP combined PPE + intangibles tag — better than nothing
            'PurchaseOfPropertyPlantAndEquipmentIntangibleAssetsOtherThanGoodwillInvestmentPropertyAndOtherNoncurrentAssets',
            'AcquisitionsThroughBusinessCombinationsPropertyPlantAndEquipment',
        ],
        'gross_profit': [
            'GrossProfit',
        ],
        'interest_expense': [
            'FinanceCosts',
            'InterestExpense',
        ],
        'dividends_paid': [
            'DividendsPaid',
            'DividendsPaidClassifiedAsFinancingActivities',
            'DividendsPaidToEquityHoldersOfParentClassifiedAsFinancingActivities',
        ],
        'shares_outstanding': [
            'NumberOfSharesOutstanding',
            'NumberOfSharesIssued',
            'WeightedAverageNumberOfOrdinarySharesOutstanding',
        ],
        'tax_provision': [
            'IncomeTaxExpenseContinuingOperations',
        ],
        'pretax_income': [
            'ProfitLossBeforeTax',
        ],
    }

    _TAXONOMIES = (('us-gaap', _XBRL_TAG_MAP), ('ifrs-full', _XBRL_TAG_MAP_IFRS))

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

    def _extract_annual_values(self, facts_json, tag_list, form_filter='10-K',
                               units_key='USD', taxonomy_key='us-gaap'):
        """Extract annual values for a concept from XBRL facts.

        Tries each tag in tag_list until one has data.  Filters for the
        specified form type.  Handles the complexity of XBRL entries which
        include segment breakdowns and prior-year restatements alongside
        total annual figures.

        Selection logic per fiscal year:
        1. Only 10-K / 20-F / 40-F filings with fp='FY'
        2. Period must span ~1 full year (340-400 days)
        3. Period end year must match the fiscal year
        4. If multiple match, keep the latest filed date

        Args:
            facts_json: Raw companyfacts JSON.
            tag_list: List of XBRL tag names to try (US-GAAP or IFRS).
            form_filter: Filing form type ('10-K', '20-F', or '40-F').
            taxonomy_key: 'us-gaap' (default) or 'ifrs-full' for foreign issuers.

        Returns:
            dict {fiscal_year (int): value (float)} sorted by year,
            or empty dict if no data found.
        """
        if not facts_json:
            return {}

        us_gaap = facts_json.get('facts', {}).get(taxonomy_key, {})

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
                if form not in (form_filter, '20-F', '40-F'):
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
                                 point_in_time=False, taxonomy_key='us-gaap'):
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

        us_gaap = facts_json.get('facts', {}).get(taxonomy_key, {})

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
                if form not in ('10-K', '10-Q', '20-F', '40-F'):
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
    # Dual-taxonomy + currency-aware concept resolution
    # ------------------------------------------------------------------

    def _detect_currency(self, facts_json, concept):
        """Return the reporting currency for a concept across both taxonomies.

        Inspects the unit keys present on each candidate tag, returning the
        first ISO-shaped currency code (3 uppercase letters, not 'shares'
        or 'pure'). Prefers USD if available, else falls back to whichever
        currency the foreign filer actually uses.

        Returns None if no currency-keyed entries exist for any candidate tag.
        """
        if not facts_json:
            return None
        facts = facts_json.get('facts', {})
        for taxonomy_key, tag_map in self._TAXONOMIES:
            ns = facts.get(taxonomy_key, {})
            if not ns:
                continue
            for tag in tag_map.get(concept, []):
                units = ns.get(tag, {}).get('units', {})
                if 'USD' in units:
                    return 'USD'
                for k in units:
                    if len(k) == 3 and k.isalpha() and k.isupper():
                        return k
        return None

    def _extract_concept_annual(self, facts_json, concept, units_key=None):
        """Try US-GAAP first, fall back to IFRS, auto-detect currency.

        Returns (values_dict, taxonomy_key, currency) tuple. values_dict is
        empty if no entries were found in either taxonomy.

        units_key: explicit override (e.g. 'shares' for share counts). If
        None, the method auto-picks USD or the foreign filer's currency.
        """
        ccy = units_key or self._detect_currency(facts_json, concept) or 'USD'
        for taxonomy_key, tag_map in self._TAXONOMIES:
            tags = tag_map.get(concept, [])
            if not tags:
                continue
            vals = self._extract_annual_values(
                facts_json, tags, units_key=ccy, taxonomy_key=taxonomy_key)
            if vals:
                return vals, taxonomy_key, ccy
        return {}, None, ccy

    def _extract_concept_periodic(self, facts_json, concept, units_key=None,
                                  point_in_time=False):
        """Periodic equivalent of _extract_concept_annual.

        Returns (values_dict, taxonomy_key, currency) tuple.
        """
        ccy = units_key or self._detect_currency(facts_json, concept) or 'USD'
        for taxonomy_key, tag_map in self._TAXONOMIES:
            tags = tag_map.get(concept, [])
            if not tags:
                continue
            vals = self._extract_periodic_values(
                facts_json, tags, units_key=ccy,
                point_in_time=point_in_time, taxonomy_key=taxonomy_key)
            if vals:
                return vals, taxonomy_key, ccy
        return {}, None, ccy

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

        Flow concepts (revenue, net income, OCF, capex, etc.) are returned
        as one value per fiscal year keyed by integer year — sourced from
        10-K / 20-F / 40-F annual filings (US-GAAP or IFRS taxonomy).
        Shares outstanding remains point-in-time keyed by period-end date
        ("YYYY-MM-DD"), since balance-sheet items carry useful intra-year
        detail.

        Foreign-currency filings (EUR, JPY, DKK, etc.) are auto-detected
        and converted to USD using year-end FX rates from yfinance, so
        the returned values are always USD-denominated.

        Args:
            ticker: Stock ticker symbol.
            min_years: Minimum years desired (informational, not enforced).

        Returns:
            dict with revenue_history, earnings_history, years_available,
            reporting_currency, fx_converted, or None on failure.
        """
        facts = self.fetch_company_facts(ticker)
        if not facts:
            return None

        def _flow(concept):
            vals, _tax, ccy = self._extract_concept_annual(facts, concept)
            return vals, ccy

        rev, rev_ccy             = _flow('revenue')
        ni,  ni_ccy              = _flow('net_income')
        ocf, ocf_ccy             = _flow('operating_cash_flow')
        capex, capex_ccy         = _flow('capex')
        gp, gp_ccy               = _flow('gross_profit')
        intexp, intexp_ccy       = _flow('interest_expense')
        div, div_ccy             = _flow('dividends_paid')
        shares, _tax_s, _ccy_s   = self._extract_concept_periodic(
            facts, 'shares_outstanding', units_key='shares', point_in_time=True)

        if not rev and not ni:
            return None

        # The reporting currency is whichever non-USD currency appears on the
        # primary income-statement concepts. If revenue is in JPY but a US-GAAP
        # subsidiary tag happens to carry USD on, say, dividends, treat the
        # filer as JPY.
        currencies = [c for c in (rev_ccy, ni_ccy, ocf_ccy, capex_ccy,
                                  gp_ccy, intexp_ccy, div_ccy) if c]
        reporting_ccy = next((c for c in currencies if c != 'USD'), 'USD')
        fx_converted = reporting_ccy != 'USD'

        if fx_converted:
            fx = _get_fx_rates_to_usd(reporting_ccy)
            rev    = _apply_fx_annual(rev, fx)
            ni     = _apply_fx_annual(ni, fx)
            ocf    = _apply_fx_annual(ocf, fx)
            capex  = _apply_fx_annual(capex, fx)
            gp     = _apply_fx_annual(gp, fx)
            intexp = _apply_fx_annual(intexp, fx)
            div    = _apply_fx_annual(div, fx)
            # shares are unit-counts, not currency — leave alone.

        all_series = [rev, ni, ocf, capex, gp, intexp, div, shares]
        years_available = max((len(s) for s in all_series if s), default=0)

        return {
            'revenue_history':          rev,
            'earnings_history':         ni,
            'operating_cf_history':     ocf,
            'capex_history':            capex,
            'gross_profit_history':     gp,
            'interest_expense_history': intexp,
            'dividends_paid_history':   div,
            'shares_history':           shares,
            'years_available':          years_available,
            'reporting_currency':       reporting_ccy,
            'fx_converted':             fx_converted,
        }

    def build_yfinance_shape(self, ticker, year_limit=None):
        """Construct a yfinance-shaped financials dict from SEC XBRL data.

        Returned shape matches what models/ratios.py expects:
        {balance_sheet, income_statement, cash_flow, info}, with statement
        DataFrames indexed by line-item label and columned by fiscal-year-end
        Timestamp (latest year first).

        info is sparse: only carries `symbol` and a `_source` tag. Callers
        that want CAPM / market-cap-weighted WACC should merge yfinance's
        info dict on top of this result.

        Args:
            ticker: Stock ticker symbol.
            year_limit: Optional cap on the number of year columns kept
                (latest first). Defaults to None — return all available
                history (typically 10-16 years). For a value-investor /
                DCF pipeline this is preferable: through-cycle ROIC and
                tax rates smooth out one-off years and are a more defensible
                input to long-horizon terminal-value calculations.

        Returns:
            dict or None if no XBRL data is available for ticker.
        """
        import pandas as pd

        facts = self.fetch_company_facts(ticker)
        if not facts:
            return None

        def _ann(concept):
            tags = self._XBRL_TAG_MAP.get(concept, [])
            return self._extract_annual_values(facts, tags) if tags else {}

        # Income statement (flow concepts)
        revenue       = _ann('revenue')
        net_income    = _ann('net_income')
        op_income     = _ann('operating_income')
        gross_profit  = _ann('gross_profit')
        interest_exp  = _ann('interest_expense')
        tax_provision = _ann('tax_provision')
        pretax_income = _ann('pretax_income')

        # Cash flow (flow concepts)
        op_cf         = _ann('operating_cash_flow')

        # Balance sheet (point-in-time concepts). Their entries in XBRL only
        # carry end dates, no durations — _extract_annual_values' duration
        # filter conditional skips them naturally, and the fy match keeps the
        # right period-end value per fiscal year.
        equity        = _ann('total_equity')
        debt          = _ann('total_debt')
        cash          = _ann('cash')
        assets        = _ann('total_assets')

        # Need at least revenue or net income to consider the data usable.
        if not revenue and not net_income:
            return None

        years = sorted(set(revenue) | set(net_income) | set(op_income) |
                       set(equity) | set(debt) | set(assets) | set(pretax_income),
                       reverse=True)
        if not years:
            return None

        # Truncate to the most recent year_limit years (latest first).
        # XBRL routinely returns 10–16 years; a long window pulls in pre-
        # crisis filings that yfinance no longer reports, producing ROIC
        # values that diverge meaningfully from the recent window.
        if year_limit and year_limit > 0:
            years = years[:year_limit]

        # ----- Derive missing line items from related XBRL fields -----
        # XBRL tagging varies across companies — many file pretax_income but not
        # operating_income (NKE), or operating_income but not pretax_income (REITs
        # like BXP). The accounting identity is pretax ≈ operating - interest_exp
        # (treating other_income/expense as ≈ 0). Use it to fill missing entries
        # so calculate_roic's all([op, pretax, equity]) gate can pass. Never
        # overwrite a value that XBRL already provided.
        for y in years:
            op = op_income.get(y)
            pti = pretax_income.get(y)
            intexp = interest_exp.get(y) or 0
            if op is None and pti is not None:
                op_income[y] = pti + intexp
            elif pti is None and op is not None:
                pretax_income[y] = op - intexp
            # Last-resort derivation: pretax = net_income + tax_provision
            if pretax_income.get(y) is None:
                ni = net_income.get(y)
                tx = tax_provision.get(y)
                if ni is not None and tx is not None:
                    pretax_income[y] = ni + tx

        cols = [pd.Timestamp(year=y, month=12, day=31) for y in years]

        income_df = pd.DataFrame({
            col: {
                'Total Revenue':    revenue.get(y),
                'Net Income':       net_income.get(y),
                'Operating Income': op_income.get(y),
                'Gross Profit':     gross_profit.get(y),
                'Interest Expense': interest_exp.get(y),
                'Tax Provision':    tax_provision.get(y),
                'Pretax Income':    pretax_income.get(y),
            } for y, col in zip(years, cols)
        })

        balance_df = pd.DataFrame({
            col: {
                'Stockholders Equity':       equity.get(y),
                'Total Debt':                debt.get(y),
                'Cash And Cash Equivalents': cash.get(y),
                'Total Assets':              assets.get(y),
            } for y, col in zip(years, cols)
        })

        cash_flow_df = pd.DataFrame({
            col: {
                'Operating Cash Flow': op_cf.get(y),
            } for y, col in zip(years, cols)
        })

        return {
            'balance_sheet':    balance_df,
            'income_statement': income_df,
            'cash_flow':        cash_flow_df,
            'info':             {'symbol': ticker, '_source': 'sec_xbrl'},
            'growth_estimates': None,
            'earnings_history': None,
        }
