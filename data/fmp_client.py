# data/fmp_client.py
"""Financial Modeling Prep (FMP) data client — drop-in replacement for YFinanceClient.

Requires FMP_API_KEY environment variable (or pass api_key to constructor).

Uses the FMP /stable/ API (new endpoint structure as of Sep 2025).
All requests use ?apikey=KEY&symbol=TICKER query params (not URL path params).

Plan coverage:
  - All endpoints below are available on the free/starter plan unless noted.
  - Earnings surprises, analyst recommendations, short interest: NOT available
    on the plan associated with the test key (404 at /stable/).
    These fields return None and the pipeline handles them gracefully.

Coverage vs yfinance:
  IMPROVED  : sector/industry, ROIC, EV/EBITDA, fundamentals reliability
  EQUIVALENT: price history, dividends, financials, market cap metrics
  MISSING   : recommendationKey/numberOfAnalystOpinions (no stable endpoint),
              heldPercentInsiders/Institutions, sharesShort/shortRatio,
              earnings_history surprisePercent (404 on this plan)

Statement methodology:
  - All statements: annual period, 4 most recent fiscal years
  - Matches yfinance .balance_sheet / .financials / .cashflow (annual)
"""

import os
import time
import json
from datetime import date, timedelta
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
import pandas as pd

_BASE = 'https://financialmodelingprep.com/stable'


# ---------------------------------------------------------------------------
# Field mappings: FMP JSON key → yfinance-compatible row name
# Must match canonical names in models/field_keys.py
# ---------------------------------------------------------------------------
_IS_MAP = {
    'revenue':                                  'Total Revenue',
    'grossProfit':                              'Gross Profit',
    'operatingIncome':                          'Operating Income',
    'netIncome':                                'Net Income',
    'ebitda':                                   'EBITDA',
    'incomeBeforeTax':                          'Pretax Income',
    'interestExpense':                          'Interest Expense',
    'incomeTaxExpense':                         'Tax Provision',
    'sellingGeneralAndAdministrativeExpenses':  'Selling General And Administration',
    'researchAndDevelopmentExpenses':           'Research And Development',
}

_BS_MAP = {
    'totalStockholdersEquity':  'Stockholders Equity',
    'totalAssets':              'Total Assets',
    'cashAndCashEquivalents':   'Cash And Cash Equivalents',
    'goodwill':                 'Goodwill',
    'totalCurrentAssets':       'Current Assets',
    'totalCurrentLiabilities':  'Current Liabilities',
    'totalDebt':                'Total Debt',
    'longTermDebt':             'Long Term Debt',
    'netReceivables':           'Accounts Receivable',
    'propertyPlantEquipmentNet':'Net PPE',
}

_CF_MAP = {
    'freeCashFlow':             'Free Cash Flow',
    'operatingCashFlow':        'Operating Cash Flow',
    'depreciationAndAmortization': 'Depreciation And Amortization',
    'commonDividendsPaid':      'Common Stock Dividend Paid',
    'commonStockRepurchased':   'Repurchase Of Capital Stock',
}


class FMPClient:
    def __init__(self, api_key=None, request_delay=0.3):
        self._api_key = api_key or os.environ.get('FMP_API_KEY', '')
        self._request_delay = request_delay
        self._last_request_time = 0
        self._cache = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _throttle(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self._request_delay:
            time.sleep(self._request_delay - elapsed)
        self._last_request_time = time.time()

    def _get(self, path, params=None):
        """GET from FMP /stable/ API.  Returns parsed JSON or None on error."""
        self._throttle()
        url = _BASE + path + '?apikey=' + self._api_key
        if params:
            for k, v in params.items():
                url += f'&{k}={v}'
        req = Request(url, headers={'Accept': 'application/json'})
        try:
            with urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode())
        except HTTPError as e:
            if e.code in (401, 402, 403, 404):
                return None   # Plan or endpoint limitation — degrade gracefully
            raise
        except URLError:
            return None

    def _build_statement_df(self, records, code_map, n_periods=4):
        """Convert a list of FMP statement dicts to a yfinance-style DataFrame.

        Returns DataFrame with row=field name, columns=dates descending.
        """
        if not records:
            return pd.DataFrame()
        periods = records[:n_periods]
        columns = {}
        for rec in periods:
            dt = pd.Timestamp(rec.get('date', '1970-01-01')).tz_localize(None)
            columns[dt] = {
                code_map[k]: rec[k]
                for k in code_map
                if rec.get(k) is not None
            }
        dates = sorted(columns.keys(), reverse=True)
        all_fields = set(f for d in columns.values() for f in d)
        out = {f: {dt: columns[dt].get(f) for dt in dates} for f in all_fields}
        df = pd.DataFrame(out).T
        df.columns = dates
        return df

    def _build_growth_estimates(self, estimates):
        """Build growth_estimates DataFrame mirroring yfinance structure.

        yfinance index: 'LTG', '+1y', '0y'  /  column: 'stockTrend'
        FMP /analyst-estimates returns annual forward EPS only (no historical).
        Sort ascending by date; compute growth between consecutive future periods.
          - '+1y' = nearest future year EPS growth vs the year before it
          - '0y'  = year-before-nearest vs the year before that
          - 'LTG' = None (requires Premium plan)
        """
        if not estimates or len(estimates) < 2:
            return None
        try:
            # Sort ascending so index 0 = nearest period
            asc = sorted(estimates, key=lambda e: e.get('date', ''))

            def _growth(newer, older):
                eps_n = newer.get('epsAvg')
                eps_o = older.get('epsAvg')
                if eps_n is not None and eps_o and eps_o != 0:
                    return (eps_n - eps_o) / abs(eps_o)
                return None

            rows = {
                '+1y': _growth(asc[1], asc[0]) if len(asc) >= 2 else None,
                '0y':  _growth(asc[0], asc[-1]) if len(asc) >= 2 else None,
                'LTG': None,
            }
            non_null = {k: v for k, v in rows.items() if v is not None}
            if not non_null:
                return None
            return pd.DataFrame({'stockTrend': rows})
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Public interface — mirrors YFinanceClient exactly
    # ------------------------------------------------------------------

    def fetch_financials(self, ticker, as_of=None):
        """Return dict matching YFinanceClient.fetch_financials() structure."""
        cache_key = ('financials', ticker)
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = {
            'balance_sheet':    pd.DataFrame(),
            'income_statement': pd.DataFrame(),
            'cash_flow':        pd.DataFrame(),
            'info':             {},
            'growth_estimates': None,
            'earnings_history': None,  # Not available on current plan
        }

        # 1. Company profile
        try:
            profiles = self._get('/profile', {'symbol': ticker})
            if profiles:
                p = profiles[0]
                # Parse 52w range "low-high"
                low52, high52 = None, None
                rng = p.get('range', '')
                if rng and '-' in rng:
                    parts = rng.split('-')
                    try:
                        low52  = float(parts[0].strip())
                        high52 = float(parts[1].strip())
                    except (ValueError, IndexError):
                        pass

                result['info'].update({
                    'longName':            p.get('companyName', ''),
                    'shortName':           p.get('companyName', ''),
                    'longBusinessSummary': p.get('description', ''),
                    'sector':              p.get('sector', ''),
                    'industry':            p.get('industry', ''),
                    'currentPrice':        p.get('price'),
                    'regularMarketPrice':  p.get('price'),
                    'marketCap':           p.get('marketCap'),
                    'beta':                p.get('beta'),
                    'fiftyTwoWeekLow':     low52,
                    'fiftyTwoWeekHigh':    high52,
                    'averageVolume':       p.get('averageVolume'),
                    'dividendRate':        p.get('lastDividend'),
                })
        except Exception:
            pass

        # 2. Shares outstanding and float
        try:
            floats = self._get('/shares-float', {'symbol': ticker})
            if floats:
                f = floats[0]
                result['info']['floatShares']      = f.get('floatShares')
                result['info']['sharesOutstanding'] = f.get('outstandingShares')
        except Exception:
            pass

        # 3. TTM valuation ratios (no limit needed — single TTM record)
        try:
            ratios = self._get('/ratios-ttm', {'symbol': ticker})
            if ratios:
                r = ratios[0]
                result['info'].update({
                    'trailingPE':           r.get('priceToEarningsRatioTTM'),
                    'forwardPE':            r.get('priceToEarningsRatioTTM'),
                    'priceToBook':          r.get('priceToBookRatioTTM'),
                    'pegRatio':             r.get('priceEarningsToGrowthRatioTTM'),
                    'payoutRatio':          r.get('dividendPayoutRatioTTM'),
                    'trailingEps':          r.get('netIncomePerShareTTM'),
                    'bookValue':            r.get('bookValuePerShareTTM'),
                    'enterpriseToEbitda':   r.get('enterpriseValueMultipleTTM'),
                    'enterpriseToRevenue':  r.get('priceToSalesRatioTTM'),
                })
        except Exception:
            pass

        # 4. Annual key metrics (EV, ROIC — most recent fiscal year)
        try:
            fy_start = (date.today() - timedelta(days=400)).isoformat()
            metrics = self._get('/key-metrics',
                                {'symbol': ticker, 'period': 'annual', 'from': fy_start})
            if metrics:
                m = metrics[0]
                result['info']['enterpriseValue'] = m.get('enterpriseValue') or result['info'].get('enterpriseValue')
                # Only override PE/PB if TTM lookup failed
                if not result['info'].get('trailingPE'):
                    result['info']['trailingPE'] = m.get('peRatio')
                    result['info']['forwardPE']  = m.get('peRatio')
                if not result['info'].get('priceToBook'):
                    result['info']['priceToBook'] = m.get('pbRatio')
        except Exception:
            pass

        # 5. Revenue / earnings growth rates (most recent fiscal year)
        try:
            growth = self._get('/financial-growth',
                               {'symbol': ticker, 'period': 'annual', 'from': fy_start})
            if growth:
                g = growth[0]
                result['info']['revenueGrowth']  = g.get('revenueGrowth')
                result['info']['earningsGrowth'] = g.get('netIncomeGrowth')
        except Exception:
            pass

        # 6. Analyst price targets
        try:
            targets = self._get('/price-target-consensus', {'symbol': ticker})
            if targets:
                t = targets[0]
                result['info']['targetMeanPrice']   = t.get('targetConsensus')
                result['info']['targetHighPrice']   = t.get('targetHigh')
                result['info']['targetLowPrice']    = t.get('targetLow')
                result['info']['targetMedianPrice'] = t.get('targetMedian')
        except Exception:
            pass

        # 7. Key executives (companyOfficers)
        try:
            execs = self._get('/key-executives', {'symbol': ticker})
            if execs:
                result['info']['companyOfficers'] = [
                    {'name': e.get('name', ''), 'title': e.get('title', '')}
                    for e in execs[:5]
                ]
        except Exception:
            pass

        # 8. Financial statements (annual, ~4 years using from-date)
        stmt_start = (date.today() - timedelta(days=4 * 370)).isoformat()
        try:
            is_data = self._get('/income-statement',
                                {'symbol': ticker, 'period': 'annual', 'from': stmt_start})
            if is_data:
                result['income_statement'] = self._build_statement_df(is_data, _IS_MAP)
        except Exception:
            pass

        try:
            bs_data = self._get('/balance-sheet-statement',
                                {'symbol': ticker, 'period': 'annual', 'from': stmt_start})
            if bs_data:
                result['balance_sheet'] = self._build_statement_df(bs_data, _BS_MAP)
        except Exception:
            pass

        try:
            cf_data = self._get('/cash-flow-statement',
                                {'symbol': ticker, 'period': 'annual', 'from': stmt_start})
            if cf_data:
                result['cash_flow'] = self._build_statement_df(cf_data, _CF_MAP)
        except Exception:
            pass

        # 9. Analyst EPS estimates → growth_estimates DataFrame
        try:
            estimates = self._get('/analyst-estimates',
                                  {'symbol': ticker, 'period': 'annual'})
            result['growth_estimates'] = self._build_growth_estimates(estimates)
        except Exception:
            pass

        self._cache[cache_key] = result
        return result

    def fetch_dividends(self, ticker, period='10y'):
        """Return pandas Series of dividend dates/amounts.

        Matches YFinanceClient.fetch_dividends() interface.
        """
        cache_key = ('dividends', ticker, period)
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            years = int(period.replace('y', ''))
        except Exception:
            years = 10
        start = (date.today() - timedelta(days=years * 365)).isoformat()

        try:
            # Use 'from' date — avoids the plan limit on 'limit' param
            data = self._get('/dividends', {'symbol': ticker, 'from': start})
            if not data:
                return pd.Series(dtype=float)
            divs = {
                pd.Timestamp(d['date']).tz_localize(None): d.get('adjDividend') or d.get('dividend', 0)
                for d in data
                if d.get('date', '') >= start
                   and (d.get('adjDividend') or d.get('dividend', 0)) > 0
            }
            result = pd.Series(divs, dtype=float).sort_index()
        except Exception:
            result = pd.Series(dtype=float)

        self._cache[cache_key] = result
        return result

    def fetch_history(self, ticker, period='5y'):
        """Return pandas Series of daily close prices.

        Matches YFinanceClient.fetch_history() interface.
        Used for beta calculation in CAPM.
        """
        cache_key = ('history', ticker, period)
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            years = int(period.replace('y', ''))
        except Exception:
            years = 5
        start = (date.today() - timedelta(days=years * 365)).isoformat()

        try:
            # 'from' param returns all trading days since that date (no limit needed)
            data = self._get('/historical-price-eod/light',
                             {'symbol': ticker, 'from': start})
            if not data:
                return pd.Series(dtype=float)
            # API returns newest-first; sort ascending for time-series use
            data = sorted(data, key=lambda x: x['date'])
            idx  = [pd.Timestamp(p['date']).tz_localize(None) for p in data]
            vals = [p.get('price') for p in data]
            result = pd.Series(vals, index=idx, dtype=float)
        except Exception:
            result = pd.Series(dtype=float)

        self._cache[cache_key] = result
        return result

    def fetch_price_history(self, ticker, period='1y'):
        """Return list of {"d": "YYYY-MM-DD", "c": float} dicts.

        Matches YFinanceClient.fetch_price_history() interface.
        Used for the Charts tab in the HTML report.
        """
        cache_key = ('price_history', ticker, period)
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            years = int(period.replace('y', ''))
        except Exception:
            years = 1
        start = (date.today() - timedelta(days=years * 365)).isoformat()

        try:
            data = self._get('/historical-price-eod/light',
                             {'symbol': ticker, 'from': start})
            if not data:
                return []
            data = sorted(data, key=lambda x: x['date'])
            result = [
                {'d': p['date'], 'c': round(float(p['price']), 2)}
                for p in data
                if p.get('price') is not None
            ]
        except Exception:
            result = []

        self._cache[cache_key] = result
        return result
