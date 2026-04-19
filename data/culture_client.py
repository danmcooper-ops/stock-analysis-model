# data/culture_client.py
"""
Extracts company culture proxy metrics from yfinance data and optional
external sources (Glassdoor public pages).

Metrics extracted
-----------------
From yfinance info dict:
  employees          int   — fullTimeEmployees headcount
  ceo_total_pay      int   — CEO total compensation in dollars
  compensation_risk  int   — yfinance compensation risk score 1–10 (lower = better)

From yfinance cash_flow DataFrame:
  sbc                float — stock-based compensation (most recent fiscal year)

From Glassdoor (best-effort, graceful fallback):
  glassdoor_rating   float — overall employer rating 1.0–5.0
  glassdoor_ceo_pct  int   — % of employees who approve of the CEO
  glassdoor_rec_pct  int   — % who would recommend to a friend

All network calls are wrapped in try/except with short timeouts so a
Glassdoor failure never blocks the main pipeline.
"""

import time
import json
import urllib.request
import urllib.parse


_GLASSDOOR_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/123.0.0.0 Safari/537.36'
    ),
    'Accept': 'application/json, text/html, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.glassdoor.com/',
}

_glassdoor_cache: dict = {}   # company_name → result dict


class CultureClient:
    """Extract culture proxy metrics from yfinance data and Glassdoor."""

    def __init__(self, glassdoor_enabled: bool = True,
                 glassdoor_timeout: float = 5.0,
                 request_delay: float = 1.5):
        self._glassdoor_enabled = glassdoor_enabled
        self._glassdoor_timeout = glassdoor_timeout
        self._request_delay = request_delay
        self._last_gd_request = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, info: dict, financials: dict) -> dict:
        """Return culture metrics from yfinance info and financials dicts.

        Args:
            info:       The ``info`` dict from YFinanceClient.fetch_financials().
            financials: The full financials dict (cash_flow, income_statement…).

        Returns:
            Dict with keys: employees, ceo_total_pay, compensation_risk, sbc.
            Glassdoor fields are fetched separately via fetch_glassdoor().
        """
        result = {
            'employees': None,
            'ceo_total_pay': None,
            'compensation_risk': None,
            'sbc': None,
        }

        # --- Employee headcount -------------------------------------------
        employees = info.get('fullTimeEmployees')
        if employees is not None:
            try:
                employees = int(employees)
                result['employees'] = employees if employees > 0 else None
            except (TypeError, ValueError):
                pass

        # --- CEO total compensation ---------------------------------------
        officers = info.get('companyOfficers') or []
        ceo = next(
            (o for o in officers
             if 'ceo' in (o.get('title') or '').lower()
             or 'chief executive' in (o.get('title') or '').lower()),
            officers[0] if officers else None,
        )
        if ceo:
            pay = ceo.get('totalPay')
            if pay is not None:
                try:
                    ceo_pay = int(pay)
                    result['ceo_total_pay'] = ceo_pay if ceo_pay > 0 else None
                except (TypeError, ValueError):
                    pass

        # --- yfinance compensation risk score (1–10, lower = better) ----
        crisk = info.get('compensationRisk')
        if crisk is not None:
            try:
                result['compensation_risk'] = int(crisk)
            except (TypeError, ValueError):
                pass

        # --- Stock-based compensation (cash flow) -------------------------
        try:
            cf = financials.get('cash_flow')
            if cf is not None and not cf.empty:
                for label in ('Stock Based Compensation',
                              'StockBasedCompensation',
                              'Share Based Compensation'):
                    if label in cf.index:
                        val = cf.loc[label].iloc[0]
                        if val is not None:
                            import numpy as np
                            if not (isinstance(val, float) and np.isnan(val)):
                                result['sbc'] = float(val)
                                break
        except Exception:
            pass

        return result

    # ------------------------------------------------------------------
    # Glassdoor (best-effort, cached, throttled)
    # ------------------------------------------------------------------

    def fetch_glassdoor(self, company_name: str, ticker: str) -> dict:
        """Try to fetch Glassdoor employer rating for a company.

        Returns a dict with glassdoor_rating, glassdoor_ceo_pct,
        glassdoor_rec_pct — all None on any failure.
        """
        empty = {
            'glassdoor_rating': None,
            'glassdoor_ceo_pct': None,
            'glassdoor_rec_pct': None,
        }

        if not self._glassdoor_enabled or not company_name:
            return empty

        # Use ticker as cache key (more stable than name)
        if ticker in _glassdoor_cache:
            return _glassdoor_cache[ticker]

        # Throttle requests
        elapsed = time.time() - self._last_gd_request
        if elapsed < self._request_delay:
            time.sleep(self._request_delay - elapsed)
        self._last_gd_request = time.time()

        try:
            # Glassdoor employer autocomplete endpoint (returns JSON)
            term = urllib.parse.quote(company_name.split('(')[0].strip())
            url = (
                f'https://www.glassdoor.com/api/v1/employer/find.htm'
                f'?autocomplete=true&term={term}&pageSize=1'
            )
            req = urllib.request.Request(url, headers=_GLASSDOOR_HEADERS)
            with urllib.request.urlopen(req, timeout=self._glassdoor_timeout) as resp:
                raw = resp.read().decode('utf-8', errors='replace')

            data = json.loads(raw)
            # Response is a list of employer objects
            if not isinstance(data, list) or not data:
                _glassdoor_cache[ticker] = empty
                return empty

            emp = data[0]
            rating = emp.get('overallRating') or emp.get('rating')
            ceo_pct = emp.get('ceoApproval') or emp.get('ceoRating')
            rec_pct = emp.get('recommendToFriend') or emp.get('recommendPercent')

            result = {
                'glassdoor_rating': float(rating) if rating is not None else None,
                'glassdoor_ceo_pct': int(round(float(ceo_pct) * 100))
                                     if ceo_pct is not None and ceo_pct <= 1
                                     else (int(ceo_pct) if ceo_pct is not None else None),
                'glassdoor_rec_pct': int(round(float(rec_pct) * 100))
                                     if rec_pct is not None and rec_pct <= 1
                                     else (int(rec_pct) if rec_pct is not None else None),
            }
            _glassdoor_cache[ticker] = result
            return result

        except Exception:
            _glassdoor_cache[ticker] = empty
            return empty
