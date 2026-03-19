# data/validation.py
"""
Data quality checks for yfinance financial data.
Flags missing fields, stale data, and extreme values.
"""
from datetime import datetime


def validate_financials(financials, ticker):
    """
    Run quality checks on yfinance data.

    Returns dict with:
      - quality_score: 0-100 (higher = more complete/reliable)
      - warnings: list of warning strings
      - missing_critical: list of critical missing fields
    """
    warnings = []
    missing_critical = []
    score = 100

    info = (financials.get('info') or {}) if financials else {}
    bs = financials.get('balance_sheet') if financials else None
    inc = financials.get('income_statement') if financials else None
    cf = financials.get('cash_flow') if financials else None

    # --- Critical fields in info ---
    for field in ('marketCap', 'currentPrice', 'sharesOutstanding'):
        if not info.get(field):
            missing_critical.append(f'info.{field}')
            score -= 15

    # --- Financial statements exist and have data ---
    if bs is None or (hasattr(bs, 'empty') and bs.empty):
        missing_critical.append('balance_sheet')
        score -= 20
    elif hasattr(bs, 'shape') and bs.shape[1] < 2:
        warnings.append(f'{ticker}: Only {bs.shape[1]} year(s) of balance sheet data')
        score -= 5

    if inc is None or (hasattr(inc, 'empty') and inc.empty):
        missing_critical.append('income_statement')
        score -= 20
    elif hasattr(inc, 'shape') and inc.shape[1] < 2:
        warnings.append(f'{ticker}: Only {inc.shape[1]} year(s) of income data')
        score -= 5

    if cf is None or (hasattr(cf, 'empty') and cf.empty):
        missing_critical.append('cash_flow')
        score -= 20
    elif hasattr(cf, 'index'):
        has_fcf = any('Free Cash Flow' in str(idx) for idx in cf.index)
        if not has_fcf:
            warnings.append(f'{ticker}: No Free Cash Flow row in cash flow statement')
            score -= 10

    # --- Value sanity checks ---
    mcap = info.get('marketCap')
    if mcap and mcap < 1e6:
        warnings.append(f'{ticker}: Market cap suspiciously low (${mcap:,.0f})')
        score -= 10

    price = info.get('currentPrice') or info.get('regularMarketPrice')
    if price is not None and price <= 0:
        warnings.append(f'{ticker}: Price is non-positive (${price})')
        score -= 15

    pe = info.get('trailingPE')
    if pe is not None and (pe < 0 or pe > 500):
        warnings.append(f'{ticker}: P/E ratio extreme ({pe:.1f})')
        score -= 5

    # --- Data staleness check ---
    if cf is not None and hasattr(cf, 'columns') and len(cf.columns) > 0:
        latest_col = cf.columns[0]
        if hasattr(latest_col, 'year'):
            try:
                age_days = (datetime.now() - latest_col.to_pydatetime()).days
                if age_days > 450:  # > 15 months old
                    warnings.append(
                        f'{ticker}: Financial data may be stale ({age_days} days old)')
                    score -= 10
            except Exception:
                pass

    return {
        'quality_score': max(0, score),
        'warnings': warnings,
        'missing_critical': missing_critical,
    }
