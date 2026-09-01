# models/ratios.py
"""Core financial ratios: WACC, ROIC, DuPont decomposition, fundamental growth."""
import warnings as _py_warnings

from models.field_keys import (
    _get, EQUITY_KEYS, DEBT_KEYS, CASH_KEYS, CURRENT_ASSETS_KEYS,
    CURRENT_LIABILITIES_KEYS, NET_INCOME_KEYS, TOTAL_ASSETS_KEYS,
    OPERATING_INCOME_KEYS, INTEREST_KEYS, DA_KEYS, REVENUE_KEYS,
)


def compute_ratios(financials):
    """Quick ratios: ROE, Debt-to-Equity, Current Ratio, ROA.

    Uses `is not None` for net_income (not truthiness) so negative or zero
    earnings produce a real ROE — a negative ROE is signal, not absence.
    Equity / current_liabilities still need the zero guard for division.
    Returns a dict with the ratios plus a 'warnings' list.
    """
    ratios = {'warnings': []}
    bs = financials.get('balance_sheet')
    inc = financials.get('income_statement')
    if bs is None or inc is None or bs.empty or inc.empty:
        ratios['warnings'].append('balance_sheet or income_statement missing/empty')
        return ratios

    latest_bs = bs.iloc[:, 0]
    latest_inc = inc.iloc[:, 0]

    total_equity = _get(latest_bs, EQUITY_KEYS, allow_zero=False)
    total_assets = _get(latest_bs, TOTAL_ASSETS_KEYS, allow_zero=False)
    total_debt = _get(latest_bs, DEBT_KEYS)
    current_assets = _get(latest_bs, CURRENT_ASSETS_KEYS)
    current_liabilities = _get(latest_bs, CURRENT_LIABILITIES_KEYS, allow_zero=False)
    net_income = _get(latest_inc, NET_INCOME_KEYS)

    if total_equity and net_income is not None:
        ratios['ROE'] = net_income / total_equity
        if ratios['ROE'] < 0:
            ratios['warnings'].append(f"Negative ROE ({ratios['ROE']:.1%})")
        if total_equity < 0:
            ratios['warnings'].append(
                'Negative equity — ROE sign is meaningless (both sides flip)'
            )
    # True Total Debt / Equity (interest-bearing debt, not total liabilities —
    # the total-leverage view lives in dupont_leverage).
    if total_equity and total_debt is not None:
        ratios['Debt-to-Equity'] = total_debt / total_equity
    if current_assets is not None and current_liabilities:
        ratios['Current Ratio'] = current_assets / current_liabilities
    if total_assets and net_income is not None:
        ratios['ROA'] = net_income / total_assets
    return ratios


# Cost-of-debt sanity band, relative to the risk-free rate. Below the floor
# the implied rate reflects legacy low-coupon debt and understates the
# MARGINAL cost of borrowing; above the cap the denominator (period-end
# debt) is almost certainly wrong (debt repaid mid-year, lease interest in
# the numerator, etc.).
KD_SPREAD_CAP = 0.10
TAX_RATE_DEFAULT = 0.21
TAX_RATE_YEARS = 3


def effective_tax_rate(inc, *, max_years=TAX_RATE_YEARS, default=TAX_RATE_DEFAULT):
    """Effective tax rate for the debt tax shield.

    Averages Tax Provision / Pretax Income over the latest ``max_years``
    columns that have POSITIVE pretax income (each clipped to [0, 0.50]).
    Loss years are skipped: dividing a tax benefit by a pretax loss yields a
    positive "rate" that would grant a shield to a firm paying no tax. A
    genuine zero provision on positive pretax income counts as 0%, not as
    missing.

    Returns 0.0 when no positive-pretax year exists in the window but the
    latest year is a known loss (no shield to claim), else ``default``.
    """
    if inc is None or inc.empty:
        return default
    rates = []
    for col_idx in range(min(max_years, inc.shape[1])):
        col = inc.iloc[:, col_idx]
        pretax = _get(col, ['Pretax Income'])
        tax = _get(col, ['Tax Provision'])
        if pretax is None or tax is None or pretax <= 0:
            continue
        rates.append(max(0.0, min(tax / pretax, 0.50)))
    if rates:
        return sum(rates) / len(rates)
    latest_pretax = _get(inc.iloc[:, 0], ['Pretax Income'])
    if latest_pretax is not None and latest_pretax <= 0:
        return 0.0
    return default


def calculate_wacc(financials, cost_of_equity, *,
                   risk_free_rate=None, credit_spread=0.025):
    """
    WACC = (E/V)*Re + (D/V)*Rd*(1-T).
    Uses market cap for equity weight (not book), per worksheet Step 4C.

    Cost of debt = interest expense / AVERAGE debt (latest and prior
    period-end, when a prior column exists) so a mid-year repayment or
    issuance doesn't distort the rate. When ``risk_free_rate`` is given the
    result is clamped to [Rf, Rf + KD_SPREAD_CAP] with a RuntimeWarning on
    either clip: legacy low-coupon debt understates the marginal cost, and
    a rate far above Rf means the denominator is wrong.

    Hardened: when debt exists but interest_expense is missing or zero,
    fall back to cost_of_debt = risk_free_rate + credit_spread instead of
    silently treating debt as free. Pass risk_free_rate from the macro
    context; the default credit_spread is a conservative 2.5%.

    If both interest_expense AND risk_free_rate are missing, falls back
    to cost_of_equity × 0.6 as a last-resort placeholder. WACC may
    misstate cost of capital meaningfully in that case — check upstream
    data quality.

    Tax rate: see ``effective_tax_rate`` (multi-year, positive-pretax only).

    Returns the WACC as a float, or None when equity cannot be sourced.
    """
    bs = financials.get('balance_sheet')
    inc = financials.get('income_statement')
    info = financials.get('info') or {}

    if bs is None or inc is None or bs.empty or inc.empty:
        return None

    latest_bs = bs.iloc[:, 0]
    latest_inc = inc.iloc[:, 0]

    market_cap = info.get('marketCap')
    total_equity_book = _get(latest_bs, EQUITY_KEYS, allow_zero=False)
    total_equity = market_cap if (market_cap and market_cap > 0) else total_equity_book

    total_debt = _get(latest_bs, DEBT_KEYS) or 0
    interest_expense = _get(latest_inc, INTEREST_KEYS)

    if not total_equity or total_equity <= 0:
        return None

    total_capital = total_equity + total_debt
    weight_equity = total_equity / total_capital
    weight_debt = total_debt / total_capital

    # Cost of debt — credit-spread fallback when interest expense missing.
    # The old behavior of cost_of_debt = 0 silently understated WACC for any
    # levered firm with a sparse cash-flow statement.
    if total_debt > 0:
        if interest_expense and interest_expense > 0:
            prior_debt = _get(bs.iloc[:, 1], DEBT_KEYS) if bs.shape[1] > 1 else None
            debt_base = ((total_debt + prior_debt) / 2
                         if prior_debt and prior_debt > 0 else total_debt)
            cost_of_debt = interest_expense / debt_base
            if risk_free_rate is not None and risk_free_rate > 0:
                kd_floor, kd_cap = risk_free_rate, risk_free_rate + KD_SPREAD_CAP
                if cost_of_debt < kd_floor:
                    _py_warnings.warn(
                        f'calculate_wacc: implied cost of debt {cost_of_debt:.2%} is '
                        f'below the risk-free rate {kd_floor:.2%} (legacy coupons); '
                        f'floored at Rf for the marginal cost',
                        RuntimeWarning, stacklevel=2)
                    cost_of_debt = kd_floor
                elif cost_of_debt > kd_cap:
                    _py_warnings.warn(
                        f'calculate_wacc: implied cost of debt {cost_of_debt:.2%} '
                        f'exceeds Rf + {KD_SPREAD_CAP:.0%} — interest/debt mismatch '
                        f'(mid-year repayment or lease interest); capped at {kd_cap:.2%}',
                        RuntimeWarning, stacklevel=2)
                    cost_of_debt = kd_cap
        elif risk_free_rate is not None and risk_free_rate > 0:
            cost_of_debt = risk_free_rate + credit_spread
        else:
            _py_warnings.warn(
                'calculate_wacc: no interest expense and no risk-free rate — '
                'cost of debt fabricated as cost_of_equity * 0.6; WACC may '
                'misstate cost of capital',
                RuntimeWarning,
                stacklevel=2,
            )
            cost_of_debt = cost_of_equity * 0.6
    else:
        cost_of_debt = 0

    tax_rate = effective_tax_rate(inc)
    return weight_equity * cost_of_equity + weight_debt * cost_of_debt * (1 - tax_rate)


def calculate_roic(financials):
    """Per-year and average ROIC = NOPAT / (Equity + Debt − Cash).

    Returns dict with 'roic_by_year', 'avg_roic', 'warnings' (years skipped
    because invested_capital was non-positive, for instance), plus the
    per-year intermediates 'nopat_by_year' and 'invested_capital_by_year'
    (same year keys and skip conditions as roic_by_year) so consumers can
    compute incremental ROIC (ΔNOPAT/ΔIC) without re-deriving them.
    """
    bs = financials.get('balance_sheet')
    inc = financials.get('income_statement')
    if bs is None or inc is None or bs.empty or inc.empty:
        return None

    common_years = bs.columns.intersection(inc.columns)
    if len(common_years) == 0:
        return None

    roic_by_year = {}
    nopat_by_year = {}
    invested_capital_by_year = {}
    warnings = []
    for year in common_years:
        bs_year = bs[year]
        inc_year = inc[year]

        operating_income = _get(inc_year, OPERATING_INCOME_KEYS)
        tax_provision = _get(inc_year, ['Tax Provision'])
        pretax_income = _get(inc_year, ['Pretax Income'], allow_zero=False)
        total_equity = _get(bs_year, EQUITY_KEYS, allow_zero=False)
        total_debt = _get(bs_year, DEBT_KEYS)
        cash = _get(bs_year, CASH_KEYS)

        if operating_income is None or pretax_income is None or total_equity is None:
            continue

        tax_rate = (tax_provision / pretax_income) if tax_provision and pretax_income else 0.21
        tax_rate = max(0.0, min(tax_rate, 0.50))
        nopat = operating_income * (1 - tax_rate)
        invested_capital = total_equity + (total_debt or 0) - (cash or 0)
        if invested_capital <= 0:
            warnings.append(
                f"{year}: invested_capital <= 0 (cash > equity + debt), skipped"
            )
            continue

        year_key = str(year.year) if hasattr(year, 'year') else str(year)
        roic_by_year[year_key] = nopat / invested_capital
        nopat_by_year[year_key] = nopat
        invested_capital_by_year[year_key] = invested_capital

    if not roic_by_year:
        return None

    avg_roic = sum(roic_by_year.values()) / len(roic_by_year)
    return {'roic_by_year': roic_by_year, 'avg_roic': avg_roic,
            'warnings': warnings,
            'nopat_by_year': nopat_by_year,
            'invested_capital_by_year': invested_capital_by_year}


def dupont_decomposition(net_income, revenue, total_assets, equity):
    """3-factor DuPont decomposition: ROE = Margin x Turnover x Leverage."""
    if (net_income is None or revenue is None or
            total_assets is None or equity is None):
        return None
    if revenue <= 0 or total_assets <= 0 or equity <= 0:
        return None

    margin = net_income / revenue
    turnover = revenue / total_assets
    leverage = total_assets / equity
    return {
        'margin': margin,
        'turnover': turnover,
        'leverage': leverage,
        'roe': margin * turnover * leverage,
    }


def compute_dupont(financials):
    """Compute DuPont decomposition from financials dict."""
    bs = financials.get('balance_sheet')
    inc = financials.get('income_statement')
    if bs is None or inc is None or bs.empty or inc.empty:
        return None
    latest_bs = bs.iloc[:, 0]
    latest_inc = inc.iloc[:, 0]
    ni = _get(latest_inc, NET_INCOME_KEYS)
    rev = _get(latest_inc, REVENUE_KEYS)
    ta = _get(latest_bs, TOTAL_ASSETS_KEYS)
    eq = _get(latest_bs, EQUITY_KEYS, allow_zero=False)
    return dupont_decomposition(ni, rev, ta, eq)


# ---------------------------------------------------------------------------
# Piotroski F-Score (Worksheet Step 1 quality filter)
# ---------------------------------------------------------------------------


def calculate_fundamental_growth(financials, roic_override=None):
    """Fundamental growth rate = Reinvestment Rate × ROIC.

    Reinvestment Rate = (Capex - D&A + ΔWorkingCapital) / NOPAT
    NOPAT = Operating Income × (1 - tax_rate)

    Returns dict with 'fundamental_growth', 'reinvestment_rate', 'roic_used'
    or empty dict on insufficient data.
    """
    inc = financials.get('income_statement')
    cf = financials.get('cash_flow')
    bs = financials.get('balance_sheet')
    if inc is None or inc.empty or cf is None or cf.empty:
        return {}

    latest_inc = inc.iloc[:, 0]
    latest_cf = cf.iloc[:, 0]

    operating_income = _get(latest_inc, OPERATING_INCOME_KEYS)
    if not operating_income or operating_income <= 0:
        return {}

    tax_provision = _get(latest_inc, ['Tax Provision'])
    pretax_income = _get(latest_inc, ['Pretax Income'], allow_zero=False)
    tax_rate = (tax_provision / pretax_income) if (tax_provision and pretax_income) else 0.21
    tax_rate = max(0, min(tax_rate, 0.50))
    nopat = operating_income * (1 - tax_rate)
    if nopat <= 0:
        return {}

    capex = _get(latest_cf, ['Capital Expenditure', 'Capital Expenditures'])
    da = _get(latest_cf, DA_KEYS)
    if capex is None or da is None:
        return {}
    capex = abs(capex)
    da = abs(da)

    # Delta working capital (need 2 years of balance sheet)
    delta_wc = 0
    if bs is not None and not bs.empty and bs.shape[1] >= 2:
        curr_bs = bs.iloc[:, 0]
        prev_bs = bs.iloc[:, 1]
        ca_curr = _get(curr_bs, CURRENT_ASSETS_KEYS) or 0
        cl_curr = _get(curr_bs, CURRENT_LIABILITIES_KEYS) or 0
        ca_prev = _get(prev_bs, CURRENT_ASSETS_KEYS) or 0
        cl_prev = _get(prev_bs, CURRENT_LIABILITIES_KEYS) or 0
        delta_wc = (ca_curr - cl_curr) - (ca_prev - cl_prev)

    net_reinvestment = capex - da + delta_wc
    reinvestment_rate = net_reinvestment / nopat
    reinvestment_rate = max(0, min(reinvestment_rate, 1.0))  # clamp [0, 1]

    # ROIC
    roic_val = roic_override
    if roic_val is None:
        roic_result = calculate_roic(financials)
        roic_val = roic_result['avg_roic'] if roic_result else None
    if roic_val is None or roic_val <= 0:
        return {}

    fundamental_growth = reinvestment_rate * roic_val
    fundamental_growth = max(0, min(fundamental_growth, 0.30))  # clamp [0, 30%]

    return {
        'fundamental_growth': fundamental_growth,
        'reinvestment_rate': reinvestment_rate,
        'roic_used': roic_val,
    }
