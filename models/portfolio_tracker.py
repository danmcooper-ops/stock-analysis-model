# models/portfolio_tracker.py
"""Portfolio tracking: enrichment, P&L, returns, and alert detection.

All functions are pure (no I/O).  The data layer (PortfolioClient) handles
all file and network access; these functions operate on plain dicts/lists.
"""
from datetime import date, datetime

import pandas as pd


# ---------------------------------------------------------------------------
# Rating helpers
# ---------------------------------------------------------------------------

_RATING_RANK = {'BUY': 3, 'LEAN BUY': 2, 'HOLD': 1, 'PASS': 0}


def _rating_rank(rating):
    return _RATING_RANK.get(rating, -1)


# ---------------------------------------------------------------------------
# Holding enrichment
# ---------------------------------------------------------------------------

def _aggregate_lots(holdings):
    """Aggregate multiple lots of the same ticker into one record.

    Shares are summed; cost_basis becomes the weighted average.
    The earliest purchase_date is used.
    """
    by_ticker = {}
    for h in holdings:
        t = h['ticker']
        if t not in by_ticker:
            by_ticker[t] = {
                'ticker': t,
                'shares': 0.0,
                'cost_value': 0.0,
                'purchase_date': h['purchase_date'],
                'notes': h.get('notes', ''),
            }
        lot = by_ticker[t]
        shares = float(h['shares'])
        lot['shares'] += shares
        lot['cost_value'] += shares * float(h['cost_basis'])
        # Keep earliest purchase date
        if h['purchase_date'] < lot['purchase_date']:
            lot['purchase_date'] = h['purchase_date']

    aggregated = []
    for t, lot in by_ticker.items():
        shares = lot['shares']
        cost_value = lot['cost_value']
        aggregated.append({
            'ticker': t,
            'shares': shares,
            'cost_basis': cost_value / shares if shares > 0 else 0.0,
            'cost_value': cost_value,
            'purchase_date': lot['purchase_date'],
            'notes': lot['notes'],
        })
    return aggregated


def enrich_holdings(holdings, current_prices, analysis_results_by_ticker):
    """Join live prices and model outputs onto each holding.

    Parameters
    ----------
    holdings : list of dict
        Raw holdings from PortfolioClient.load_holdings()['holdings'].
    current_prices : dict
        {ticker: float or None} from PortfolioClient.fetch_current_prices().
    analysis_results_by_ticker : dict
        {ticker: result_dict} built from today's results JSON.

    Returns
    -------
    list of dict
        One enriched dict per ticker (lots aggregated).
    """
    aggregated = _aggregate_lots(holdings)
    enriched = []

    for lot in aggregated:
        ticker = lot['ticker']
        price = current_prices.get(ticker)
        result = analysis_results_by_ticker.get(ticker, {})

        shares = lot['shares']
        cost_basis = lot['cost_basis']
        cost_value = lot['cost_value']
        market_value = shares * price if price is not None else None

        unrealized_pnl = (market_value - cost_value) if market_value is not None else None
        unrealized_pnl_pct = (unrealized_pnl / cost_value) if (cost_value > 0 and unrealized_pnl is not None) else None

        in_universe = bool(result)

        record = {
            # Position basics
            'ticker': ticker,
            'shares': shares,
            'cost_basis': cost_basis,
            'cost_value': cost_value,
            'purchase_date': lot['purchase_date'],
            'notes': lot.get('notes', ''),

            # Live pricing
            'current_price': price,
            'market_value': market_value,
            'unrealized_pnl': unrealized_pnl,
            'unrealized_pnl_pct': unrealized_pnl_pct,

            # Weight (filled in by compute_holding_weights)
            'position_weight': None,

            # Alpha (filled in by compute_portfolio_returns)
            'return_since_purchase': unrealized_pnl_pct,
            'benchmark_return_same_period': None,
            'alpha': None,

            # From analysis model
            'in_universe': in_universe,
            'rating': result.get('rating', 'NOT IN UNIVERSE' if not in_universe else None),
            'dcf_fv': result.get('dcf_fv'),
            'mos': result.get('mos'),
            'mc_confidence': result.get('mc_confidence'),
            '_composite_score': result.get('_composite_score'),
            '_score_valuation': result.get('_score_valuation'),
            '_score_quality': result.get('_score_quality'),
            '_score_moat': result.get('_score_moat'),
            '_score_growth': result.get('_score_growth'),
            '_score_ownership': result.get('_score_ownership'),
            'spread': result.get('spread'),
            'sector': result.get('sector', 'Unknown'),
            'company_name': result.get('company_name', ticker),
            'piotroski': result.get('piotroski'),
            'roic': result.get('roic'),
            'wacc': result.get('wacc'),
            'epv_fv': result.get('epv_fv'),
            'rim_fv': result.get('rim_fv'),
            'ddm_fv': result.get('ddm_fv'),
            'macro_regime': result.get('macro_regime'),

            # Valuation gap flag
            'valuation_gap_pct': None,
            'valuation_gap_alert': False,
        }

        # Compute valuation gap
        dcf_fv = record['dcf_fv']
        if dcf_fv and price and dcf_fv > 0:
            gap = (price - dcf_fv) / dcf_fv
            record['valuation_gap_pct'] = gap

        enriched.append(record)

    return enriched


# ---------------------------------------------------------------------------
# Portfolio weights
# ---------------------------------------------------------------------------

def compute_holding_weights(enriched_holdings):
    """Compute each holding's % of total portfolio market value in-place.

    Sets 'position_weight' on each holding dict.  Holdings without a price
    are excluded from the weight denominator.

    Returns
    -------
    list of dict
        Same list with 'position_weight' populated.
    """
    total_value = sum(
        h['market_value'] for h in enriched_holdings
        if h.get('market_value') is not None
    )
    for h in enriched_holdings:
        mv = h.get('market_value')
        if mv is not None and total_value > 0:
            h['position_weight'] = mv / total_value
        else:
            h['position_weight'] = None
    return enriched_holdings


# ---------------------------------------------------------------------------
# P&L summary
# ---------------------------------------------------------------------------

def compute_portfolio_pnl(enriched_holdings, realized_gains=None):
    """Compute aggregate portfolio P&L.

    Parameters
    ----------
    enriched_holdings : list of dict
    realized_gains : list of dict, optional
        From holdings.json['realized_gains'].

    Returns
    -------
    dict
        {total_cost_basis, total_market_value, unrealized_pnl,
         unrealized_pnl_pct, realized_pnl_ytd}
    """
    total_cost = sum(
        h['cost_value'] for h in enriched_holdings
        if h.get('cost_value') is not None
    )
    total_mv = sum(
        h['market_value'] for h in enriched_holdings
        if h.get('market_value') is not None
    )
    unrealized_pnl = total_mv - total_cost if (total_mv and total_cost) else None
    unrealized_pnl_pct = unrealized_pnl / total_cost if (total_cost > 0 and unrealized_pnl is not None) else None

    # Realized gains YTD
    realized_pnl_ytd = 0.0
    if realized_gains:
        this_year = str(date.today().year)
        for rg in realized_gains:
            if str(rg.get('sale_date', '')).startswith(this_year):
                shares_sold = float(rg.get('shares_sold', 0))
                sale_price = float(rg.get('sale_price', 0))
                cost = float(rg.get('cost_basis', 0))
                realized_pnl_ytd += shares_sold * (sale_price - cost)

    return {
        'total_cost_basis': total_cost,
        'total_market_value': total_mv,
        'unrealized_pnl': unrealized_pnl,
        'unrealized_pnl_pct': unrealized_pnl_pct,
        'realized_pnl_ytd': realized_pnl_ytd,
    }


# ---------------------------------------------------------------------------
# Return vs. benchmark
# ---------------------------------------------------------------------------

def _compute_return_since(series, since_date_str):
    """Compute total return of a price series since a given date.

    Parameters
    ----------
    series : pandas.Series
        DatetimeIndex → price.
    since_date_str : str
        ISO date string.

    Returns
    -------
    float or None
    """
    if series is None or len(series) == 0:
        return None
    try:
        since_ts = pd.Timestamp(since_date_str)
        # Remove tz info for comparison if needed
        idx = series.index
        if idx.tz is not None:
            since_ts = since_ts.tz_localize(idx.tz)
        slice_ = series[idx >= since_ts]
        if len(slice_) < 2:
            return None
        start_price = float(slice_.iloc[0])
        end_price = float(slice_.iloc[-1])
        if start_price <= 0:
            return None
        return (end_price - start_price) / start_price
    except Exception:
        return None


def _ytd_start():
    """Return ISO string for Jan 1 of the current year."""
    return f"{date.today().year}-01-01"


def compute_portfolio_returns(enriched_holdings, benchmark_series, ticker_histories=None):
    """Compute per-holding alpha and portfolio-level return vs benchmark.

    Parameters
    ----------
    enriched_holdings : list of dict
        Already weight-enriched holdings.
    benchmark_series : pandas.Series
        DatetimeIndex → benchmark close price.
    ticker_histories : dict, optional
        {ticker: pandas.Series} of price histories.  If None, alpha is not
        computed per-holding (requires a separate fetch).

    Returns
    -------
    dict
        {portfolio_return_ytd, benchmark_return_ytd,
         portfolio_return_1y, benchmark_return_1y,
         portfolio_alpha_ytd}
    """
    ticker_histories = ticker_histories or {}

    # Per-holding alpha
    for h in enriched_holdings:
        ticker = h['ticker']
        purchase_date = h.get('purchase_date', _ytd_start())

        # Benchmark return over holding period
        bench_ret = _compute_return_since(benchmark_series, purchase_date)
        h['benchmark_return_same_period'] = bench_ret

        # Stock return over holding period (use ticker history if available)
        hist = ticker_histories.get(ticker)
        if hist is not None and len(hist) > 0:
            stock_ret = _compute_return_since(hist, purchase_date)
        else:
            stock_ret = h.get('return_since_purchase')  # fallback: cost-basis calc
        h['return_since_purchase'] = stock_ret

        # Alpha
        if stock_ret is not None and bench_ret is not None:
            h['alpha'] = stock_ret - bench_ret
        else:
            h['alpha'] = None

    # Portfolio-level YTD: weighted sum of YTD returns
    ytd_start = _ytd_start()
    bench_ytd = _compute_return_since(benchmark_series, ytd_start)

    weighted_return_ytd = 0.0
    weight_sum = 0.0
    for h in enriched_holdings:
        w = h.get('position_weight') or 0.0
        purchase_date = h.get('purchase_date', ytd_start)
        # For holdings opened before Jan 1, use YTD return; otherwise use full return
        effective_start = max(purchase_date, ytd_start)
        hist = ticker_histories.get(h['ticker'])
        if hist is not None and len(hist) > 0:
            ret = _compute_return_since(hist, effective_start)
        else:
            ret = h.get('return_since_purchase') if purchase_date >= ytd_start else None
        if ret is not None and w > 0:
            weighted_return_ytd += w * ret
            weight_sum += w

    portfolio_return_ytd = weighted_return_ytd / weight_sum if weight_sum > 0 else None

    # 1-year return
    one_year_ago = f"{date.today().year - 1}-{date.today().month:02d}-{date.today().day:02d}"
    bench_1y = _compute_return_since(benchmark_series, one_year_ago)

    return {
        'portfolio_return_ytd': portfolio_return_ytd,
        'benchmark_return_ytd': bench_ytd,
        'portfolio_return_1y': None,  # requires full year of history per holding
        'benchmark_return_1y': bench_1y,
        'portfolio_alpha_ytd': (
            portfolio_return_ytd - bench_ytd
            if portfolio_return_ytd is not None and bench_ytd is not None
            else None
        ),
    }


# ---------------------------------------------------------------------------
# Alert detection
# ---------------------------------------------------------------------------

def detect_alerts(enriched_holdings, prev_results_by_ticker,
                  valuation_gap_threshold=0.20, score_drop_threshold=10.0):
    """Detect rating changes, valuation divergence, and score drops.

    Parameters
    ----------
    enriched_holdings : list of dict
    prev_results_by_ticker : dict
        {ticker: result_dict} from the prior day's results JSON.
        Pass {} on the first run.
    valuation_gap_threshold : float
        Fractional gap |price - dcf_fv| / dcf_fv that triggers a MEDIUM alert.
    score_drop_threshold : float
        Point drop in composite score that triggers a MEDIUM alert.

    Returns
    -------
    list of dict
        Sorted: HIGH severity first.
    """
    today = date.today().isoformat()
    alerts = []

    for h in enriched_holdings:
        ticker = h['ticker']
        prev = prev_results_by_ticker.get(ticker, {})

        # 1. Not in universe alert
        if not h.get('in_universe'):
            alerts.append({
                'ticker': ticker,
                'alert_type': 'not_in_universe',
                'severity': 'MEDIUM',
                'message': (
                    f"{ticker} is not in today's analysis universe "
                    f"(may no longer pass ROIC screen or data unavailable)"
                ),
                'prior_value': None,
                'current_value': None,
                'date': today,
            })
            continue  # no further alerts if not in universe

        # 2. Rating change alert
        curr_rating = h.get('rating')
        prev_rating = prev.get('rating')
        if curr_rating and prev_rating and curr_rating != prev_rating:
            downgrade = _rating_rank(prev_rating) > _rating_rank(curr_rating)
            alerts.append({
                'ticker': ticker,
                'alert_type': 'rating_downgrade' if downgrade else 'rating_upgrade',
                'severity': 'HIGH' if downgrade else 'LOW',
                'message': (
                    f"{ticker} rating {'downgraded' if downgrade else 'upgraded'} "
                    f"from {prev_rating} to {curr_rating}"
                ),
                'prior_value': prev_rating,
                'current_value': curr_rating,
                'date': today,
            })

        # 3. Valuation gap alert
        gap = h.get('valuation_gap_pct')
        dcf_fv = h.get('dcf_fv')
        price = h.get('current_price')
        if gap is not None and abs(gap) > valuation_gap_threshold:
            direction = 'above' if gap > 0 else 'below'
            h['valuation_gap_alert'] = True
            alerts.append({
                'ticker': ticker,
                'alert_type': 'valuation_gap',
                'severity': 'MEDIUM',
                'message': (
                    f"{ticker} price ${price:.2f} is {abs(gap):.0%} {direction} "
                    f"DCF fair value (${dcf_fv:.2f})"
                ),
                'prior_value': dcf_fv,
                'current_value': price,
                'date': today,
            })

        # 4. Composite score drop alert
        curr_score = h.get('_composite_score')
        prev_score = prev.get('_composite_score')
        if curr_score is not None and prev_score is not None:
            drop = prev_score - curr_score
            if drop > score_drop_threshold:
                alerts.append({
                    'ticker': ticker,
                    'alert_type': 'score_drop',
                    'severity': 'MEDIUM',
                    'message': (
                        f"{ticker} composite score dropped {drop:.1f} pts "
                        f"({prev_score:.1f} → {curr_score:.1f})"
                    ),
                    'prior_value': prev_score,
                    'current_value': curr_score,
                    'date': today,
                })

    # Sort: HIGH first, then MEDIUM, then LOW
    severity_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    alerts.sort(key=lambda a: severity_order.get(a['severity'], 3))
    return alerts


# ---------------------------------------------------------------------------
# Realized gain summary
# ---------------------------------------------------------------------------

def summarize_realized_gains(realized_gains):
    """Summarize realized gains by year.

    Parameters
    ----------
    realized_gains : list of dict

    Returns
    -------
    dict
        {year_str: total_gain_float}
    """
    by_year = {}
    for rg in realized_gains or []:
        sale_date = str(rg.get('sale_date', ''))
        year = sale_date[:4] if len(sale_date) >= 4 else 'Unknown'
        shares = float(rg.get('shares_sold', 0))
        gain = shares * (float(rg.get('sale_price', 0)) - float(rg.get('cost_basis', 0)))
        by_year[year] = by_year.get(year, 0.0) + gain
    return by_year
