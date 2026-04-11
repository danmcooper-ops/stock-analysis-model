# scripts/report_portfolio_html.py
"""Portfolio HTML report builder — renders the portfolio Jinja2 report."""
import os
import json
from datetime import date

import jinja2
import numpy as np


def _json_default(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if not np.isnan(obj) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def fmt_pct(val, decimals=1):
    if val is None:
        return 'N/A'
    sign = '+' if val > 0 else ''
    return f"{sign}{val:.{decimals}%}"


def fmt_dollar(val):
    if val is None:
        return 'N/A'
    sign = '+' if val > 0 else ''
    return f"{sign}${abs(val):,.2f}"


def fmt_dollar_short(val):
    if val is None:
        return 'N/A'
    if abs(val) >= 1e9:
        return f"${val/1e9:.1f}B"
    if abs(val) >= 1e6:
        return f"${val/1e6:.1f}M"
    if abs(val) >= 1e3:
        return f"${val/1e3:.1f}K"
    return f"${val:,.2f}"


def build_portfolio_html(portfolio_state, filename):
    """Render the interactive portfolio HTML report via Jinja2 template.

    Parameters
    ----------
    portfolio_state : dict
        The full portfolio state dict from run_portfolio_tracker().
    filename : str
        Output file path.
    """
    holdings = portfolio_state.get('holdings', [])
    alerts = portfolio_state.get('alerts', [])
    concentration = portfolio_state.get('concentration', {})

    # Build chart-ready holdings list (remove non-serializable types)
    holdings_json = json.dumps(holdings, default=_json_default)
    alerts_json = json.dumps(alerts, default=_json_default)

    # Sector weights for donut chart
    sector_weights = concentration.get('sector_weights', {})
    sector_chart_data = json.dumps([
        {'sector': s, 'weight': w}
        for s, w in sorted(sector_weights.items(), key=lambda x: -x[1])
    ], default=_json_default)

    # Summary metrics
    total_mv = portfolio_state.get('total_market_value')
    total_cost = portfolio_state.get('total_cost_basis')
    upnl = portfolio_state.get('unrealized_pnl')
    upnl_pct = portfolio_state.get('unrealized_pnl_pct')
    realized_ytd = portfolio_state.get('realized_pnl_ytd', 0)
    port_ytd = portfolio_state.get('portfolio_return_ytd')
    bench_ytd = portfolio_state.get('benchmark_return_ytd')
    alpha_ytd = portfolio_state.get('portfolio_alpha_ytd')
    bench = portfolio_state.get('benchmark', 'SPY')
    high_alerts = sum(1 for a in alerts if a.get('severity') == 'HIGH')
    med_alerts = sum(1 for a in alerts if a.get('severity') == 'MEDIUM')

    template_dir = os.path.join(os.path.dirname(__file__), '..', 'templates')
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_dir),
        autoescape=False,
    )
    env.filters['fmt_pct'] = fmt_pct
    env.filters['fmt_dollar'] = fmt_dollar
    env.filters['fmt_dollar_short'] = fmt_dollar_short

    template = env.get_template('portfolio_report.html')
    html = template.render(
        portfolio_name=portfolio_state.get('portfolio_name', 'My Portfolio'),
        generated_at=date.today().strftime('%Y-%m-%d'),
        benchmark=bench,
        macro_regime=portfolio_state.get('macro_regime', 'N/A'),

        # Summary metrics
        total_mv=total_mv,
        total_cost=total_cost,
        upnl=upnl,
        upnl_pct=upnl_pct,
        realized_ytd=realized_ytd,
        port_ytd=port_ytd,
        bench_ytd=bench_ytd,
        alpha_ytd=alpha_ytd,
        high_alerts=high_alerts,
        med_alerts=med_alerts,
        n_holdings=len(holdings),

        # JSON blobs for JS
        holdings_json=holdings_json,
        alerts_json=alerts_json,
        sector_chart_data=sector_chart_data,

        # Concentration
        concentration=concentration,

        fmt_pct=fmt_pct,
        fmt_dollar=fmt_dollar,
        fmt_dollar_short=fmt_dollar_short,
    )

    with open(filename, 'w') as f:
        f.write(html)
