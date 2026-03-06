# scripts/analyze_stock.py
import sys
import os
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from urllib.request import urlopen, Request
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.sec_edgar import SECEdgarClient
from data.yfinance_client import YFinanceClient
from models.capm import calculate_beta, expected_return, geometric_annualized_return
from models.dcf import calculate_dcf, project_fcf, calculate_terminal_value
from models.comparisons import compute_ratios, calculate_roic, compute_ratios_trend

RISK_FREE_RATE = 0.04
MARKET_TICKER = "^GSPC"
TERMINAL_GROWTH_RATE = 0.03
ROIC_THRESHOLD = 0.10


def _read_wiki_tables(url):
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as resp:
        html = resp.read().decode("utf-8")
    return pd.read_html(io.StringIO(html))

def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = _read_wiki_tables(url)
    df = tables[0]
    return df['Symbol'].tolist()

def get_nyse_tickers():
    try:
        df = pd.read_csv("nyse_tickers.csv")
        return df['Symbol'].tolist()
    except FileNotFoundError:
        print("nyse_tickers.csv not found, skipping NYSE tickers.")
        return []

def get_dow_tickers():
    url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
    tables = _read_wiki_tables(url)
    for table in tables:
        if 'Symbol' in table.columns:
            return table['Symbol'].tolist()
    return []


def run_capm(yf_client, ticker, market_history):
    stock_history = yf_client.fetch_history(ticker, period="5y")
    aligned = pd.DataFrame({'stock': stock_history, 'market': market_history}).dropna()
    if len(aligned) < 30:
        return None

    stock_returns = aligned['stock'].pct_change().dropna().values
    market_returns = aligned['market'].pct_change().dropna().values

    beta_result = calculate_beta(stock_returns, market_returns, adjust=True)
    market_annual_return = geometric_annualized_return(market_returns)

    if market_annual_return is None:
        return None

    er = expected_return(RISK_FREE_RATE, beta_result['adjusted_beta'], market_annual_return)

    return {
        'beta': beta_result['adjusted_beta'],
        'raw_beta': beta_result['raw_beta'],
        'r_squared': beta_result['r_squared'],
        'se_beta': beta_result['se_beta'],
        'expected_return': er,
        'n_observations': beta_result['n_observations'],
    }


def run_dcf(yf_data, discount_rate):
    cf = yf_data.get('cash_flow')
    if cf is None or cf.empty:
        return None
    fcf_row = None
    for label in ['Free Cash Flow']:
        if label in cf.index:
            fcf_row = cf.loc[label]
            break
    if fcf_row is None:
        return None

    fcf_values = fcf_row.dropna().sort_index().values.tolist()
    if not fcf_values or discount_rate is None or discount_rate <= 0:
        return None

    projection = project_fcf(fcf_values, projection_years=5)
    if projection is None:
        return None

    tv_result = calculate_terminal_value(
        projection['projected_fcfs'][-1],
        discount_rate,
        TERMINAL_GROWTH_RATE,
    )

    ev = calculate_dcf(
        projection['projected_fcfs'],
        discount_rate,
        tv_result['terminal_value'],
    )

    return {
        'enterprise_value': ev,
        'growth_rate': projection['growth_rate'],
        'terminal_value': tv_result['terminal_value'],
        'tv_clamped': tv_result['was_clamped'],
    }


def format_summary(ticker, capm_result, dcf_result, ratios, roic_data, trend_data=None):
    lines = [f"Ticker: {ticker}", ""]

    if roic_data:
        lines.append(f"5Y Avg ROIC: {roic_data['avg_roic']:.2%}")
        for year, roic in sorted(roic_data['roic_by_year'].items()):
            lines.append(f"  {year}: {roic:.2%}")
    lines.append("")

    if capm_result:
        lines.append(f"CAPM Raw Beta: {capm_result['raw_beta']:.4f}")
        lines.append(f"CAPM Adj Beta: {capm_result['beta']:.4f}")
        lines.append(f"R-squared: {capm_result['r_squared']:.4f}")
        if capm_result['se_beta'] is not None:
            lines.append(f"Beta Std Error: {capm_result['se_beta']:.4f}")
        lines.append(f"Expected Return: {capm_result['expected_return']:.2%}")
        lines.append(f"Observations: {capm_result['n_observations']}")
    else:
        lines.append("CAPM: insufficient data")
    lines.append("")

    if dcf_result:
        lines.append(f"DCF Enterprise Value: ${dcf_result['enterprise_value']:,.0f}")
        lines.append(f"FCF Growth Rate: {dcf_result['growth_rate']:.2%}")
        if dcf_result['tv_clamped']:
            lines.append("  (terminal growth rate was clamped)")
    else:
        lines.append("DCF: insufficient data")
    lines.append("")

    if ratios:
        for name, val in ratios.items():
            lines.append(f"{name}: {val:.4f}")
    else:
        lines.append("Ratios: insufficient data")

    if trend_data and trend_data.get('trends'):
        lines.append("")
        lines.append("Trends:")
        for name, direction in trend_data['trends'].items():
            lines.append(f"  {name}: {direction}")

    return "\n".join(lines)


if __name__ == "__main__":
    sp500 = set(get_sp500_tickers())
    nyse = set(get_nyse_tickers())
    dow = set(get_dow_tickers())
    all_tickers = sorted(sp500 | nyse | dow)

    sec_email = os.environ.get('SEC_EDGAR_EMAIL', 'your_email@example.com')
    if sec_email == 'your_email@example.com':
        print("WARNING: Set SEC_EDGAR_EMAIL env var for SEC EDGAR access.")
    sec_client = SECEdgarClient(sec_email)
    yf_client = YFinanceClient()

    # Phase 1: Screen tickers by 5-year average ROIC >= 10%
    print(f"Screening {len(all_tickers)} tickers for 5Y avg ROIC >= {ROIC_THRESHOLD:.0%}...")
    qualifying = []
    roic_cache = {}
    for i, ticker in enumerate(all_tickers, 1):
        try:
            yf_data = yf_client.fetch_financials(ticker)
            roic_data = calculate_roic(yf_data)
            if roic_data and roic_data['avg_roic'] >= ROIC_THRESHOLD:
                qualifying.append(ticker)
                roic_cache[ticker] = roic_data
                print(f"  [{i}/{len(all_tickers)}] {ticker} - ROIC {roic_data['avg_roic']:.2%} PASS")
            else:
                avg = roic_data['avg_roic'] if roic_data else None
                print(f"  [{i}/{len(all_tickers)}] {ticker} - ROIC {avg:.2% if avg is not None else 'N/A'} skip")
        except Exception as e:
            print(f"  [{i}/{len(all_tickers)}] {ticker} - error: {e}")

    print(f"\n{len(qualifying)} tickers passed ROIC screen out of {len(all_tickers)} total.\n")

    # Phase 2: Full analysis on qualifying tickers
    print("Fetching market history for CAPM benchmark...")
    market_history = yf_client.fetch_history(MARKET_TICKER, period="5y")

    os.makedirs("output", exist_ok=True)
    pdf_filename = os.path.join("output", "stock_analysis_results.pdf")
    with PdfPages(pdf_filename) as pdf:
        for ticker in qualifying:
            print(f"Analyzing {ticker}...")
            try:
                sec_data = sec_client.fetch_filings(ticker)
                yf_data = yf_client.fetch_financials(ticker)
                roic_data = roic_cache.get(ticker)

                capm_result = run_capm(yf_client, ticker, market_history)
                if capm_result and capm_result['expected_return'] and capm_result['expected_return'] > 0:
                    discount_rate = capm_result['expected_return']
                else:
                    discount_rate = 0.10
                dcf_result = run_dcf(yf_data, discount_rate)
                ratios = compute_ratios(yf_data)
                trend_data = compute_ratios_trend(yf_data)

                fig, ax = plt.subplots(figsize=(8, 6))
                ax.axis('off')
                summary = format_summary(ticker, capm_result, dcf_result, ratios, roic_data, trend_data)
                ax.text(0.05, 0.95, summary, va='top', ha='left', fontsize=10,
                        family='monospace', transform=ax.transAxes)
                pdf.savefig(fig)
                plt.close(fig)
            except Exception as e:
                print(f"Error analyzing {ticker}: {e}")
    print(f"Analysis complete. Results saved to {pdf_filename}")
