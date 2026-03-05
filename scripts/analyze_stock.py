# scripts/analyze_stock.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.sec_edgar import SECEdgarClient
from data.yfinance_client import YFinanceClient
from models.capm import calculate_beta, expected_return
from models.dcf import calculate_dcf
from models.comparisons import compute_ratios

# Example usage
if __name__ == "__main__":
    ticker = "AAPL"

    # SEC EDGAR
    sec_email = "your_email@example.com"  # Replace with your email
    sec_client = SECEdgarClient(sec_email)
    sec_data = sec_client.fetch_filings(ticker)

    # yFinance
    yf_client = YFinanceClient()
    yf_data = yf_client.fetch_financials(ticker)


    # CAPM Example
    # stock_returns, market_returns = ...
    # beta = calculate_beta(stock_returns, market_returns)
    # er = expected_return(risk_free_rate, beta, market_return)

    # DCF Example
    # free_cash_flows, discount_rate, terminal_value, periods = ...
    # dcf_value = calculate_dcf(free_cash_flows, discount_rate, terminal_value, periods)

    # Comparison Example
    # ratios = compute_ratios(yf_data)

    print("Analysis complete.")
