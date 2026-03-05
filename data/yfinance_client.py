# data/yfinance_client.py
import yfinance as yf
import pandas as pd

class YFinanceClient:
    def fetch_financials(self, ticker):
        stock = yf.Ticker(ticker)
        financials = {
            'balance_sheet': stock.balance_sheet,
            'income_statement': stock.financials,
            'cash_flow': stock.cashflow
        }
        return financials
