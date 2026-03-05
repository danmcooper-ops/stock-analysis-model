# data/finnhub_client.py
import finnhub
import pandas as pd

class FinnhubClient:
    def __init__(self, api_key):
        self.client = finnhub.Client(api_key=api_key)

    def fetch_financials(self, symbol):
        bs = self.client.financials_reported(symbol=symbol, freq='annual')
        # Parse bs for balance sheet, income statement, cash flow
        # Return as pandas DataFrame or dict
        return bs
