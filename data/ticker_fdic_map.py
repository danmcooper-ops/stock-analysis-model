"""Map public bank-holding-company tickers to FDIC certificate (CERT)
of the primary bank subsidiary.

CERTs verified via FDIC institutions endpoint where possible. For multi-
subsidiary holdings we use the largest sub by assets.

Coverage focuses on the ~40 largest US-listed banks by market cap;
smaller regionals can be added incrementally.
"""

TICKER_TO_CERT = {
    # Universal / global banks
    "JPM": 628,      # JPMorgan Chase Bank, NA
    "BAC": 3510,     # Bank of America, NA
    "WFC": 3511,     # Wells Fargo Bank, NA
    "C": 7213,       # Citibank, NA
    "USB": 6548,     # U.S. Bank, NA
    "PNC": 6384,     # PNC Bank, NA
    "TFC": 9846,     # Truist Bank
    "COF": 33954,    # Capital One, NA
    "BK": 639,       # Bank of New York Mellon
    "STT": 14,       # State Street Bank & Trust
    "TD": 18409,     # TD Bank, NA
    "NTRS": 913,     # Northern Trust
    # Investment / broker banks
    "GS": 33124,     # Goldman Sachs Bank USA
    "MS": 32992,     # Morgan Stanley Bank, NA
    "SCHW": 57450,   # Charles Schwab Bank
    # Cards / consumer finance
    "DFS": 5649,     # Discover Bank
    "AXP": 27471,    # American Express NB
    "ALLY": 57803,   # Ally Bank
    # Major regionals
    "MTB": 588,      # M&T Bank
    "FITB": 6672,    # Fifth Third Bank, NA
    "KEY": 17534,    # KeyBank, NA
    "RF": 12368,     # Regions Bank
    "HBAN": 6560,    # Huntington National Bank
    "CMA": 983,      # Comerica Bank
    "CFG": 57957,    # Citizens Bank, NA
    "FCNCA": 11063,  # First-Citizens Bank & Trust
    "WAL": 57512,    # Western Alliance Bank
    "EWBC": 31628,   # East West Bank
    "SNV": 873,      # Synovus Bank
    "BOKF": 4214,    # BOKF, NA
    "WBS": 18221,    # Webster Bank, NA
    "WTFC": 33935,   # Wintrust Bank
    "ZION": 2270,    # Zions Bancorporation NA
    "ONB": 3722,     # Old National Bank
    "BPOP": 17298,   # Banco Popular de Puerto Rico
    "FHN": 4977,     # First Horizon Bank
    "PNFP": 35583,   # Pinnacle Bank
    "VLY": 9396,     # Valley National Bank
    "UMBF": 8273,    # UMB Bank, NA
    "ASB": 5296,     # Associated Bank, NA
    "HOMB": 11241,   # Centennial Bank
    "FFIN": 4976,    # First Financial Bank
    "COLB": 23262,   # Columbia State Bank
    "FLG": 32541,    # Flagstar Bank (was NYCB)
    # Pending verification — disabled until correct CERT confirmed:
    # "CFR": ?,      # Frost Bank — CERT 952 returned 1991-dated record
    # "BPOP": ?,     # Banco Popular Puerto Rico — CERT 17298 had no data
}
