# scripts/update_russell2000.py
"""Refresh data/russell2000_tickers.txt from Vanguard's VTWO holdings API.

VTWO tracks the Russell 2000, and Vanguard's public portfolio-holding API
returns the full constituent list as JSON (paginated, 500 per page) with no
auth — unlike the iShares IWM download, which sits behind a bot check.

Run after the annual Russell reconstitution (late June) or whenever the list
feels stale:

    python scripts/update_russell2000.py

Tickers are written in yfinance dash style (BF-B, not BF.B) to match the
rest of the pipeline. The report reads the file at render time
(see report_html.py), so a rescore_and_render picks up a refreshed list.
"""
import json
import os
import time
import urllib.request

_API = ("https://investor.vanguard.com/investment-products/etfs/profile/api/"
        "VTWO/portfolio-holding/stock?start={start}&count=500")
_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "data", "russell2000_tickers.txt")


def fetch_constituents():
    tickers = set()
    as_of = None
    start = 1
    while True:
        req = urllib.request.Request(_API.format(start=start),
                                     headers={"User-Agent": "Mozilla/5.0"})
        data = json.load(urllib.request.urlopen(req, timeout=30))
        as_of = (data.get("asOfDate") or as_of or "")[:10]
        entities = (data.get("fund") or {}).get("entity") or []
        if not entities:
            break
        for e in entities:
            t = (e.get("ticker") or "").strip().upper()
            if t and t not in ("N/A", "-"):
                tickers.add(t.replace(".", "-"))
        size = int(data.get("size") or 0)
        start += 500
        if start > size:
            break
        time.sleep(0.5)
    return sorted(tickers), as_of


def main():
    tickers, as_of = fetch_constituents()
    if len(tickers) < 1500:
        raise SystemExit(
            f"Only {len(tickers)} tickers returned — refusing to overwrite "
            f"{os.path.basename(_OUT)} with a suspiciously short list.")
    with open(_OUT, "w", encoding="utf-8") as f:
        f.write("# Russell 2000 constituents — sourced from Vanguard VTWO holdings\n")
        f.write("# (investor.vanguard.com portfolio-holding API). Refresh with:\n")
        f.write("#   python scripts/update_russell2000.py\n")
        f.write(f"# as_of: {as_of}  count: {len(tickers)}\n")
        f.write("# Tickers use yfinance dash style (BF-B, not BF.B).\n")
        for t in tickers:
            f.write(t + "\n")
    print(f"Wrote {len(tickers)} tickers (as of {as_of}) to {_OUT}")


if __name__ == "__main__":
    main()
