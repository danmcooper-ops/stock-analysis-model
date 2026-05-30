"""FDIC BankFind Suite API client.

Read-only access to FDIC institutions and financials endpoints. Caches
responses on disk so repeated runs don't re-hit the API.

API base: https://api.fdic.gov/banks/
Docs:     https://banks.data.fdic.gov/docs/

Financials endpoint returns one row per institution per quarter. Key
fields used here:
- ROA, ROE                   return on assets / equity (percent units)
- NIMY                       net interest margin year-to-date (percent)
- EEFFR                      efficiency ratio = nonint expense / revenue (percent)
- RBCT1CER                   Common Equity Tier 1 capital ratio (percent)
- NCLNLS / LNLSGR            non-current loans / total loans-and-leases
- ASSET, DEP                 total assets, deposits (dollars)
"""
import json
import os
import time
import urllib.parse
import urllib.request

_BASE = "https://api.fdic.gov/banks"
_CACHE = os.path.join(os.path.dirname(__file__), "cache", "fdic")
_FIN_FIELDS = "CERT,REPDTE,ASSET,DEP,NETINC,ROA,ROE,NIMY,EEFFR,RBCT1CER,NCLNLS,LNLSGR"
_DEFAULT_TTL_DAYS = 30


def _cache_path(key):
    os.makedirs(_CACHE, exist_ok=True)
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in key)
    return os.path.join(_CACHE, safe[-200:] + ".json")


def _get(url, ttl_days=_DEFAULT_TTL_DAYS):
    cp = _cache_path(url)
    if os.path.exists(cp) and time.time() - os.path.getmtime(cp) < ttl_days * 86400:
        with open(cp) as f:
            return json.load(f)
    req = urllib.request.Request(url, headers={"User-Agent": "stock-analysis/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read().decode())
    with open(cp, "w") as f:
        json.dump(data, f)
    return data


def institutions(name=None, cert=None, fields="CERT,NAME,STNAME,ASSET,ACTIVE", limit=5):
    """Look up institutions by CERT (exact) or by NAME (LIKE pattern)."""
    params = {"fields": fields, "limit": str(limit)}
    if cert is not None:
        params["filters"] = "CERT:" + str(cert)
    elif name:
        params["filters"] = 'NAME:"' + name + '"'
    else:
        raise ValueError("Provide either cert or name")
    url = _BASE + "/institutions?" + urllib.parse.urlencode(params)
    resp = _get(url)
    return [d["data"] for d in resp.get("data", [])]


def financials(cert, repdte=None, fields=_FIN_FIELDS):
    """Return the latest call-report financials for a CERT.

    Returns a dict with the requested fields, or None if no record exists.
    """
    flt = "CERT:" + str(cert)
    if repdte:
        flt += " AND REPDTE:" + str(repdte)
    params = {
        "filters": flt,
        "fields": fields,
        "limit": "1",
        "sort_by": "REPDTE",
        "sort_order": "DESC",
    }
    url = _BASE + "/financials?" + urllib.parse.urlencode(params)
    resp = _get(url)
    items = resp.get("data", [])
    return items[0]["data"] if items else None
