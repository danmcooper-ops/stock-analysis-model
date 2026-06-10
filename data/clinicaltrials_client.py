"""ClinicalTrials.gov API v2 client.

Read-only access to the public ClinicalTrials.gov study registry. Used to
approximate a drug company's active clinical pipeline depth by counting
the interventional studies it sponsors that are currently underway.

No API key required. Responses are cached on disk so repeated runs don't
re-hit the API.

API base: https://clinicaltrials.gov/api/v2/
Docs:     https://clinicaltrials.gov/data-api/api
"""
import json
import os
import re
import time
import urllib.parse
import urllib.request

_BASE = "https://clinicaltrials.gov/api/v2/studies"
_CACHE = os.path.join(os.path.dirname(__file__), "cache", "clinicaltrials")
_DEFAULT_TTL_DAYS = 30

# Studies that represent live pipeline activity (excludes completed,
# terminated, withdrawn, etc.).
_ACTIVE_STATUSES = "RECRUITING|ACTIVE_NOT_RECRUITING|ENROLLING_BY_INVITATION|NOT_YET_RECRUITING"

# Characters that carry meaning in the Essie advanced-query syntax used by
# filter.advanced. A sponsor name containing any of these (e.g. "Roche
# Holding (AG)") would otherwise produce a malformed query and a 400.
# We neutralize them to spaces so the lookup degrades to a plain
# token match rather than failing.
_ESSIE_META = re.compile(r'[\[\]()"{}]')


def _sanitize(name):
    """Strip Essie metacharacters and collapse whitespace."""
    if not name:
        return ""
    return " ".join(_ESSIE_META.sub(" ", name).split()).strip()


def _cache_path(key):
    os.makedirs(_CACHE, exist_ok=True)
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in key)
    return os.path.join(_CACHE, safe[-200:] + ".json")


def _get(url, ttl_days=_DEFAULT_TTL_DAYS, meta=None):
    """Cached GET. When `meta` is a dict, it is populated with
    cache_hit / cache_age_days so callers can record provenance."""
    cp = _cache_path(url)
    if os.path.exists(cp) and time.time() - os.path.getmtime(cp) < ttl_days * 86400:
        if meta is not None:
            meta.update(cache_hit=True,
                        cache_age_days=(time.time() - os.path.getmtime(cp)) / 86400.0)
        with open(cp) as f:
            return json.load(f)
    req = urllib.request.Request(url, headers={"User-Agent": "stock-analysis/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read().decode())
    with open(cp, "w") as f:
        json.dump(data, f)
    if meta is not None:
        meta.update(cache_hit=False, cache_age_days=0.0)
    return data


def active_pipeline_count(sponsor, meta=None):
    """Count active interventional studies sponsored by `sponsor`.

    Returns an int (0 or more), or None if the request fails. The count is
    a proxy for clinical-pipeline depth — it includes all active
    interventional trials the company is listed as a sponsor on, across
    phases. `None` is reserved for request errors so callers can
    distinguish "no pipeline" (0) from "lookup failed" (None).
    Pass a dict as `meta` to receive cache_hit / cache_age_days.
    """
    name = _sanitize(sponsor)
    if not name:
        return None
    # Match on the LEAD sponsor field, not a loose full-text query. The
    # loose `query.spons` does OR-token matching, so a common token like
    # "Pharmaceuticals" collapses many unrelated companies to the same
    # (wrong) count. AREA[LeadSponsorName] scopes the match to the lead
    # sponsor name, yielding distinct, company-specific counts.
    adv = "AREA[LeadSponsorName]%s AND AREA[StudyType]INTERVENTIONAL" % name
    params = {
        "filter.advanced": adv,
        "filter.overallStatus": _ACTIVE_STATUSES,
        "countTotal": "true",
        "pageSize": "1",
        "fields": "NCTId",
    }
    url = _BASE + "?" + urllib.parse.urlencode(params)
    try:
        resp = _get(url, meta=meta)
    except Exception:
        return None
    tc = resp.get("totalCount") if isinstance(resp, dict) else None
    return tc if isinstance(tc, int) else None
