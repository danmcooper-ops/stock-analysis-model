"""User-defined portfolio groupings: named sets of tickers, hand-picked,
rule-driven, or both.

A portfolio is *membership only* (no shares or cost basis — that is
``portfolio/holdings.json``'s job). A ticker may sit in any number of
portfolios. Definitions live in ``portfolio/portfolios.json``:

    {"version": 1,
     "portfolios": [
        {"id": "semis", "name": "Semiconductors", "color": "#6b8afd",
         "description": "", "created": "2026-09-24",
         "tickers": ["NVDA", "AMD"], "exclude": [], "rule": null},
        {"id": "energy-buys", "name": "Energy BUYs", "created": "2026-09-24",
         "tickers": [], "exclude": ["XOM"],
         "rule": {"ratings": ["BUY", "LEAN BUY"], "sectors": ["Energy"],
                  "countries": null,
                  "cf": [{"key": "mcap", "min": 2e9, "max": null},
                         {"key": "industry", "txt": "OIL"}]}}]}

Members = (``tickers`` ∪ rows matching ``rule``) − ``exclude``.

A rule is the report's own filter state (Ratings / Sectors / Countries
multi-selects plus the column filters), so "save the current filter as a
portfolio" and "open a portfolio as a filter" are the same object. The one
difference: min/max on percent-formatted columns are stored as *fractions*
(the raw row value), never the percent typed into the UI — the report
converts on save/restore, so a rule never depends on display units.

``rule_matches`` mirrors ``passOther`` in ``templates/report.html``
clause for clause; ``tests/fixtures/portfolio_rule_cases.json`` pins both
implementations to the same answers.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from datetime import date

SCHEMA_VERSION = 1

DEFAULT_PORTFOLIOS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'portfolio', 'portfolios.json')

# Assigned in id order to portfolios without an explicit color, so a
# portfolio keeps its color across runs as long as the set of colorless
# portfolios doesn't change. Mid-saturation hues that read on both the
# light and dark report themes.
PALETTE = ['#4f7cff', '#e8833a', '#2fa36b', '#c9469b', '#8a63d2',
           '#d4a72c', '#1fa2b8', '#d9534f', '#6c8a3a', '#9b6b43']

_ID_RE = re.compile(r'^[a-z0-9][a-z0-9-]{0,39}$')
_COLOR_RE = re.compile(r'^#[0-9a-fA-F]{6}$')
_RULE_LIST_KEYS = ('ratings', 'sectors', 'countries')


def slugify(name):
    """A valid portfolio id from a display name ('Energy BUYs' -> 'energy-buys')."""
    s = re.sub(r'[^a-z0-9]+', '-', str(name).lower()).strip('-')
    return s[:40].rstrip('-') or 'portfolio'


def _norm_tickers(seq, where):
    if seq is None:
        return []
    if not isinstance(seq, list):
        raise ValueError(f"{where}: must be a list of tickers")
    out, seen = [], set()
    for t in seq:
        if not isinstance(t, str) or not t.strip():
            raise ValueError(f"{where}: bad ticker {t!r}")
        u = t.strip().upper()
        if u not in seen:
            seen.add(u)
            out.append(u)
    return sorted(out)


def _norm_bound(v, where):
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"{where}: bound must be a finite number or null, got {v!r}")
    return v


def normalize_rule(rule, where='rule'):
    """Validate and canonicalize a rule dict; None stays None.

    Raises ValueError on a malformed rule, and on a rule with no
    constraint at all — that would silently make the portfolio the whole
    universe.
    """
    if rule is None:
        return None
    if not isinstance(rule, dict):
        raise ValueError(f"{where}: must be an object or null")
    unknown = set(rule) - set(_RULE_LIST_KEYS) - {'cf'}
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {sorted(unknown)}")
    out = {}
    for k in _RULE_LIST_KEYS:
        v = rule.get(k)
        if v is None:
            out[k] = None
            continue
        if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            raise ValueError(f"{where}.{k}: must be a list of strings or null")
        out[k] = sorted(set(v))
    cf_out = []
    for i, c in enumerate(rule.get('cf') or []):
        w = f"{where}.cf[{i}]"
        if not isinstance(c, dict) or not isinstance(c.get('key'), str) or not c['key']:
            raise ValueError(f"{w}: needs a column 'key'")
        if 'txt' in c and c['txt'] is not None:
            if set(c) - {'key', 'txt'}:
                raise ValueError(f"{w}: a text clause takes only key and txt")
            t = str(c['txt']).strip().upper()
            if not t:
                raise ValueError(f"{w}: empty text")
            cf_out.append({'key': c['key'], 'txt': t})
            continue
        if set(c) - {'key', 'min', 'max'}:
            raise ValueError(f"{w}: a numeric clause takes only key, min and max")
        lo = _norm_bound(c.get('min'), w + '.min')
        hi = _norm_bound(c.get('max'), w + '.max')
        if lo is None and hi is None:
            raise ValueError(f"{w}: needs min and/or max")
        cf_out.append({'key': c['key'], 'min': lo, 'max': hi})
    out['cf'] = cf_out
    if all(out[k] is None for k in _RULE_LIST_KEYS) and not cf_out:
        raise ValueError(f"{where}: has no constraints (it would match every stock)")
    return out


def normalize_portfolio(p, today=None):
    if not isinstance(p, dict):
        raise ValueError("portfolio entry must be an object")
    pid = p.get('id')
    if not isinstance(pid, str) or not _ID_RE.match(pid):
        raise ValueError(f"portfolio id {pid!r}: use 1-40 chars of a-z, 0-9 and '-'")
    name = p.get('name')
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{pid}: needs a name")
    color = p.get('color')
    if color is not None and not (isinstance(color, str) and _COLOR_RE.match(color)):
        raise ValueError(f"{pid}: color must look like #rrggbb")
    created = p.get('created') or (today or date.today().isoformat())
    try:
        date.fromisoformat(created)
    except (TypeError, ValueError):
        raise ValueError(f"{pid}: created must be YYYY-MM-DD") from None
    out = {
        'id': pid,
        'name': name.strip(),
        'color': color.lower() if color else None,
        'description': str(p.get('description') or ''),
        'created': created,
        'tickers': _norm_tickers(p.get('tickers'), f"{pid}.tickers"),
        'exclude': _norm_tickers(p.get('exclude'), f"{pid}.exclude"),
        'rule': normalize_rule(p.get('rule'), f"{pid}.rule"),
    }
    # An empty portfolio (no tickers, no rule) is legitimate: just created,
    # not filled yet.
    return out


def normalize(doc, today=None):
    """Validate a whole definitions document; returns the canonical form."""
    if not isinstance(doc, dict):
        raise ValueError("portfolios file must be a JSON object")
    ver = doc.get('version', SCHEMA_VERSION)
    if ver != SCHEMA_VERSION:
        raise ValueError(f"unsupported portfolios version {ver!r}")
    raw = doc.get('portfolios') or []
    if not isinstance(raw, list):
        raise ValueError("'portfolios' must be a list")
    pfs = [normalize_portfolio(p, today) for p in raw]
    seen = set()
    for p in pfs:
        if p['id'] in seen:
            raise ValueError(f"duplicate portfolio id {p['id']!r}")
        seen.add(p['id'])
    return {'version': SCHEMA_VERSION, 'portfolios': pfs}


def load_portfolios(path=None):
    """Read and validate the definitions file. A missing file is an empty set."""
    path = path or DEFAULT_PORTFOLIOS_PATH
    if not os.path.exists(path):
        return {'version': SCHEMA_VERSION, 'portfolios': []}
    with open(path, encoding='utf-8') as f:
        doc = json.load(f)
    return normalize(doc)


def save_portfolios(doc, path=None):
    """Validate and write atomically; indented so git diffs stay readable.

    Colors are materialized (see ``with_colors``) so the file always holds
    exactly what the report shows — a browser export then never differs
    from the file by a palette color alone.
    """
    path = path or DEFAULT_PORTFOLIOS_PATH
    doc = normalize(doc)
    doc['portfolios'] = with_colors(doc['portfolios'])
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.portfolios.', suffix='.tmp', dir=d)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
            f.write('\n')
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return doc


def revision(doc):
    """Short content hash of the definitions; the report compares it to
    the revision a browser's local edits were based on."""
    blob = json.dumps(normalize(doc), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:12]


def with_colors(portfolios):
    """Copies of the portfolios with every color filled in (see PALETTE)."""
    out, i = [], 0
    for p in sorted(portfolios, key=lambda x: x['id']):
        q = dict(p)
        if not q.get('color'):
            q['color'] = PALETTE[i % len(PALETTE)]
            i += 1
        out.append(q)
    order = {p['id']: n for n, p in enumerate(portfolios)}
    out.sort(key=lambda q: order[q['id']])
    return out


# ---------------------------------------------------------------------------
# Rule evaluation — must stay in lockstep with passOther() in report.html
# ---------------------------------------------------------------------------

_JS_NUM_RE = re.compile(r'^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$')


def js_num(v):
    """Python twin of the report's ``_num()``: JS ``+v`` restricted to
    finite results; anything else is None (N/A)."""
    if v is None:
        return None
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, str):
        s = v.strip()
        if s == '':
            return 0.0          # JS: +'' === 0
        if not _JS_NUM_RE.match(s):
            return None         # includes 'Infinity', 'NaN', '1_000'
        f = float(s)
        return f if math.isfinite(f) else None
    return None


def _js_str(v):
    if v is None:
        return ''
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, float) and v.is_integer() and math.isfinite(v):
        return str(int(v))
    return str(v)


def rule_matches(rule, row):
    """True when *row* passes every clause of *rule* (a normalized rule).

    ``ratings``/``sectors``/``countries``: None means any; a list means the
    row's value must be in it (an empty list matches nothing, as in the
    report). Numeric clauses drop rows whose value is N/A per ``js_num``,
    or whose scoring-gate twin (``_gate_<key>``) is present but N/A — the
    report does the same so a filter never passes a row the Gate Matrix
    shows as N/A. Text clauses are case-insensitive substring matches.
    """
    if not rule:
        return False
    if rule.get('ratings') is not None and row.get('rating') not in rule['ratings']:
        return False
    if rule.get('sectors') is not None and row.get('sector') not in rule['sectors']:
        return False
    if rule.get('countries') is not None and row.get('country') not in rule['countries']:
        return False
    for c in rule.get('cf') or ():
        key = c['key']
        if c.get('txt') is not None:
            if c['txt'] not in _js_str(row.get(key)).upper():
                return False
            continue
        v = js_num(row.get(key))
        if v is None:
            return False
        gk = '_gate_' + key
        if gk in row and js_num(row[gk]) is None:
            return False
        if c.get('min') is not None and v < c['min']:
            return False
        if c.get('max') is not None and v > c['max']:
            return False
    return True


def resolve_members(portfolio, rows_by_ticker):
    """Resolve one portfolio against a universe.

    Returns ``{'members': [...], 'missing': [...], 'manual': [...],
    'ruled': [...]}``, each sorted. ``members`` are tickers present in
    *rows_by_ticker*; hand-picked tickers that are not are listed in
    ``missing`` (kept in the definition — they may come back tomorrow).
    """
    excl = set(portfolio.get('exclude') or ())
    manual = [t for t in portfolio.get('tickers') or () if t not in excl]
    rule = portfolio.get('rule')
    ruled = []
    if rule:
        ruled = [t for t, r in rows_by_ticker.items()
                 if t not in excl and rule_matches(rule, r)]
    present = {t for t in manual if t in rows_by_ticker} | set(ruled)
    return {
        'members': sorted(present),
        'missing': sorted(t for t in manual if t not in rows_by_ticker),
        'manual': sorted(manual),
        'ruled': sorted(ruled),
    }


def rows_by_ticker(rows):
    return {str(r['ticker']).upper(): r for r in rows if r.get('ticker')}


def resolve_all(portfolios, rows):
    """{portfolio id: resolve_members(...)} for every portfolio."""
    by_tk = rows_by_ticker(rows)
    return {p['id']: resolve_members(p, by_tk) for p in portfolios}


def membership_index(portfolios, rows):
    """{ticker: [portfolio ids]} over the tickers present in *rows*,
    ids in definition order. Tickers in no portfolio are omitted."""
    resolved = resolve_all(portfolios, rows)
    idx = {}
    for p in portfolios:
        for t in resolved[p['id']]['members']:
            idx.setdefault(t, []).append(p['id'])
    return idx


# ---------------------------------------------------------------------------
# Merging / diffing definitions (the CLI's `import`, the report's export)
# ---------------------------------------------------------------------------

def decode_share(text):
    """Parse a share link (``…#pf=<base64url JSON>``), a bare ``pf=`` token,
    or plain JSON text (a report export or one portfolio) into
    ``(definitions document, base_rev or None)``. ``base_rev`` is the
    revision of the file a report export's edits started from."""
    import base64
    s = text.strip()
    m = re.search(r'(?:^|[#&])pf=([A-Za-z0-9_\-]+=*)', s)
    if m:
        tok = m.group(1)
        tok += '=' * (-len(tok) % 4)
        s = base64.urlsafe_b64decode(tok.encode('ascii')).decode('utf-8')
    obj = json.loads(s)
    if isinstance(obj, dict) and 'portfolios' not in obj and 'id' in obj:
        obj = {'version': SCHEMA_VERSION, 'portfolios': [obj]}
    base = obj.get('base_rev') if isinstance(obj, dict) else None
    return normalize(obj), (base if isinstance(base, str) else None)


def _rule_summary(rule):
    if not rule:
        return 'none'
    parts = []
    for k in _RULE_LIST_KEYS:
        if rule.get(k) is not None:
            parts.append(f"{k}={','.join(rule[k]) or '(none)'}")
    for c in rule.get('cf') or ():
        if c.get('txt') is not None:
            parts.append(f"{c['key']}~{c['txt']}")
        else:
            lo = '' if c.get('min') is None else f"{c['min']:g}<="
            hi = '' if c.get('max') is None else f"<={c['max']:g}"
            parts.append(f"{lo}{c['key']}{hi}")
    return '; '.join(parts)


def diff_portfolios(old, new):
    """Human-readable lines describing how *new* differs from *old*
    (both lists of normalized portfolios)."""
    o = {p['id']: p for p in old}
    n = {p['id']: p for p in new}
    lines = []
    for pid in sorted(n.keys() - o.keys()):
        p = n[pid]
        lines.append(f"+ {pid} ({p['name']}): {len(p['tickers'])} tickers, "
                     f"rule: {_rule_summary(p['rule'])}")
    for pid in sorted(o.keys() - n.keys()):
        lines.append(f"- {pid} ({o[pid]['name']})")
    for pid in sorted(o.keys() & n.keys()):
        a, b = o[pid], n[pid]
        if a == b:
            continue
        lines.append(f"~ {pid}:")
        for fld in ('name', 'color', 'description', 'created'):
            if a.get(fld) != b.get(fld):
                lines.append(f"    {fld}: {a.get(fld)!r} -> {b.get(fld)!r}")
        for fld in ('tickers', 'exclude'):
            add = sorted(set(b[fld]) - set(a[fld]))
            rem = sorted(set(a[fld]) - set(b[fld]))
            if add:
                lines.append(f"    {fld} +{' +'.join(add)}")
            if rem:
                lines.append(f"    {fld} -{' -'.join(rem)}")
        if a['rule'] != b['rule']:
            lines.append(f"    rule: {_rule_summary(a['rule'])} -> {_rule_summary(b['rule'])}")
    return lines


def merge_portfolios(current, incoming, overwrite=False):
    """Merge *incoming* portfolios into *current* (lists, normalized).

    New ids are appended; identical ids are no-ops. An id present in both
    with different content raises ValueError naming the collisions unless
    *overwrite*, in which case the incoming version wins.
    """
    cur = {p['id']: p for p in current}
    clashes = [p['id'] for p in incoming if p['id'] in cur and cur[p['id']] != p]
    if clashes and not overwrite:
        raise ValueError("these portfolios differ from the file: "
                         + ', '.join(sorted(clashes))
                         + " (use --overwrite to take the imported versions)")
    out = [dict(p) for p in current]
    pos = {p['id']: i for i, p in enumerate(out)}
    for p in incoming:
        if p['id'] in pos:
            out[pos[p['id']]] = p
        else:
            pos[p['id']] = len(out)
            out.append(p)
    return out
