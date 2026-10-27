"""Positions on portfolio groupings: lots, cash, realized sales, targets.

A portfolio (``models/portfolio_groups.py``) is membership first. Two
optional layers make one a holdings record too, kept apart because the
repository and its GitHub Pages report are public:

* **Strategy**, in ``portfolio/portfolios.json`` beside the definition
  (published, shared by links and exports — weights, never dollars)::

    "targets": {"DECK": 0.10, "CL": 0.05},
    "weighting": "held"

* **Book**, in the private ``portfolio/positions.json`` (gitignored, never
  shipped in the report), keyed by portfolio id::

    {"version": 1, "portfolios": {"core": {
        "positions": {"DECK": [{"shares": 25, "cost": 88.5,
                                "date": "2024-06-10", "note": "Core"}]},
        "realized":  [{"ticker": "MSFT", "shares": 10, "cost": 310.0,
                       "price": 415.0, "date": "2025-09-12",
                       "bought": "2023-01-04"}],
        "cash": 1250.0}}}

``cost`` and ``price`` are per share. Lots are kept as bought (never
averaged in storage), so a sale consumes them first-in first-out and each
realized entry carries the cost and purchase date of the lot it came from.
A held or targeted ticker is also a hand-picked member (the CLI adds it to
``tickers``) and cannot be excluded.

``weighting`` says how the portfolio's stats and NAV weigh members:
``equal`` (the default, one member one vote), ``model`` (the run's
``position_weight``, ``models/portfolio.position_sizes``), ``target``
(``targets``) or ``held`` (shares × price, from the book; a portfolio
with no book falls back to equal).

``POSITIONS_FILE`` overrides the book's path. Apart from the file I/O at
the end, every function here is pure.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import date

WEIGHTINGS = ('equal', 'model', 'target', 'held')
STRATEGY_KEYS = ('targets', 'weighting')
BOOK_KEYS = ('positions', 'realized', 'cash')
BOOK_VERSION = 1
DEFAULT_POSITIONS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'portfolio', 'positions.json')
_EPS = 1e-9


def _num(v, where, lo=None, strict=False):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"{where}: must be a finite number, got {v!r}")
    if lo is not None and (v <= lo if strict else v < lo):
        raise ValueError(f"{where}: must be {'>' if strict else '>='} {lo}, got {v!r}")
    return float(v)


def _day(v, where):
    try:
        return date.fromisoformat(v).isoformat()
    except (TypeError, ValueError):
        raise ValueError(f"{where}: date must be YYYY-MM-DD, got {v!r}") from None


def _ticker(t, where):
    if not isinstance(t, str) or not t.strip():
        raise ValueError(f"{where}: bad ticker {t!r}")
    return t.strip().upper()


def _note(d, out):
    n = str(d.get('note') or '').strip()
    if n:
        out['note'] = n
    return out


def normalize_lot(lot, where):
    if not isinstance(lot, dict):
        raise ValueError(f"{where}: a lot must be an object")
    unknown = set(lot) - {'shares', 'cost', 'date', 'note'}
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {sorted(unknown)}")
    return _note(lot, {'shares': _num(lot.get('shares'), where + '.shares', 0, strict=True),
                       'cost': _num(lot.get('cost'), where + '.cost', 0),
                       'date': _day(lot.get('date'), where + '.date')})


def normalize_sale(s, where):
    if not isinstance(s, dict):
        raise ValueError(f"{where}: a sale must be an object")
    unknown = set(s) - {'ticker', 'shares', 'cost', 'price', 'date', 'bought', 'note'}
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {sorted(unknown)}")
    out = {'ticker': _ticker(s.get('ticker'), where + '.ticker'),
           'shares': _num(s.get('shares'), where + '.shares', 0, strict=True),
           'cost': _num(s.get('cost'), where + '.cost', 0),
           'price': _num(s.get('price'), where + '.price', 0),
           'date': _day(s.get('date'), where + '.date')}
    if s.get('bought'):
        out['bought'] = _day(s['bought'], where + '.bought')
    return _note(s, out)


def normalize_book(p, pid):
    """The canonical book (positions / realized / cash) of raw dict *p*,
    holding only the keys that are set."""
    if not isinstance(p, dict):
        raise ValueError(f"{pid}: a book must be an object")
    unknown = set(p) - set(BOOK_KEYS)
    if unknown:
        raise ValueError(f"{pid}: unknown book key(s) {sorted(unknown)}")
    out = {}
    raw = p.get('positions')
    if raw:
        if not isinstance(raw, dict):
            raise ValueError(f"{pid}.positions: must be an object of ticker -> lots")
        pos = {}
        for t, lots in raw.items():
            tk = _ticker(t, f"{pid}.positions")
            if not isinstance(lots, list):
                raise ValueError(f"{pid}.positions.{tk}: must be a list of lots")
            norm = [normalize_lot(lot, f"{pid}.positions.{tk}[{i}]") for i, lot in enumerate(lots)]
            if norm:
                pos.setdefault(tk, []).extend(norm)
        for tk in pos:
            pos[tk].sort(key=lambda lot: lot['date'])
        if pos:
            out['positions'] = dict(sorted(pos.items()))
    sales = p.get('realized')
    if sales:
        if not isinstance(sales, list):
            raise ValueError(f"{pid}.realized: must be a list")
        out['realized'] = sorted((normalize_sale(s, f"{pid}.realized[{i}]")
                                  for i, s in enumerate(sales)),
                                 key=lambda s: (s['date'], s['ticker']))
    if p.get('cash') is not None:
        cash = _num(p['cash'], f"{pid}.cash", 0)
        if cash:
            out['cash'] = cash
    return out


def normalize_strategy(p, pid):
    """The canonical strategy fields (targets / weighting) of raw portfolio
    *p*, holding only the keys that are set — a portfolio without them gets
    ``{}``, so files and revisions from before they existed are unchanged."""
    out = {}
    raw_t = p.get('targets')
    if raw_t:
        if not isinstance(raw_t, dict):
            raise ValueError(f"{pid}.targets: must be an object of ticker -> weight")
        tg = {_ticker(t, f"{pid}.targets"): _num(w, f"{pid}.targets.{t}", 0, strict=True)
              for t, w in raw_t.items()}
        if any(w > 1 for w in tg.values()) or sum(tg.values()) > 1 + 1e-6:
            raise ValueError(f"{pid}.targets: weights are fractions that sum to at most 1")
        out['targets'] = dict(sorted(tg.items()))
    w = p.get('weighting') or 'equal'
    if w not in WEIGHTINGS:
        raise ValueError(f"{pid}: weighting must be one of {', '.join(WEIGHTINGS)}")
    if w == 'target' and 'targets' not in out:
        raise ValueError(f"{pid}: weighting 'target' needs targets")
    if w != 'equal':
        out['weighting'] = w
    return out


def has_strategy(p):
    return any(k in p for k in STRATEGY_KEYS)


# ---------------------------------------------------------------------------
# Trades
# ---------------------------------------------------------------------------

def buy(p, ticker, shares, cost, day, note=None):
    """Add a lot to book *p* in place (cash is untouched: a book's cash is
    what the user says it is, not a ledger)."""
    tk = _ticker(ticker, 'buy')
    lot = normalize_lot({'shares': shares, 'cost': cost, 'date': day,
                         **({'note': note} if note else {})}, f"buy {tk}")
    pos = dict(p.get('positions') or {})
    pos[tk] = sorted(list(pos.get(tk) or []) + [lot], key=lambda x: x['date'])
    p['positions'] = pos
    return lot


def sell(p, ticker, shares, price, day, note=None):
    """Sell *shares* of *ticker* first-in first-out, in place.

    Each lot consumed (or partly consumed) becomes one realized entry with
    that lot's cost and purchase date. Raises ValueError when the
    portfolio holds fewer shares. Returns the new realized entries.
    """
    tk = _ticker(ticker, 'sell')
    left = _num(shares, f"sell {tk}.shares", 0, strict=True)
    price = _num(price, f"sell {tk}.price", 0)
    day = _day(day, f"sell {tk}.date")
    lots = [dict(lot) for lot in (p.get('positions') or {}).get(tk) or []]
    held = sum(lot['shares'] for lot in lots)
    if held + _EPS < left:
        raise ValueError(f"sell {tk}: holds {held:g} shares, cannot sell {left:g}")
    sales = []
    while left > _EPS and lots:
        lot = lots[0]
        take = min(lot['shares'], left)
        sale = {'ticker': tk, 'shares': take, 'cost': lot['cost'], 'price': price,
                'date': day, 'bought': lot['date']}
        if note:
            sale['note'] = note
        sales.append(sale)
        lot['shares'] -= take
        left -= take
        if lot['shares'] <= _EPS:
            lots.pop(0)
    pos = dict(p.get('positions') or {})
    if lots:
        pos[tk] = lots
    else:
        pos.pop(tk, None)
    if pos:
        p['positions'] = pos
    else:
        p.pop('positions', None)
    p['realized'] = list(p.get('realized') or []) + sales
    return sales


# ---------------------------------------------------------------------------
# Valuation of the holdings
# ---------------------------------------------------------------------------

def _price(row):
    """A row's usable price: a finite positive number, or numeric text (as
    the report's ``_num`` reads it, so the two holdings views agree)."""
    v = (row or {}).get('price')
    if isinstance(v, str):
        try:
            v = float(v)
        except ValueError:
            return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
        return None
    return float(v)


def shares_as_of(book, day=None):
    """{ticker: shares} the book held at the end of *day* (all open lots
    when None): open lots bought by then, plus lots sold after it (a
    realized entry with its ``bought`` date). A sale migrated without a
    purchase date cannot be placed and is left out."""
    out = {}
    for t, lots in (book.get('positions') or {}).items():
        sh = sum(lot['shares'] for lot in lots if day is None or lot['date'] <= day)
        if sh > 0:
            out[t] = sh
    if day is not None:
        for s in book.get('realized') or ():
            if s.get('bought') and s['bought'] <= day < s['date']:
                out[s['ticker']] = out.get(s['ticker'], 0.0) + s['shares']
    return out


def realized_by_year(realized):
    """{year: realized gain} over *realized* entries."""
    out = {}
    for s in realized or ():
        y = s['date'][:4]
        out[y] = out.get(y, 0.0) + s['shares'] * (s['price'] - s['cost'])
    return out


def holdings_summary(p, by_tk, aliases=None, prices=None, today=None):
    """Value book *p*'s positions against today's rows.

    *prices* ({ticker: price}) overrides a row's price (live quotes, or a
    holding the universe no longer carries). A held ticker that was folded
    into another listing (``aliases``) is priced from the kept row.

    Returns ``{'rows': [...], 'totals': {...}}`` or None for a portfolio
    that is empty. Each row: ticker, shares, cost (average
    per share), cost_value, first (earliest lot date), lots, price,
    market_value, pnl, pnl_pct, weight (of total value incl. cash), and
    note (the newest lot's). Totals: cost, market_value (priced holdings),
    unpriced (tickers with no price), cash, value (market_value + cash),
    pnl, pnl_pct, realized (all time), realized_ytd, realized_by_year.
    """
    pos = p.get('positions') or {}
    cash = float(p.get('cash') or 0.0)
    if not pos and not cash and not p.get('realized'):
        return None
    aliases = aliases or {}
    prices = prices or {}
    rows = []
    for tk, lots in pos.items():
        shares = sum(lot['shares'] for lot in lots)
        cost_value = sum(lot['shares'] * lot['cost'] for lot in lots)
        kept = tk if tk in by_tk else aliases.get(tk, tk)
        px = prices.get(tk) or prices.get(kept) or _price(by_tk.get(kept))
        mv = shares * px if px is not None else None
        pnl = mv - cost_value if mv is not None else None
        notes = [lot['note'] for lot in lots if lot.get('note')]
        rows.append({'ticker': tk, 'row': kept, 'shares': shares,
                     'cost': cost_value / shares if shares else None,
                     'cost_value': cost_value, 'first': lots[0]['date'],
                     'lots': len(lots), 'price': px, 'market_value': mv, 'pnl': pnl,
                     'pnl_pct': pnl / cost_value if pnl is not None and cost_value > 0 else None,
                     'note': notes[-1] if notes else ''})
    mv_total = sum(r['market_value'] for r in rows if r['market_value'] is not None)
    priced_cost = sum(r['cost_value'] for r in rows if r['market_value'] is not None)
    value = mv_total + cash
    for r in rows:
        r['weight'] = r['market_value'] / value if r['market_value'] is not None and value > 0 else None
    rows.sort(key=lambda r: -(r['market_value'] or 0))
    by_year = realized_by_year(p.get('realized'))
    year = (today or date.today().isoformat())[:4]
    pnl = mv_total - priced_cost if rows else None
    return {'rows': rows, 'totals': {
        'cost': sum(r['cost_value'] for r in rows),
        'market_value': mv_total,
        'unpriced': [r['ticker'] for r in rows if r['market_value'] is None],
        'cash': cash,
        'value': value,
        'pnl': pnl,
        'pnl_pct': pnl / priced_cost if pnl is not None and priced_cost > 0 else None,
        'realized': sum(by_year.values()),
        'realized_ytd': by_year.get(year, 0.0),
        'realized_by_year': by_year,
    }}


# ---------------------------------------------------------------------------
# Migration from portfolio/holdings.json
# ---------------------------------------------------------------------------

def from_holdings_doc(doc):
    """A book built from the old ``portfolio/holdings.json`` shape
    (``models/portfolio_tracker.py``)."""
    pos = {}
    for h in doc.get('holdings') or ():
        lot = {'shares': h['shares'], 'cost': h['cost_basis'], 'date': h['purchase_date']}
        if h.get('notes'):
            lot['note'] = h['notes']
        pos.setdefault(str(h['ticker']).upper(), []).append(lot)
    realized = []
    for rg in doc.get('realized_gains') or ():
        s = {'ticker': rg['ticker'], 'shares': rg['shares_sold'], 'cost': rg['cost_basis'],
             'price': rg['sale_price'], 'date': rg['sale_date']}
        if rg.get('notes'):
            s['note'] = rg['notes']
        realized.append(s)
    out = {'positions': pos, 'realized': realized}
    if doc.get('cash'):
        out['cash'] = doc['cash']
    return out


# ---------------------------------------------------------------------------
# The private book file
# ---------------------------------------------------------------------------

def positions_path(path=None):
    return path or os.environ.get('POSITIONS_FILE') or DEFAULT_POSITIONS_PATH


def normalize_books(doc):
    if not isinstance(doc, dict):
        raise ValueError("positions file must be a JSON object")
    if doc.get('version', BOOK_VERSION) != BOOK_VERSION:
        raise ValueError(f"unsupported positions version {doc.get('version')!r}")
    raw = doc.get('portfolios') or {}
    if not isinstance(raw, dict):
        raise ValueError("'portfolios' must be an object of id -> book")
    books = {str(pid): normalize_book(b, str(pid)) for pid, b in raw.items()}
    return {'version': BOOK_VERSION,
            'portfolios': {pid: b for pid, b in sorted(books.items()) if b}}


def load_books(path=None):
    """``{portfolio id: book}`` from the private file; missing is empty."""
    path = positions_path(path)
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as f:
        return normalize_books(json.load(f))['portfolios']


def save_books(books, path=None):
    """Validate and write atomically, owner-readable only."""
    path = positions_path(path)
    doc = normalize_books({'version': BOOK_VERSION, 'portfolios': books})
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.positions.', suffix='.tmp', dir=d)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
            f.write('\n')
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return doc['portfolios']
