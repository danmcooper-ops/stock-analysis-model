# data/issuers.py
"""One row per issuer.

The universe lists some companies several times (see
``data.us_listings.fetch_issuer_map``): ADR and ordinary lines, preferred
series and notes without a dash, a filer's exchange-traded notes, second
share classes. Every copy carries the same filer's statements, so each one
added a second vote to sector medians, peer percentiles and profit pools,
and the copies contradicted each other on the page (54 of 181 such
companies on 2026-10-07 had different ratings on different lines).

Which line to keep is decided by where it trades, then how much:

1. NYSE / Nasdaq / CBOE before OTC;
2. the higher 3-month average dollar volume — SEC lists a filer's primary
   ticker first, but not always the right one: BIPH ahead of BIP (Brookfield
   Infrastructure's units, $28M a day against $0.2M), and for OTC-only
   foreign issuers the dormant ADR ahead of the traded ordinary line
   (Dollarama: DLMAY $0.01M, DLMAF $3.6M), whose stale quote skews MoS;
3. SEC's order, then the ticker, so the choice is deterministic.

Measured over the ten snapshots to 2026-10-07 the choice changed for three
companies, all thinly traded preferred-only or A/B pairs.

Pure functions; the issuer map is passed in. A row whose ticker the map does
not know is left alone.
"""
import math
from collections import defaultdict

MAJOR_EXCHANGES = frozenset({'NYSE', 'Nasdaq', 'CBOE'})

# FINRA's fifth-letter identifiers on five-letter OTC symbols: G/H/I mark
# convertible bonds, L "miscellaneous", M/N/O/P the 4th..1st preferred
# series. An issuer whose only public lines are preferreds (Ameren Illinois:
# AILIH..AILLO; Wisconsin Electric: WELPP) otherwise kept one of them as its
# row, rated as common stock (AILIN: HOLD at MoS +79% on 2026-10-08) —
# Yahoo reports every preferred as EQUITY with the issuer's name and common
# share count, and SEC cover tags are absent or ambiguous. Over SEC's whole
# list the rule matches 103 tickers, every one a preferred, convertible,
# depositary/CDI line or trust certificate; wherever the issuer has common
# stock it trades under another ticker. J/K (voting classes) and S/T are
# left alone: ambiguous, and the issuer collapse folds them when a common
# line exists. OTC only — exchange-listed five-letter symbols are classes
# (GOOGL, BELFB).
PREFERRED_FIFTH_LETTERS = frozenset('GHILMNOP')


def is_otc_preferred_symbol(ticker, issuer_map):
    """True for an OTC five-letter symbol carrying a preferred/convertible
    fifth letter (see above); False for anything the map does not know."""
    t = str(ticker or '').upper()
    return (len(t) == 5 and t.isalpha() and t[-1] in PREFERRED_FIFTH_LETTERS
            and _issuer(issuer_map, t).get('exchange') == 'OTC')


def _issuer(issuer_map, ticker):
    return (issuer_map or {}).get(str(ticker or '').upper()) or {}


def _listing_rank(issuer_map, ticker, dollar_volume=None):
    info = _issuer(issuer_map, ticker)
    dv = dollar_volume if (isinstance(dollar_volume, (int, float))
                           and math.isfinite(dollar_volume)) else 0.0
    return (info.get('exchange') not in MAJOR_EXCHANGES, -dv,
            info.get('rank', math.inf), str(ticker))


def collapse_duplicate_listings(rows, issuer_map):
    """Keep one row per CIK; return ``(rows, folded)``.

    *rows* keeps its order. OTC preferred symbols (``is_otc_preferred_symbol``)
    are removed first and appear in *folded* mapped to ``None``. Each kept
    row that absorbed others carries ``listing_aliases`` (the folded tickers,
    sorted), so anything reading the snapshot can resolve an old ticker to
    the row that now stands for it; *folded* is ``{folded ticker: kept
    ticker}``. A ``listing_aliases`` left over from an earlier pass (a
    carried-forward row) is replaced.
    """
    groups = defaultdict(list)
    drop, folded = set(), {}
    for r in rows:
        r.pop('listing_aliases', None)
        if is_otc_preferred_symbol(r.get('ticker'), issuer_map):
            drop.add(id(r))
            folded[r['ticker']] = None
            continue
        cik = _issuer(issuer_map, r.get('ticker')).get('cik')
        if cik:
            groups[cik].append(r)
    for members in groups.values():
        if len(members) < 2:
            continue
        members = sorted(members, key=lambda r: _listing_rank(
            issuer_map, r.get('ticker'), r.get('avg_dollar_volume_3m')))
        keep = members[0]
        gone = [m['ticker'] for m in members[1:]]
        keep['listing_aliases'] = sorted(gone)
        for m, t in zip(members[1:], gone, strict=True):
            drop.add(id(m))
            folded[t] = keep['ticker']
    return [r for r in rows if id(r) not in drop], folded


def one_listing_per_issuer(tickers, issuer_map):
    """*tickers* with every issuer's extra lines and OTC preferred symbols
    removed, order kept.

    For passes that run before dollar volume is known (the sector exit
    multiples): any one line stands for the issuer there, since the lines
    share its statements, so the choice uses exchange and SEC order only.
    """
    tickers = [t for t in tickers if not is_otc_preferred_symbol(t, issuer_map)]
    best = {}
    for t in tickers:
        cik = _issuer(issuer_map, t).get('cik')
        if cik and (cik not in best or
                    _listing_rank(issuer_map, t) < _listing_rank(issuer_map, best[cik])):
            best[cik] = t
    keep = set(best.values())
    return [t for t in tickers
            if not _issuer(issuer_map, t).get('cik') or t in keep]


def alias_map(rows):
    """``{folded ticker: kept ticker}`` read back from rows' ``listing_aliases``."""
    out = {}
    for r in rows or ():
        for a in r.get('listing_aliases') or ():
            out[str(a).upper()] = r.get('ticker')
    return out
