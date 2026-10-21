"""ETF look-through: an ETF read through the company rows the model scores.

An ETF has no statements, so none of DCF/EPV/RIM/DDM applies to it. Its
holdings do: a fund holding weight ``w_i`` of company *i*, whose fair value is
``FV_i`` at price ``P_i``, is worth ``V = Σ w_i · FV_i / P_i`` per dollar on the
model's numbers, and ``1 − 1/V`` is its margin of safety under the same
definition as a row's (``scoring.py``: ``(FV − P) / FV``). Averaging the
holdings' MoS instead would be wrong — MoS is concave in FV/P.

The absolute figure mostly restates the model's market-wide bias: on
2026-10-20 SPY read V ≈ 0.77, a −30% MoS, and every broad fund read about as
"overvalued". What the model can say about one fund is *relative*, so the
headline is ``V / V_SPY − 1`` (``relative_to``), with the absolute MoS shown
second. Never feed an ETF into profile_verdict: its −20% MoS AVOID flag would
fire on nearly every fund.

Rules, each forced by the 2026-10-06..20 spike over the 14 SPDRs:
- FV/P is clipped to ``[1/3, 3]`` (symmetric in log, so the clip does not
  itself pull V down); the clipped weight is reported. REITs bind the low
  clip often (VTR read FV/P 0.09), so XLRE's read is largely clipped.
- A holding with no fair value today takes its last fair value from the
  prior few snapshots, divided by today's price (``carried``). On 10-14
  EQIX, VTR and ESS (13% of XLRE) lost theirs for two days, and counting
  them at FV/P = 1 swung XLRE 13pp and back.
- Still missing after that, a holding counts at FV/P = 1 in the base value
  and at the low clip in ``V_low``: missing fair values are not random
  (loss-makers like MRNA, APP, BA), so the honest answer is a band.
- Cash and money-market lines count at FV/P = 1. A holding with no row in
  the snapshot (below the mcap floor, skipped, stopped trading) is left out
  of the denominator and reported as ``unmatched``.
- Below ``min_coverage`` of weight valued (today's or carried, plus cash) the
  fund reads INSUFFICIENT and carries no value.

Weights are the file's, drifted to the run date by each holding's adjusted
price return when the file is older than the run (``drift``) — never shares ×
price, because the parquets are split-adjusted and the file's share counts
are not.
"""
import math

ETF_METHOD = 1
FVP_CLIP = (1.0 / 3.0, 3.0)
MIN_COVERAGE = 0.80
TOP_HOLDINGS = 10
TOP_CONTRIBUTORS = 3


def _pos(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if v > 0 and math.isfinite(v) else None


def lookthrough(holdings, rows_by_ticker, aliases=None, prev_fv=None, drift=None,
                clip=FVP_CLIP, min_coverage=MIN_COVERAGE):
    """Read one fund through the snapshot's rows.

    *holdings* is ``data.etf_holdings.parse_ssga_xlsx``'s output;
    *rows_by_ticker* maps ticker -> row (``price``, ``_fv_effective``,
    ``rating``, ``sector``, ``company_name``); *aliases* maps a folded listing
    to its kept row (GOOG -> GOOGL); *prev_fv* maps ticker -> its last known
    ``_fv_effective``; *drift* maps ticker -> ``Padj(run) / Padj(as_of)``.
    """
    aliases = aliases or {}
    prev_fv = prev_fv or {}
    drift = drift or {}
    lo, hi = clip
    items = []
    for h in holdings.get('holdings') or []:
        w = h.get('weight')
        if not isinstance(w, (int, float)) or not w > 0:
            continue
        tk = aliases.get(h.get('ticker') or '', h.get('ticker') or '')
        it = {'ticker': tk, 'name': h.get('name'), 'w': float(w), 'x': None,
              'state': None, 'row': None}
        if h.get('kind') == 'cash':
            it.update(x=1.0, state='cash')
        else:
            row = rows_by_ticker.get(tk) if tk else None
            if row is None:
                it['state'] = 'unmatched'
            else:
                it['row'] = row
                it['w'] *= drift.get(tk) or 1.0
                p, fv = _pos(row.get('price')), _pos(row.get('_fv_effective'))
                pfv = _pos(prev_fv.get(tk))
                if p and fv:
                    it.update(x=fv / p, state='valued')
                elif p and pfv:
                    it.update(x=pfv / p, state='carried')
                else:
                    it['state'] = 'missing'
        items.append(it)
    # Share classes held as separate lines (GOOG + GOOGL) are one company:
    # sum them onto the kept row, so it is listed and ranked once.
    merged, seen = [], {}
    for it in items:
        if it['row'] is None:
            merged.append(it)
        elif it['ticker'] in seen:
            seen[it['ticker']]['w'] += it['w']
        else:
            seen[it['ticker']] = it
            merged.append(it)
    items = merged

    total = sum(i['w'] for i in items)
    if not total:
        return {'fund': holdings.get('fund'), 'status': 'insufficient',
                'reason': 'no holdings', 'method': ETF_METHOD}
    share = {s: sum(i['w'] for i in items if i['state'] == s) / total
             for s in ('valued', 'carried', 'missing', 'cash', 'unmatched')}
    in_scope = [i for i in items if i['state'] != 'unmatched']
    den = sum(i['w'] for i in in_scope)
    valued_w = share['valued'] + share['carried'] + share['cash']
    out = {
        'fund': holdings.get('fund'), 'fund_name': holdings.get('fund_name'),
        'as_of': holdings.get('as_of'), 'source': holdings.get('source'),
        'n_holdings': len(items), 'method': ETF_METHOD,
        'valued_w': valued_w, 'carried_w': share['carried'], 'missing_w': share['missing'],
        'cash_w': share['cash'], 'unmatched_w': share['unmatched'],
        'matched_w': 1.0 - share['unmatched'],
    }

    def _clip(x):
        return min(max(x, lo), hi)

    v = v_low = clipped = 0.0
    for i in in_scope:
        if i['x'] is None:
            v += i['w'] * 1.0
            v_low += i['w'] * lo
        else:
            c = _clip(i['x'])
            if c != i['x']:
                clipped += i['w']
            v += i['w'] * c
            v_low += i['w'] * c
    v, v_low = v / den, v_low / den
    out['clipped_w'] = clipped / total

    # Rating mix and sector weights over the matched companies (cash and
    # unmatched lines excluded), as shares of that matched weight.
    cos = [i for i in in_scope if i['row'] is not None]
    cw = sum(i['w'] for i in cos) or 1.0
    mix, sectors = {}, {}
    for i in cos:
        r = i['row'].get('rating') or 'N/A'
        mix[r] = mix.get(r, 0.0) + i['w'] / cw
        s = i['row'].get('sector') or 'Other'
        sectors[s] = sectors.get(s, 0.0) + i['w'] / cw
    out['rating_mix'] = mix
    out['sectors'] = dict(sorted(sectors.items(), key=lambda kv: -kv[1]))

    def _entry(i, contrib=None):
        e = {'t': i['ticker'], 'n': (i['row'] or {}).get('company_name') or i['name'],
             'w': i['w'] / total, 'fvp': i['x'], 'st': i['state'],
             'r': (i['row'] or {}).get('rating')}
        if contrib is not None:
            e['c'] = contrib
        return e

    out['top'] = [_entry(i) for i in sorted(items, key=lambda i: -i['w'])[:TOP_HOLDINGS]]
    # What moves V away from 1: each holding's pull, w * (clipped FV/P - V).
    pulls = [(i, i['w'] / den * (_clip(i['x']) - v)) for i in cos if i['x'] is not None]
    pulls.sort(key=lambda t: t[1])
    out['drags'] = [_entry(i, c) for i, c in pulls[:TOP_CONTRIBUTORS] if c < 0]
    out['lifts'] = [_entry(i, c) for i, c in reversed(pulls[-TOP_CONTRIBUTORS:]) if c > 0]

    if valued_w < min_coverage:
        out.update(status='insufficient',
                   reason=f'{valued_w:.0%} of weight has a fair value (needs {min_coverage:.0%})')
        return out
    out.update(status='ok', V=v, V_low=v_low, mos=1.0 - 1.0 / v, mos_low=1.0 - 1.0 / v_low)
    return out


def relative_to(results, base='SPY'):
    """Stamp each ok fund's value relative to *base*'s: ``rel`` from the base
    values, and ``rel_lo``/``rel_hi`` the band from pairing the base values
    and the missing-at-low-clip values. A no-op when *base* is unusable."""
    b = results.get(base)
    if not b or b.get('status') != 'ok':
        return results
    for r in results.values():
        if r.get('status') != 'ok':
            continue
        pair = (r['V'] / b['V'] - 1.0, r['V_low'] / b['V_low'] - 1.0)
        r['rel'] = pair[0]
        r['rel_lo'], r['rel_hi'] = min(pair), max(pair)
        r['rel_base'] = base
    return results
