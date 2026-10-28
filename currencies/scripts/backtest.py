"""Does the composite score predict forward returns?

    python scripts/backtest.py --results-dir output --horizons 30,90,180

Reads every `results_<date>.json` snapshot, and for each one whose
`snapshot + horizon` is in the past, measures what actually happened next.

**Total return, not spot return.** The correct measure for a currency
position is the spot move *plus* the carry earned while holding it. A
high-carry currency that depreciates 3% while paying 8% annualized made
money. Scoring against spot alone would systematically condemn exactly the
positions the Carry pillar is designed to find.

Carry is accrued from the snapshot's own recorded `nominal_carry`, which is
the rate differential known at decision time — not a later revision. That
keeps the measurement free of look-ahead bias.

The quartile test is the gate-admission rule inherited from the sibling stock
model: a signal belongs in the composite only if its top quartile reliably
out-earns its bottom quartile on forward return.

Caveat printed on every run: daily snapshots overlap, so observations are
**not independent** and the spreads below are not significance-tested.
"""

import argparse
import glob
import json
import os
import statistics
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import currency_meta, ecb_client

DEFAULT_HORIZONS = (30, 90, 180)
MIN_ROWS_FOR_QUARTILES = 8

# Signals tested individually, alongside the composite.
SIGNALS = [
    ('_composite_score', 'Composite'),
    ('_score_pillar_valuation', 'Valuation pillar'),
    ('_score_pillar_carry', 'Carry pillar'),
    ('_score_pillar_external', 'External pillar'),
    ('_score_pillar_momentum', 'Momentum pillar'),
    ('_score_pillar_stability', 'Stability pillar'),
    ('reer_z20', 'REER z-score (raw)'),
    ('real_carry', 'Real carry (raw)'),
]


def load_snapshots(results_dir):
    out = []
    for path in sorted(glob.glob(os.path.join(results_dir, 'results_*.json'))):
        try:
            with open(path, encoding='utf-8') as f:
                payload = json.load(f)
            stamp = payload.get('date')
            if stamp:
                out.append((date.fromisoformat(stamp), payload, path))
        except (ValueError, OSError):
            continue
    return out


def _rate_on_or_before(series_map, target):
    """Latest rate at or before `target`. FX does not trade every calendar
    day, so an exact-date lookup would silently drop weekends and holidays."""
    best = None
    for d, v in series_map:
        if d <= target:
            best = v
        else:
            break
    return best


def total_return(spot_series, start, end, nominal_carry_pct):
    """Spot return plus accrued carry over the window, as a fraction."""
    s0 = _rate_on_or_before(spot_series, start.isoformat())
    s1 = _rate_on_or_before(spot_series, end.isoformat())
    if not s0 or not s1:
        return None
    spot_ret = s1 / s0 - 1.0
    days = (end - start).days
    carry = ((nominal_carry_pct or 0.0) / 100.0) * (days / 365.0)
    return spot_ret + carry


def quartile_spread(pairs):
    """(top-quartile mean, bottom-quartile mean, spread) over (signal, ret)."""
    if len(pairs) < MIN_ROWS_FOR_QUARTILES:
        return None
    ordered = sorted(pairs, key=lambda x: x[0])
    k = max(1, len(ordered) // 4)
    bottom = [r for _s, r in ordered[:k]]
    top = [r for _s, r in ordered[-k:]]
    return statistics.fmean(top), statistics.fmean(bottom), \
        statistics.fmean(top) - statistics.fmean(bottom)


def spearman(pairs):
    """Rank correlation, ties averaged. None for fewer than three points."""
    if len(pairs) < 3:
        return None

    def ranks(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        out = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    a = ranks([p[0] for p in pairs])
    b = ranks([p[1] for p in pairs])
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    return num / (da * db) if da and db else None


def analyze(snapshots, spot_hist, horizon):
    """Collect (signal, forward_return) pairs across every usable snapshot."""
    today = date.today()
    by_signal = {k: [] for k, _ in SIGNALS}
    by_rating = {}
    used = 0

    for snap_date, payload, _path in snapshots:
        end = snap_date + timedelta(days=horizon)
        if end > today:
            continue
        used += 1
        for r in payload.get('results', []):
            series = spot_hist.get(r.get('code'))
            if not series:
                continue
            ret = total_return(series, snap_date, end, r.get('nominal_carry'))
            if ret is None:
                continue
            for key, _label in SIGNALS:
                v = r.get(key)
                if isinstance(v, (int, float)):
                    by_signal[key].append((float(v), ret))
            by_rating.setdefault(r.get('rating', 'UNRATED'), []).append(ret)

    return by_signal, by_rating, used


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results-dir', default='output')
    ap.add_argument('--horizons', default=','.join(map(str, DEFAULT_HORIZONS)))
    args = ap.parse_args(argv)

    horizons = [int(h) for h in args.horizons.split(',') if h.strip()]
    snapshots = load_snapshots(args.results_dir)
    if not snapshots:
        print('No results_*.json snapshots in %s' % args.results_dir,
              file=sys.stderr)
        return 2

    print('Loaded %d snapshot(s): %s to %s'
          % (len(snapshots), snapshots[0][0], snapshots[-1][0]))

    per_eur = ecb_client.fetch_reference_rates()
    if not per_eur:
        print('ECB spot history unavailable; cannot measure returns.',
              file=sys.stderr)
        return 2
    spot_hist = ecb_client.build_usd_histories(per_eur, currency_meta.CODES)

    any_output = False
    for horizon in horizons:
        by_signal, by_rating, used = analyze(snapshots, spot_hist, horizon)
        print('\n' + '=' * 72)
        print('Horizon %d days — %d snapshot(s) old enough to measure'
              % (horizon, used))
        print('=' * 72)
        if not used:
            oldest = snapshots[0][0]
            need = oldest + timedelta(days=horizon)
            print('  None yet. The oldest snapshot is %s, so this horizon '
                  'becomes measurable on %s.' % (oldest, need))
            continue
        any_output = True

        print('\n  Forward total return by rating (spot + accrued carry)')
        print('  %-12s %5s %9s %9s' % ('RATING', 'N', 'MEAN', 'MEDIAN'))
        for rating in ('LONG', 'LEAN LONG', 'NEUTRAL', 'AVOID', 'UNRATED'):
            rets = by_rating.get(rating)
            if not rets:
                continue
            print('  %-12s %5d %8.2f%% %8.2f%%'
                  % (rating, len(rets), statistics.fmean(rets) * 100,
                     statistics.median(rets) * 100))

        print('\n  Quartile test — does the top quartile out-earn the bottom?')
        print('  %-22s %5s %9s %9s %9s %7s'
              % ('SIGNAL', 'N', 'TOP Q', 'BOT Q', 'SPREAD', 'RHO'))
        for key, label in SIGNALS:
            pairs = by_signal[key]
            q = quartile_spread(pairs)
            if not q:
                print('  %-22s %5d %9s' % (label, len(pairs), 'too few'))
                continue
            top, bot, spread = q
            rho = spearman(pairs)
            print('  %-22s %5d %8.2f%% %8.2f%% %8.2f%% %7s'
                  % (label, len(pairs), top * 100, bot * 100, spread * 100,
                     ('%.3f' % rho) if rho is not None else '--'))

    if any_output:
        print('\n  NOTE: snapshots overlap in time, so these observations are '
              'NOT independent.\n  Spreads are descriptive only — no '
              'significance testing is implied.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
