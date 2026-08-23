"""scripts/gate_na_report.py

Report per-gate N/A coverage for a results snapshot.

For every model gate column, counts the records whose raw gate value
(``_gate_<name>``) is None — rendered as "N/A" in the report's Financial Model
matrix — and prints the percentage of the universe affected. When an earlier
results_*.json snapshot exists in the same directory, also prints the
day-over-day delta so a data-source degradation (like the 2026-07-22 run,
where transient fetch timeouts dropped ~100 tickers) is flagged the same day
it happens instead of being discovered in the UI.

Flags:
  ⚠ HIGH   N/A share at or above --high (default 40%). Some gates are
           structurally high (sector-inapplicable, short-history) — the flag
           marks them for a look, not automatically as a problem.
  ⚠ JUMP   N/A share rose by at least --jump points (default 10) vs the
           prior snapshot — the signal that something regressed today.

Usage:
    python scripts/gate_na_report.py output/results_2026-07-23.json
    python scripts/gate_na_report.py output/results_2026-07-23.json --high 50 --jump 5

Exit code is always 0 unless the snapshot itself can't be read — coverage
flags are informational and must not block the daily pipeline.
"""

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.scoring import gate_metadata  # noqa: E402


def _load_records(path):
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    return d['results'] if (isinstance(d, dict) and 'results' in d) else d


def _prior_snapshot(path):
    """Most recent results_*.json in the same dir dated strictly before ``path``."""
    m = re.search(r'results_(\d{4}-\d{2}-\d{2})\.json$', os.path.basename(path))
    if not m:
        return None
    cur = m.group(1)
    best = None
    for p in glob.glob(os.path.join(os.path.dirname(path) or '.', 'results_*.json')):
        pm = re.search(r'results_(\d{4}-\d{2}-\d{2})\.json$', os.path.basename(p))
        if pm and pm.group(1) < cur and (best is None or pm.group(1) > best[0]):
            best = (pm.group(1), p)
    return best


def _na_pcts(records, gates):
    """{gate key: (na_count, na_pct)} over all records."""
    n = len(records) or 1
    out = {}
    for g in gates:
        k = g['key']
        na = sum(1 for r in records if not isinstance(r, dict) or r.get(k) is None)
        out[k] = (na, 100.0 * na / n)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    ap.add_argument('snapshot', help='results_YYYY-MM-DD.json to analyze')
    ap.add_argument('--high', type=float, default=40.0,
                    help='flag gates with N/A%% at or above this (default 40)')
    ap.add_argument('--jump', type=float, default=10.0,
                    help='flag gates whose N/A%% rose by at least this many '
                         'points vs the prior snapshot (default 10)')
    args = ap.parse_args()

    records = _load_records(args.snapshot)
    gates = gate_metadata()['gates']
    cur = _na_pcts(records, gates)

    prior = _prior_snapshot(args.snapshot)
    prev = {}
    if prior:
        try:
            prev = _na_pcts(_load_records(prior[1]), gates)
        except Exception as e:  # informational only — never block on the prior
            print(f"[gate_na] prior snapshot load failed ({prior[1]}): {e}")
            prior = None

    print("=" * 74)
    print(f"  GATE N/A COVERAGE  —  {os.path.basename(args.snapshot)}"
          f"  ({len(records)} records)")
    if prior:
        print(f"  Delta vs prior snapshot: {prior[0]}")
    print("=" * 74)
    print(f"  {'Gate':<22} {'Category':<10} {'N/A':>6} {'N/A %':>7} "
          f"{'Δ pts':>7}  Flags")
    print("  " + "-" * 68)

    n_high = n_jump = 0
    for g in sorted(gates, key=lambda g: -cur[g['key']][1]):
        k = g['key']
        na, pct = cur[k]
        delta = (pct - prev[k][1]) if k in prev else None
        flags = []
        if pct >= args.high:
            flags.append('⚠ HIGH')
            n_high += 1
        if delta is not None and delta >= args.jump:
            flags.append('⚠ JUMP')
            n_jump += 1
        dtxt = f"{delta:+7.1f}" if delta is not None else '      —'
        print(f"  {g['label']:<22} {g['category']:<10} {na:>6} {pct:>6.1f}% "
              f"{dtxt}  {' '.join(flags)}")

    print("  " + "-" * 68)
    print(f"  {n_high} gate(s) ≥ {args.high:.0f}% N/A; "
          f"{n_jump} gate(s) jumped ≥ {args.jump:.0f} pts vs prior"
          + ("" if prior else " (no prior snapshot for deltas)"))
    print("=" * 74)


if __name__ == '__main__':
    main()
