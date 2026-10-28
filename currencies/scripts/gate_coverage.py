"""Per-gate N/A coverage report.

    python scripts/gate_coverage.py output/results_<date>.json

Separates the two kinds of N/A so a sparse column can be diagnosed rather
than guessed at:

    n/a (struct)  the gate does not apply to that currency by design
    n/a (data)    the gate applies but the value was never fetched

A gate with high *structural* N/A is working as intended — the carry gates
are inapplicable to USD and SGD, and that is the point. A gate with high
*data* N/A is a sourcing problem: it is silently scoring 0 for currencies
that would otherwise rank differently.

Always exits 0. This is a diagnostic, not a gate on the pipeline — a coverage
regression should be visible without breaking the daily build.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.report_html import load_results
from scripts.scoring import GATES, _gate_applicable, _gp_key

# Above this share of applicable rows missing data, a gate is not really
# discriminating and its 0-scores are noise rather than judgement.
WARN_DATA_NA = 0.40


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('snapshot')
    args = ap.parse_args(argv)

    rows, _payload = load_results(args.snapshot)
    if not rows:
        print('No rows in %s' % args.snapshot, file=sys.stderr)
        return 0

    total = len(rows)
    print('%-28s %6s %8s %8s %8s' %
          ('GATE', 'PASS', 'FAIL', 'N/A str', 'N/A data'))
    print('-' * 62)

    warnings = []
    for g in GATES:
        pk = _gp_key(g.name)
        struct = sum(1 for r in rows if not _gate_applicable(g, r))
        passed = sum(1 for r in rows if r.get(pk) is True)
        failed = sum(1 for r in rows if r.get(pk) is False)
        applicable = total - struct
        data_na = applicable - passed - failed

        flag = ''
        if applicable and data_na / applicable > WARN_DATA_NA:
            flag = '  <-- sparse'
            warnings.append((g.name, data_na, applicable))

        print('%-28s %6d %8d %8d %8d%s'
              % (g.name, passed, failed, struct, data_na, flag))

    print('-' * 62)
    print('%d currencies, %d gates' % (total, len(GATES)))

    if warnings:
        print('\nGates missing data for more than %.0f%% of the currencies '
              'they apply to:' % (WARN_DATA_NA * 100))
        for name, na, applicable in warnings:
            print('  %-28s %d/%d' % (name, na, applicable))
        print('These score 0 for the missing rows, which drags their pillar '
              'down on\nsourcing grounds rather than on merit. Either fix the '
              'source or make the\ngate structurally inapplicable where the '
              'data genuinely does not exist.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
