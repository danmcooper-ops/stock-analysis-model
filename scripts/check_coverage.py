#!/usr/bin/env python3
"""Read a snapshot's provenance.coverage block and say whether the run kept
enough of the universe to publish (data/coverage.py).

    python scripts/check_coverage.py output/results_2026-09-30.json

Prints one ``COVERAGE …`` line (run.sh copies it into status.txt) and exits
0 when the floor is met or does not apply, 4 when the run is degraded, and
1 when the file cannot be read. A snapshot from before the stamp existed
reads as ``unknown`` and exits 0.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.coverage import format_coverage, read_coverage  # noqa: E402

EXIT_DEGRADED = 4


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('results', help='results_<date>.json (or .json.gz)')
    args = ap.parse_args(argv)
    try:
        cov = read_coverage(args.results)
    except Exception as e:
        print(f'COVERAGE unreadable: {args.results}: {e}')
        return 1
    print(format_coverage(cov))
    return EXIT_DEGRADED if cov and cov.get('degraded') else 0


if __name__ == '__main__':
    sys.exit(main())
