# data/coverage.py
"""Snapshot coverage floor: did tonight's run keep enough of the universe?

A run with far fewer rows than the previous complete run is far more likely
a broken night than a real change in the universe. On 2026-09-30 Yahoo
rate-limited the host and the snapshot landed with 739 rows against 2,514
the day before. The database publish refused it (``MIN_ROW_RATIO`` here was
its guard, in data/db/publish.py), which is why Supabase stayed clean — but
nothing between Phase 1 and the live site asked the same question, so the
degraded snapshot was archived and force-pushed over a good report, and the
run-quality summary's "rows without a price" check could not fire because
the missing rows were absent rather than dataless.

``coverage_check`` is that question, asked once per run and stamped into the
snapshot's ``provenance.coverage``; ``scripts/check_coverage.py`` reads it
back for run.sh, which archives a degraded snapshot (the record is kept, a
re-run of the same date supersedes it) but does not publish it.
"""

# A run with fewer rows than this share of the previous complete run is
# refused by the database publish without --force, and now also keeps the
# live report at the last good run: more likely a broken night than a real
# change.
MIN_ROW_RATIO = 0.7


def coverage_check(rows, prior_rows, prior_date=None, min_ratio=MIN_ROW_RATIO,
                   applicable=True):
    """Compare tonight's row count with the prior snapshot's.

    Returns a JSON-safe dict: ``rows``, ``prior_rows``, ``prior_date``,
    ``ratio`` (None without a prior), ``min_ratio``, ``degraded`` and a
    ``note`` saying why the floor did not apply, when it did not. A run
    with no prior snapshot, or one on an explicit ticker list
    (*applicable* False), is never degraded: there is nothing fair to
    compare it against.
    """
    rows = int(rows or 0)
    prior_rows = int(prior_rows or 0)
    out = {'rows': rows, 'prior_rows': prior_rows, 'prior_date': prior_date,
           'ratio': None, 'min_ratio': float(min_ratio), 'degraded': False,
           'note': None}
    if not applicable:
        out['note'] = 'floor not applied: explicit ticker list'
        return out
    if prior_rows <= 0:
        out['note'] = 'floor not applied: no prior snapshot'
        return out
    out['ratio'] = round(rows / prior_rows, 4)
    out['degraded'] = out['ratio'] < min_ratio
    return out


def format_coverage(cov):
    """One line for logs and status.txt: ``COVERAGE ok|degraded|unknown …``."""
    if not cov:
        return 'COVERAGE unknown (no provenance.coverage block)'
    state = 'degraded' if cov.get('degraded') else 'ok'
    ratio = cov.get('ratio')
    ratio_s = f"{ratio:.3f}" if isinstance(ratio, (int, float)) else 'n/a'
    prior = cov.get('prior_date') or 'none'
    line = (f"COVERAGE {state} rows={cov.get('rows')} prior_rows={cov.get('prior_rows')} "
            f"({prior}) ratio={ratio_s} min={cov.get('min_ratio', MIN_ROW_RATIO):.2f}")
    if cov.get('note'):
        line += f" [{cov['note']}]"
    return line


def read_coverage(path):
    """The ``provenance.coverage`` block of a results snapshot, or None when
    the file predates the stamp."""
    from data.snapshot_store import read_snapshot, split_snapshot
    meta, _rows = split_snapshot(read_snapshot(path))
    prov = meta.get('provenance') or {}
    cov = prov.get('coverage')
    return dict(cov) if isinstance(cov, dict) else None
