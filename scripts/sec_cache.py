# scripts/sec_cache.py
"""Restore and save data/cache/sec_facts/ in Supabase Storage.

data/cache/ is gitignored and the cloud container is stateless, so the
companyfacts blobs die with each run and SEC serves the whole corpus again.
See design/cache-persistence.md (Phase C).

    python scripts/sec_cache.py restore                 # before the analysis
    python scripts/sec_cache.py save                    # after it
    python scripts/sec_cache.py status

The sweep watermark (_state.json) travels with the blobs, and must: the age
backstop reads it, so blobs restored without it would be vouched for by
nothing. See data/sec_facts_cache_store.py.

Without SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY every command is a no-op
that exits 0, so a dev box behaves as it did before this cache existed.

Exit code: 0 done or skipped, 1 the store could not be read or written.
Keep the caller non-blocking: a failure here costs SEC requests, not the run.
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.sec_facts_cache import DEFAULT_CACHE_DIR, SECFactsCache  # noqa: E402
from data.sec_facts_cache_store import (DEFAULT_WORKERS, SecFactsCacheStore,  # noqa: E402
                                        StorageError)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('action', choices=['restore', 'save', 'status'])
    ap.add_argument('--cache-dir', default=DEFAULT_CACHE_DIR)
    ap.add_argument('--workers', type=int, default=DEFAULT_WORKERS)
    ap.add_argument('--all', action='store_true',
                    help='save: upload every entry, not just changed ones')
    ap.add_argument('--force', action='store_true',
                    help='save: allow a local set far smaller than what is stored')
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

    store = SecFactsCacheStore.from_env()
    if store is None:
        print('SUPABASE_URL/SUPABASE_SERVICE_ROLE_KEY not set; skipping the sec cache')
        return 0

    try:
        if a.action == 'status':
            stored = store.list_objects()
            total = sum(s for s in stored.values() if s and s > 0)
            print(f'{len(stored)} entries stored — {total / 1_048_576:.1f} MB')
            return 0

        if a.action == 'restore':
            c = store.restore(a.cache_dir, workers=a.workers)
            print(f"restored {c['restored']}/{c['stored']} entries "
                  f"({c['bytes'] / 1_048_576:.1f} MB), {c['present']} already on "
                  f"disk, {c['failed']} failed")
            _report_watermark(a.cache_dir)
            return 0

        c = store.save(a.cache_dir, workers=a.workers, force=a.force, all_files=a.all)
        print(f"uploaded {c['uploaded']} entries ({c['bytes'] / 1_048_576:.1f} MB), "
              f"{c['unchanged']} unchanged, {c['deleted']} evicted, "
              f"{c['failed']} failed ({c['local']} local, {c['stored']} were stored)")
        return 0
    except StorageError as e:
        print(f'sec cache: {e}', file=sys.stderr)
        return 1


def _report_watermark(cache_dir):
    """Say how far behind the restored watermark is.

    Worth printing rather than leaving in a debug log: a lagging sweep makes
    the whole restored cache read as missing, so this line is the difference
    between "the cache did nothing" being a mystery and being explained.
    """
    cache = SECFactsCache(cache_dir=cache_dir)
    lag = cache.sweep_lag_days()
    if lag is None:
        print('no sweep watermark restored — the age backstop falls back to file mtime')
    elif cache.sweep_is_lagging():
        print(f'WARNING: the filing sweep is {lag}d behind (backstop '
              f'{cache.max_age_days:.0f}d) — every entry will read as missing '
              f'until it catches up')
    else:
        print(f'filing sweep last walked {lag}d ago; the cache is usable')


if __name__ == '__main__':
    sys.exit(main())
