# scripts/price_cache.py
"""Restore and save output/prices/ in Supabase Storage.

The cloud container is stateless, so without this the price parquets die with
each run and step 03 re-fetches the whole universe from Yahoo every night.
See design/cache-persistence.md (Phase B).

    python scripts/price_cache.py restore                  # before step 03
    python scripts/price_cache.py save                     # after the top-up
    python scripts/price_cache.py save --all               # ignore size diffing
    python scripts/price_cache.py status                   # what is stored

Without SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY every command is a no-op
that exits 0: a developer machine and a smoke run must behave exactly as they
did before the cache existed.

Exit code: 0 done or skipped, 1 the store could not be read or written.
A restore failure is not fatal to a run — it costs the cold-cache hour, and
the caller should keep it non-blocking.
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.price_cache_store import (DEFAULT_WORKERS, PriceCacheError,  # noqa: E402
                                    PriceCacheStore)

DEFAULT_DIR = 'output/prices'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('action', choices=['restore', 'save', 'status'])
    ap.add_argument('--prices-dir', default=DEFAULT_DIR)
    ap.add_argument('--workers', type=int, default=DEFAULT_WORKERS)
    ap.add_argument('--all', action='store_true',
                    help='save: upload every parquet, not just changed ones')
    ap.add_argument('--force', action='store_true',
                    help='save: allow a local set far smaller than what is stored')
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

    store = PriceCacheStore.from_env()
    if store is None:
        print('SUPABASE_URL/SUPABASE_SERVICE_ROLE_KEY not set; skipping the price cache')
        return 0

    try:
        if a.action == 'status':
            stored = store.list_objects()
            total = sum(s for s in stored.values() if s and s > 0)
            print(f'{len(stored)} parquets stored — {total / 1_048_576:.1f} MB')
            return 0

        if a.action == 'restore':
            c = store.restore(a.prices_dir, workers=a.workers)
            print(f"restored {c['restored']}/{c['stored']} parquets "
                  f"({c['bytes'] / 1_048_576:.1f} MB), {c['present']} already on disk, "
                  f"{c['failed']} failed")
            return 0

        if not os.path.isdir(a.prices_dir):
            print(f'{a.prices_dir} does not exist; nothing to save', file=sys.stderr)
            return 1
        c = store.save(a.prices_dir, workers=a.workers, force=a.force, all_files=a.all)
        print(f"uploaded {c['uploaded']} parquets ({c['bytes'] / 1_048_576:.1f} MB), "
              f"{c['unchanged']} unchanged, {c['failed']} failed "
              f"({c['local']} local, {c['stored']} were stored)")
        return 0
    except PriceCacheError as e:
        print(f'price cache: {e}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
