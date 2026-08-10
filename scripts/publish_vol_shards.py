#!/usr/bin/env python3
"""Copy per-ticker volume shards into the pages-live worktree, by manifest.

The repo lives under ~/Desktop, which iCloud Drive syncs. iCloud continuously
resurrects shards that were deleted locally, under space-suffixed conflict
names ("AAPL 2.json", mode 600). They reappear *after* any sweep, so no copy
that walks output/vol/ can be trusted to be clean.

The client only ever fetches tickers named in prices.json's "vol" manifest, so
the manifest — not the directory listing — is the source of truth. This copies
exactly the manifested shards, prunes anything else from the destination, and
verifies the result before exiting.

Exit codes: 0 = destination matches manifest exactly, 1 = anything else.
"""

import json
import os
import shutil
import sys

REPO = "/Users/danmcooper/Desktop/Workspace Folder"
SRC = os.path.join(REPO, "output", "vol")
DST = os.path.join(REPO, ".claude", "worktrees", "pages-live", "docs", "vol")
PRICES = os.path.join(REPO, "output", "prices.json")


def main():
    with open(PRICES) as fh:
        manifest = json.load(fh).get("vol")

    if not manifest:
        print(f"ERROR: no 'vol' manifest in {PRICES}", file=sys.stderr)
        return 1

    want = set(manifest)
    print(f"manifest: {len(want)} tickers")

    os.makedirs(DST, exist_ok=True)

    copied, missing = 0, []
    for ticker in want:
        src = os.path.join(SRC, f"{ticker}.json")
        if not os.path.isfile(src):
            missing.append(ticker)
            continue
        shutil.copyfile(src, os.path.join(DST, f"{ticker}.json"))
        copied += 1

    # Prune everything not on the manifest: stale shards from a prior run and
    # iCloud conflict copies alike.
    pruned = 0
    for name in os.listdir(DST):
        if not name.endswith(".json") or name[:-5] not in want:
            path = os.path.join(DST, name)
            if os.path.isfile(path):
                os.remove(path)
                pruned += 1

    print(f"copied: {copied}  pruned: {pruned}")

    if missing:
        print(
            f"ERROR: {len(missing)} manifested shard(s) absent from {SRC}: "
            f"{', '.join(sorted(missing)[:10])}"
            f"{' ...' if len(missing) > 10 else ''}",
            file=sys.stderr,
        )
        return 1

    have = {n[:-5] for n in os.listdir(DST) if n.endswith(".json")}
    extra = have - want
    absent = want - have
    if extra or absent:
        print(
            f"ERROR: destination does not match manifest "
            f"(extra={len(extra)}, missing={len(absent)})",
            file=sys.stderr,
        )
        return 1

    print(f"OK: docs/vol matches manifest exactly ({len(have)} shards)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
