#!/usr/bin/env python3
"""Copy per-ticker shard directories into the pages-live worktree, by manifest.

Four shard families ship with the report: vol/ (daily volume) and px/
(daily closes), manifested in prices_meta.json ("vol" and "manifest" keys);
hist/ (annual EDGAR fundamentals), manifested in hist_index.json
("tickers"); all three one small JSON per ticker. And details/ (the popup's
heavy text fields), ~8 MiB numbered parts manifested in details_index.json
("parts").

The repo lives under ~/Desktop, which iCloud Drive syncs. iCloud continuously
resurrects shards that were deleted locally, under space-suffixed conflict
names ("AAPL 2.json", mode 600). They reappear *after* any sweep, so no copy
that walks the source directories can be trusted to be clean.

The client only ever fetches tickers named in the manifests, so the manifest —
not the directory listing — is the source of truth. This copies exactly the
manifested shards, prunes anything else from each destination, and verifies
the result before exiting.

Exit codes: 0 = every destination matches its manifest exactly, 1 = anything
else.
"""

import json
import os
import shutil
import sys

# Repo root: this file's parent directory. Override via STOCK_MODEL_REPO
# when running against a different checkout (e.g. a worktree).
# Destination docs/ directory: override via PAGES_DOCS when the pages-live
# checkout is not the usual worktree (the cloud routine builds it in a temp
# clone). Both are read at call time.
def _paths():
    repo = os.environ.get('STOCK_MODEL_REPO') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docs = os.environ.get('PAGES_DOCS') or os.path.join(repo, ".claude", "worktrees", "pages-live", "docs")
    return repo, docs


def families(repo, docs):
    """(label, manifest file, manifest key, source dir, dest dir) per family."""
    out = os.path.join(repo, "output")
    meta = os.path.join(out, "prices_meta.json")
    return [
        ("vol", meta, "vol", os.path.join(out, "vol"), os.path.join(docs, "vol")),
        ("px", meta, "manifest", os.path.join(out, "px"), os.path.join(docs, "px")),
        ("hist", os.path.join(out, "hist_index.json"), "tickers",
         os.path.join(out, "hist"), os.path.join(docs, "hist")),
        ("details", os.path.join(out, "details_index.json"), "parts",
         os.path.join(out, "details"), os.path.join(docs, "details")),
    ]


def sync_family(label, want, src_dir, dst_dir):
    """Copy manifested shards src→dst, prune the rest, verify. True on success."""
    print(f"[{label}] manifest: {len(want)} tickers")
    os.makedirs(dst_dir, exist_ok=True)

    copied, missing = 0, []
    for ticker in want:
        src = os.path.join(src_dir, f"{ticker}.json")
        if not os.path.isfile(src):
            missing.append(ticker)
            continue
        shutil.copyfile(src, os.path.join(dst_dir, f"{ticker}.json"))
        copied += 1

    # Prune everything not on the manifest: stale shards from a prior run and
    # iCloud conflict copies alike.
    pruned = 0
    for name in os.listdir(dst_dir):
        if not name.endswith(".json") or name[:-5] not in want:
            path = os.path.join(dst_dir, name)
            if os.path.isfile(path):
                os.remove(path)
                pruned += 1

    print(f"[{label}] copied: {copied}  pruned: {pruned}")

    if missing:
        print(
            f"ERROR: [{label}] {len(missing)} manifested shard(s) absent from "
            f"{src_dir}: {', '.join(sorted(missing)[:10])}"
            f"{' ...' if len(missing) > 10 else ''}",
            file=sys.stderr,
        )
        return False

    have = {n[:-5] for n in os.listdir(dst_dir) if n.endswith(".json")}
    extra = have - want
    absent = want - have
    if extra or absent:
        print(
            f"ERROR: [{label}] destination does not match manifest "
            f"(extra={len(extra)}, missing={len(absent)})",
            file=sys.stderr,
        )
        return False

    print(f"OK: docs/{label} matches manifest exactly ({len(have)} shards)")
    return True


def main():
    repo, docs = _paths()
    ok = True
    manifests = {}
    for label, manifest, key, src_dir, dst_dir in families(repo, docs):
        if manifest not in manifests:
            try:
                with open(manifest, encoding="utf-8") as fh:
                    manifests[manifest] = json.load(fh)
            except (OSError, ValueError) as e:
                print(f"ERROR: cannot read {manifest}: {e}", file=sys.stderr)
                manifests[manifest] = {}
        want = set(manifests[manifest].get(key) or [])
        if not want:
            print(f"ERROR: no '{key}' manifest in {manifest}", file=sys.stderr)
            ok = False
            continue
        ok = sync_family(label, want, src_dir, dst_dir) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
