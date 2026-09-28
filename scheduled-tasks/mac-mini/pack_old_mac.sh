#!/bin/bash
# scheduled-tasks/mac-mini/pack_old_mac.sh — run on the OLD Mac (the MacBook Air).
#
# Finds what the old Mac holds for the stock model, says what is worth
# moving, and packs it into a transfer folder for bootstrap_mini.sh on the new
# Mac (../MAC-MINI-SETUP.md, section 0).
#
# Almost nothing needs to move. The code, every archived snapshot and the
# state files are on GitHub. The price and companyfacts caches the cloud
# Routine saved nightly since 2026-09-10 are in Supabase Storage, and they are
# newer than anything this Mac holds. What only this Mac can have:
#   - .env, the API keys (never committed);
#   - work not on GitHub: uncommitted changes, unpushed commits, stashes;
#   - portfolio edits made in the report, which live in the browser's
#     localStorage until exported (see the reminder at the end).
# So by default the pack is .env plus an inventory. The caches are opt-in and
# only worth it without the Supabase keys: a price file weeks old is
# re-downloaded in full anyway, since freshness is read from its content.
#
# Usage:
#   pack_old_mac.sh                   inventory + pack .env into ~/StockModelTransfer
#   pack_old_mac.sh --inventory-only  report only, write nothing
#   pack_old_mac.sh --with-sec-cache  also the companyfacts cache (no-Supabase fallback)
#   pack_old_mac.sh --with-prices     also the price parquets (no-Supabase fallback)
#   pack_old_mac.sh --repo PATH       a checkout the search below would miss
#   pack_old_mac.sh --out DIR         transfer folder (default ~/StockModelTransfer)
#   pack_old_mac.sh --to USER@HOST    also rsync the folder to the new Mac over SSH
#                                     (turn on Remote Login there first)
#
# It never changes the old Mac's setup. Disabling its scheduled jobs is
# printed as commands for you to run.
set -uo pipefail

OUT="$HOME/StockModelTransfer"; EXTRA_REPO=""; TO=""
INVENTORY_ONLY=0; WITH_SEC=0; WITH_PRICES=0
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --repo) EXTRA_REPO="$2"; shift 2 ;;
    --to) TO="$2"; shift 2 ;;
    --inventory-only) INVENTORY_ONLY=1; shift ;;
    --with-sec-cache) WITH_SEC=1; shift ;;
    --with-prices) WITH_PRICES=1; shift ;;
    -h|--help) sed -n '2,33p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$OUT" in
  "$HOME/Desktop"*|"$HOME/Documents"*|*"Mobile Documents"*)
    echo "refusing --out $OUT: iCloud syncs it, and the pack holds your API keys" >&2; exit 2 ;;
esac

REPORT=""
say()  { echo "$*"; REPORT="$REPORT$*
"; }
hr()   { say ""; say "== $*"; }
say_lines() {  # PREFIX TEXT: say each non-empty line of TEXT with PREFIX
  local line
  while IFS= read -r line; do
    [ -n "$line" ] && say "$1$line"
  done <<EOF
$2
EOF
}
size() { du -sh "$1" 2>/dev/null | cut -f1; }
# File mtime in epoch seconds. GNU stat first: its -f means "file system" and
# would succeed with the wrong output; BSD stat (macOS) rejects -c and falls
# through to -f %m.
mtime() {
  local m
  m="$(stat -c %Y "$1" 2>/dev/null || stat -f %m "$1" 2>/dev/null)"
  case "$m" in ''|*[!0-9]*) echo 0 ;; *) echo "$m" ;; esac
}
# Local copies use cp, not rsync: recent macOS ships openrsync, which
# implements only part of rsync's options.
copy_tree() { mkdir -p "$2" && cp -Rp "$1/." "$2/"; }

# ------------------------------------------------------------------ find checkouts
# A checkout is any directory holding scripts/analyze_stock.py. Look where the
# runbooks kept it, then search the home folder (not ~/Library) a few levels
# deep, the Trash included: a Finder delete moves rather than deletes.
CANDIDATES=""
add_candidate() {
  local d="$1"
  [ -f "$d/scripts/analyze_stock.py" ] || return 0
  case "
$CANDIDATES
" in *"
$d
"*) return 0 ;; esac
  CANDIDATES="${CANDIDATES:+$CANDIDATES
}$d"
}
[ -n "$EXTRA_REPO" ] && add_candidate "$EXTRA_REPO"
add_candidate "$HOME/Projects/Workspace Folder"
while IFS= read -r f; do
  add_candidate "$(dirname "$(dirname "$f")")"
done <<EOF
$(find "$HOME" -maxdepth 5 -path "$HOME/Library" -prune -o \
     -path '*/scripts/analyze_stock.py' -print 2>/dev/null)
EOF
DATA_DIRS="$(find "$HOME" -maxdepth 4 -path "$HOME/Library" -prune -o \
     -type d -name StockModelData -print 2>/dev/null)"

hr "Checkouts of the stock model on this Mac"
if [ -z "$CANDIDATES" ]; then
  say "none found (pass --repo PATH if it lives somewhere unusual)"
fi

BEST_ENV=""; BEST_ENV_MTIME=0
BEST_PRICES=""; BEST_PRICES_N=0
BEST_SEC=""; BEST_SEC_MARK=""
UNSAFE_WORK=0
while IFS= read -r d; do
  [ -n "$d" ] || continue
  say ""
  say "* $d  ($(size "$d"))"
  case "$d" in "$HOME/.Trash"*) say "  in the Trash — use Finder's Put Back before relying on it" ;; esac
  case "$d" in "$HOME/Desktop"*|"$HOME/Documents"*) say "  under iCloud Drive — files may be evicted placeholders" ;; esac
  if git -C "$d" rev-parse --git-dir >/dev/null 2>&1; then
    br="$(git -C "$d" rev-parse --abbrev-ref HEAD 2>/dev/null)"
    dirty="$(git -C "$d" status --porcelain 2>/dev/null | grep -vc '^?? output/' || true)"
    unpushed="$(git -C "$d" log --branches --not --remotes --oneline 2>/dev/null | wc -l | tr -d ' ')"
    stashes="$(git -C "$d" stash list 2>/dev/null | wc -l | tr -d ' ')"
    say "  git: branch $br, $dirty changed/untracked path(s), $unpushed unpushed commit(s), $stashes stash(es)"
    if [ "$dirty" != 0 ] || [ "$unpushed" != 0 ] || [ "$stashes" != 0 ]; then
      UNSAFE_WORK=1
      say "  ! work that is not on GitHub: commit and push it (or stash it into a branch) before retiring this Mac"
      say_lines "      " "$(git -C "$d" status --short 2>/dev/null | grep -v '^?? output/' | head -15)"
      say_lines "      unpushed: " "$(git -C "$d" log --branches --not --remotes --oneline 2>/dev/null | head -10)"
    fi
    say_lines "  worktree: " "$(git -C "$d" worktree list 2>/dev/null)"
  else
    say "  not a git checkout"
  fi
  if [ -f "$d/.env" ]; then
    keys="$(grep -E '^[[:space:]]*[A-Za-z_][A-Za-z0-9_]*[[:space:]]*=' "$d/.env" | sed 's/=.*//; s/[[:space:]]//g' | tr '\n' ' ')"
    m="$(mtime "$d/.env")"
    say "  .env: keys $keys(modified $(date -r "$m" +%F 2>/dev/null || echo ?))"
    if [ "$m" -gt "$BEST_ENV_MTIME" ]; then BEST_ENV="$d/.env"; BEST_ENV_MTIME="$m"; fi
  else
    say "  .env: none"
  fi
  n="$(find "$d/output/prices" -maxdepth 1 -name '*.parquet' 2>/dev/null | wc -l | tr -d ' ')"
  if [ "$n" -gt 0 ]; then
    say "  output/prices: $n parquet(s), $(size "$d/output/prices")"
    if [ "$n" -gt "$BEST_PRICES_N" ]; then BEST_PRICES="$d/output/prices"; BEST_PRICES_N="$n"; fi
  fi
  if [ -d "$d/data/cache/sec_facts" ]; then
    # SECFactsCache keeps the sweep watermark as "last_index_sweep" in _state.json.
    mark="$(grep -oE '"last_index_sweep" *: *"[0-9-]+' "$d/data/cache/sec_facts/_state.json" 2>/dev/null | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}' | head -1)"
    say "  data/cache/sec_facts: $(find "$d/data/cache/sec_facts" -name '*.json.gz' | wc -l | tr -d ' ') blob(s), $(size "$d/data/cache/sec_facts"), sweep watermark ${mark:-none}"
    if [ -z "$BEST_SEC" ] || [[ "${mark:-0}" > "${BEST_SEC_MARK:-0}" ]]; then BEST_SEC="$d/data/cache/sec_facts"; BEST_SEC_MARK="${mark:-}"; fi
  fi
  [ -f "$d/output/snapshots.duckdb" ] && say "  output/snapshots.duckdb: $(size "$d/output/snapshots.duckdb") (derived; rebuilt on the new Mac, not moved)"
  newest="$(ls "$d"/output/results_*.json* 2>/dev/null | sed 's/.*results_//; s/\.json.*//' | sort | tail -1)"
  [ -n "$newest" ] && say "  newest snapshot in output/: $newest (the archive branch has every day; not moved)"
done <<EOF
$CANDIDATES
EOF
if [ -n "$DATA_DIRS" ]; then
  say ""
  say "StockModelData folder(s) (the old shared output/): $(echo "$DATA_DIRS" | tr '\n' ' ')"
fi

# ------------------------------------------------------------------ the rest of the old setup
hr "Scheduled jobs on this Mac (disable them before the new Mac takes over)"
found=0
for l in "$HOME"/.claude/scheduled-tasks/*; do
  [ -e "$l" ] || [ -L "$l" ] || continue
  found=1
  if [ -L "$l" ]; then say "Claude scheduled task: $(basename "$l") -> $(readlink "$l")"
  else say "Claude scheduled task: $(basename "$l")"; fi
done
for p in "$HOME"/Library/LaunchAgents/com.stockmodel.*.plist; do
  [ -e "$p" ] || continue
  found=1
  loaded="not loaded"
  launchctl list 2>/dev/null | grep -q "$(basename "$p" .plist)" && loaded="LOADED"
  say "launchd agent: $p ($loaded)"
done
[ "$found" = 0 ] && say "none"

hr "Other things on this Mac"
[ -d "$HOME/.venvs/stock-model" ] && say "venv ~/.venvs/stock-model: $(size "$HOME/.venvs/stock-model") — not moved, rebuilt on the new Mac"
[ -d "$HOME/Library/Logs/StockModel" ] && say "logs ~/Library/Logs/StockModel: $(size "$HOME/Library/Logs/StockModel") — not moved (old run logs only)"
helper="$(git config --global credential.helper 2>/dev/null || true)"
say "git credential helper: ${helper:-none}. The new Mac needs its own GitHub login (bootstrap_mini.sh checks)"
ls "$HOME"/.ssh/id_* >/dev/null 2>&1 && say "ssh keys exist in ~/.ssh — not packed; give the new Mac its own key rather than copying one"
command -v gh >/dev/null && say "GitHub CLI: $(gh auth status 2>&1 | grep -m1 -oE 'Logged in to [^ ]+ (as|account) [^ ]+' || echo 'not logged in')"

# ------------------------------------------------------------------ pack
hr "Transfer pack"
if [ "$INVENTORY_ONLY" = 1 ]; then
  say "--inventory-only: nothing written"
else
  mkdir -p "$OUT" && chmod 700 "$OUT" || { echo "cannot create $OUT" >&2; exit 1; }
  if [ -n "$BEST_ENV" ]; then
    mkdir -p "$OUT/secrets" && cp -p "$BEST_ENV" "$OUT/secrets/env" && chmod 600 "$OUT/secrets/env"
    say ".env: packed from $BEST_ENV"
  else
    say ".env: NOT FOUND — recreate it on the new Mac from your password manager (RECOVERY.md step 5)"
  fi
  if [ "$WITH_SEC" = 1 ]; then
    if [ -n "$BEST_SEC" ]; then
      copy_tree "$BEST_SEC" "$OUT/sec_facts"
      say "sec_facts: packed from $BEST_SEC ($(size "$OUT/sec_facts")). Used only when the new Mac has no Supabase keys"
    else say "sec_facts: none found"; fi
  fi
  if [ "$WITH_PRICES" = 1 ]; then
    if [ -n "$BEST_PRICES" ]; then
      copy_tree "$BEST_PRICES" "$OUT/prices"
      say "prices: packed $BEST_PRICES_N file(s) from $BEST_PRICES ($(size "$OUT/prices")). Used only without Supabase keys"
    else say "prices: none found"; fi
  fi
  printf '%s' "$REPORT" > "$OUT/INVENTORY.txt"
  { echo "packed $(date '+%F %T') on $(scutil --get ComputerName 2>/dev/null || hostname)"
    ls -1 "$OUT"; } > "$OUT/MANIFEST.txt"
  say "written to $OUT ($(size "$OUT")); INVENTORY.txt is this report"
  if [ -n "$TO" ]; then
    say "copying to $TO:StockModelTransfer/ ..."
    if rsync -a "$OUT/" "$TO:StockModelTransfer/"; then
      ssh "$TO" 'chmod 700 ~/StockModelTransfer; chmod 600 ~/StockModelTransfer/secrets/env 2>/dev/null; true'
      say "copied. On the new Mac: scheduled-tasks/mac-mini/bootstrap_mini.sh --transfer ~/StockModelTransfer"
    else
      say "rsync failed: turn on Remote Login on the new Mac (System Settings > General > Sharing), then retry"
    fi
  fi
fi

# ------------------------------------------------------------------ next steps
hr "Before you retire this Mac"
[ "$UNSAFE_WORK" = 1 ] && say "! Push the unpushed work listed above first. Nothing else here is irreplaceable."
cat <<'TXT'
1. Portfolio edits made in the report live in this browser's localStorage.
   Open the report, Portfolios > Manage > Export, and bring the file along;
   on the new Mac: python scripts/portfolios.py import <file>.
2. Move the pack to the new Mac, by one of:
     - over the network: turn on Remote Login on the new Mac, then re-run
       this script with --to you@new-mac.local
     - AirDrop or a USB drive: the ~/StockModelTransfer folder
   It holds your API keys: delete it from both Macs once bootstrap_mini.sh
   has installed .env, and never put it in iCloud Drive.
3. Stop this Mac running the jobs (whenever you like; the cloud Routines are
   the live ones until the cut-over). For each item listed above:
     launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.stockmodel.weekly.plist
     rm ~/.claude/scheduled-tasks/<name>     (the link only; the files are in git)
   and delete the matching task in the Claude app's Scheduled tasks list.
TXT
