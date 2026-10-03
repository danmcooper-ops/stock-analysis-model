#!/bin/bash
# scheduled-tasks/mac-mini/bootstrap_mini.sh — run on the NEW Mac (the Mac mini).
#
# Sets the new Mac up for the stock model: MAC-MINI-SETUP.md sections 2 and 3
# (with RECOVERY.md steps 1-4, 5 from the transfer pack, and 7) as one
# script. It can be re-run: each step checks what is already there and only
# does what is missing, and it never overwrites a local file with an older one.
# It schedules nothing and leaves both cloud Routines alone. The cut-over is
# MAC-MINI-SETUP.md section 7.
#
# Usage (from anywhere; it clones the repo if needed):
#   bash bootstrap_mini.sh --transfer ~/StockModelTransfer
#   bash bootstrap_mini.sh --check          report what is done and what is not; change nothing
#   bash bootstrap_mini.sh --skip-seed      set up code, venv and .env only
#
# Before the first run:
#   - a python3 >= 3.11 from python.org or Homebrew (`brew install python@3.13`),
#     since macOS's /usr/bin/python3 is too old;
#   - git access to the repo (`gh auth login`, or a token in the keychain
#     credential helper): the clone and every nightly push need it;
#   - Xcode command-line tools for git (`xcode-select --install`).
#
# Options:
#   --transfer DIR   the folder pack_old_mac.sh wrote (default ~/StockModelTransfer, if it exists)
#   --repo PATH      where the checkout lives (default "~/Projects/Workspace Folder",
#                    the path the runbooks and the weekly plist use)
#   --check          report only
#   --skip-seed      skip the caches, snapshots and store (section 3)
set -uo pipefail

REPO_URL="${REPO_URL:-https://github.com/danmcooper-ops/stock-analysis-model.git}"
REPO="$HOME/Projects/Workspace Folder"
VENV="$HOME/.venvs/stock-model"
TRANSFER=""; CHECK=0; SKIP_SEED=0
while [ $# -gt 0 ]; do
  case "$1" in
    --transfer|--repo)
      [ $# -ge 2 ] || { echo "$1 needs a path" >&2; exit 2; }
      if [ "$1" = --transfer ]; then TRANSFER="$2"; else REPO="$2"; fi; shift 2 ;;
    --check) CHECK=1; shift ;;
    --skip-seed) SKIP_SEED=1; shift ;;
    -h|--help) awk 'NR > 1 && /^#/ {print; next} NR > 1 {exit}' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -z "$TRANSFER" ] && [ -d "$HOME/StockModelTransfer" ] && TRANSFER="$HOME/StockModelTransfer"
SNAP_WT="$REPO/.claude/worktrees/snapshots-data"
PAGES_WT="$REPO/.claude/worktrees/pages-live"
SNAPSHOT_HISTORY=10

OKS=0; WARNS=0; TODO=""
ok()   { echo "  ok    $*"; OKS=$((OKS + 1)); }
warn() { echo "  WARN  $*"; WARNS=$((WARNS + 1)); TODO="$TODO
  - $*"; }
step() { echo; echo "== $*"; }
doing() { if [ "$CHECK" = 1 ]; then warn "not done: $*"; return 1; fi; echo "  ...   $*"; return 0; }

# ------------------------------------------------------------------ 1. this Mac
step "1. This Mac (MAC-MINI-SETUP.md section 1)"
if command -v sw_vers >/dev/null; then
  ok "macOS $(sw_vers -productVersion), $(uname -m)"
else
  warn "not macOS ($(uname -s)); the macOS checks below are skipped"
fi
tz="$(readlink /etc/localtime 2>/dev/null | sed 's#.*/zoneinfo/##')"
if [ "$tz" = America/New_York ]; then ok "time zone America/New_York"
else warn "time zone is ${tz:-unknown}; set it: sudo systemsetup -settimezone America/New_York"; fi
if command -v pmset >/dev/null; then
  sleep_v="$(pmset -g | awk '$1 == "sleep" {print $2; exit}')"
  ar="$(pmset -g | awk '$1 == "autorestart" {print $2; exit}')"
  if [ "$sleep_v" = 0 ] && [ "$ar" = 1 ]; then ok "never sleeps, restarts after a power failure"
  else warn "sleep=${sleep_v:-?} autorestart=${ar:-?}; run: sudo pmset -a sleep 0 disksleep 0 autorestart 1 womp 1"; fi
fi
if command -v fdesetup >/dev/null && fdesetup status 2>/dev/null | grep -q "On"; then
  warn "FileVault is on: after a power cut the Mac waits at the unlock screen and no job runs until you log in (section 1)"
fi
case "$REPO" in
  "$HOME/Desktop"*|"$HOME/Documents"*|*"Mobile Documents"*)
    echo "refusing --repo $REPO: iCloud Drive evicts and duplicates files there" >&2; exit 2 ;;
esac

# ------------------------------------------------------------------ 2. tools
step "2. Python and git"
pick_python() {
  local c
  for c in "${PYTHON3:-}" \
           /Library/Frameworks/Python.framework/Versions/Current/bin/python3 \
           /opt/homebrew/bin/python3 /usr/local/bin/python3 "$(command -v python3 2>/dev/null)"; do
    [ -n "$c" ] && [ -x "$c" ] || continue
    "$c" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null && { echo "$c"; return 0; }
  done
  return 1
}
if PY3="$(pick_python)"; then ok "python3 >= 3.11: $PY3 ($("$PY3" --version 2>&1))"
else
  warn "no python3 >= 3.11: install one from python.org or 'brew install python@3.13', then re-run"
  echo; echo "Stopping: everything below needs Python."; exit 1
fi
if ! command -v git >/dev/null || ! git --version >/dev/null 2>&1; then
  warn "git is missing: xcode-select --install, then re-run"; exit 1
fi
if git ls-remote -q "$REPO_URL" HEAD >/dev/null 2>&1; then ok "GitHub access to $REPO_URL"
else
  warn "cannot read $REPO_URL: log in (gh auth login, or a token in the keychain credential helper), then re-run"
  exit 1
fi

# ------------------------------------------------------------------ 3. checkout
step "3. Checkout and worktrees (RECOVERY.md steps 1-2)"
if [ -d "$REPO/.git" ]; then
  ok "checkout at $REPO ($(git -C "$REPO" rev-parse --abbrev-ref HEAD))"
elif doing "clone into $REPO"; then
  mkdir -p "$(dirname "$REPO")"
  # Blob-less: history is fetched on demand, so the data/snapshots branch's
  # years of daily archives cost only what a checkout of its tip needs.
  git clone -q --filter=blob:none "$REPO_URL" "$REPO" || { warn "clone failed"; exit 1; }
  ok "cloned into $REPO"
fi
add_worktree() {  # PATH BRANCH
  if [ -e "$1/.git" ]; then ok "worktree $(basename "$1") ($2)"; return 0; fi
  doing "worktree $(basename "$1") on $2" || return 0
  git -C "$REPO" fetch -q origin "$2" || { warn "cannot fetch $2"; return 1; }
  if git -C "$REPO" show-ref -q --verify "refs/heads/$2"; then
    git -C "$REPO" worktree add -q "$1" "$2"
  else
    git -C "$REPO" worktree add -q --track -b "$2" "$1" "origin/$2"
  fi && ok "worktree $(basename "$1") ($2), $(du -sh "$1" 2>/dev/null | cut -f1)"
}
if [ -d "$REPO/.git" ]; then
  add_worktree "$PAGES_WT" pages-live
  add_worktree "$SNAP_WT" data/snapshots
fi

# ------------------------------------------------------------------ 4. venv
step "4. Python environment (RECOVERY.md step 3)"
VPY="$VENV/bin/python"
if [ -x "$VPY" ] && "$VPY" -c "import yfinance, pandas, duckdb, scipy, certifi, pytest" 2>/dev/null; then
  ok "venv $VENV ($("$VPY" --version 2>&1))"
elif [ -d "$REPO/.git" ] && doing "build the venv at $VENV (a few minutes)"; then
  mkdir -p "$(dirname "$VENV")"
  [ -x "$VPY" ] || "$PY3" -m venv "$VENV" || { warn "venv creation failed"; exit 1; }
  "$VPY" -m pip install -q --upgrade pip
  "$VPY" -m pip install -q -r "$REPO/requirements.txt" 'pytest~=9.1' 'pytest-cov~=7.0' \
      'hypothesis~=6.165' 'ruff~=0.16' || { warn "pip install failed"; exit 1; }
  ok "venv $VENV"
elif [ ! -d "$REPO/.git" ]; then
  warn "venv not built (needs the checkout)"
fi

# ------------------------------------------------------------------ 5. .env
step "5. API keys (.env)"
ENV_FILE="$REPO/.env"
if [ -f "$ENV_FILE" ]; then
  ok ".env present"
elif [ -n "$TRANSFER" ] && [ -f "$TRANSFER/secrets/env" ]; then
  if doing "install .env from $TRANSFER"; then
    cp "$TRANSFER/secrets/env" "$ENV_FILE" && chmod 600 "$ENV_FILE" && ok ".env installed from the transfer pack"
  fi
else
  warn "no .env and none in a transfer pack: write it from your password manager (RECOVERY.md step 5)"
fi
if [ -f "$ENV_FILE" ]; then
  if [ "$CHECK" = 0 ]; then chmod 600 "$ENV_FILE"
  else case "$(ls -l "$ENV_FILE" | cut -c1-10)" in -rw-------) ;; *) warn "not done: chmod 600 .env" ;; esac; fi
  keys=" $(grep -E '^[[:space:]]*[A-Za-z_][A-Za-z0-9_]*[[:space:]]*=[[:space:]]*[^[:space:]]' "$ENV_FILE" | sed 's/=.*//; s/[[:space:]]//g' | tr '\n' ' ')"
  case "$keys" in *" SEC_EMAIL "*) ok "SEC_EMAIL set" ;; *) warn ".env has no SEC_EMAIL (required)" ;; esac
  missing=""
  for k in FMP_API_KEY TIINGO_API_KEY FINNHUB_API_KEY FRED_API_KEY MACRO_ANTHROPIC_API_KEY SUPABASE_URL SUPABASE_SERVICE_ROLE_KEY; do
    case "$keys" in *" $k "*) ;; *) missing="$missing $k" ;; esac
  done
  if [ -z "$missing" ]; then ok "optional keys all set"
  else echo "  info  optional keys not set:$missing (each step without its key degrades or skips)"; fi
  # Load it with analyze_stock.py's rules for the seeding below: KEY=VALUE,
  # # comments, the environment wins.
  envsh="$(mktemp "${TMPDIR:-/tmp}/bootstrap_env.XXXXXX")"
  "$PY3" - "$ENV_FILE" > "$envsh" <<'PYEOF'
import os, re, shlex, sys
for line in open(sys.argv[1], encoding='utf-8'):
    line = line.strip()
    if not line or line.startswith('#') or '=' not in line:
        continue
    k, v = line.split('=', 1)
    k, v = k.strip(), v.strip()
    if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', k) and k not in os.environ:
        print(f'export {k}={shlex.quote(v)}')
PYEOF
  . "$envsh"; rm -f "$envsh"
fi
HAS_SUPABASE=0
[ -n "${SUPABASE_URL:-}" ] && [ -n "${SUPABASE_SERVICE_ROLE_KEY:-}" ] && HAS_SUPABASE=1

# ------------------------------------------------------------------ 6. task links
step "6. Scheduled-task definitions (RECOVERY.md step 4)"
# Links only: every runbook carries a DORMANT banner and does nothing while the
# cloud Routines are live. Scheduling the tasks is a cut-over step (section 7).
for t in daily-stock-analysis publish-stock-report weekly-backtest; do
  link="$HOME/.claude/scheduled-tasks/$t"
  if [ ! -d "$REPO/scheduled-tasks/$t" ]; then warn "~/.claude/scheduled-tasks/$t not linked (needs the checkout)"
  elif [ "$(readlink "$link" 2>/dev/null)" = "$REPO/scheduled-tasks/$t" ]; then ok "~/.claude/scheduled-tasks/$t"
  elif [ -e "$link" ] && [ ! -L "$link" ]; then
    warn "~/.claude/scheduled-tasks/$t is a real folder, not a link: move it aside, then re-run"
  elif doing "link ~/.claude/scheduled-tasks/$t"; then
    mkdir -p "$HOME/.claude/scheduled-tasks" && ln -sfn "$REPO/scheduled-tasks/$t" "$link" && ok "linked $t"
  fi
done

# ------------------------------------------------------------------ 7. seed
step "7. Local state (MAC-MINI-SETUP.md section 3)"
if [ "$SKIP_SEED" = 1 ]; then
  echo "  info  --skip-seed"
elif [ ! -x "$VPY" ] || [ ! -d "$SNAP_WT" ]; then
  warn "seeding needs the venv and the data/snapshots worktree (steps 3-4)"
else
  cd "$REPO" || exit 1
  export SSL_CERT_FILE="${SSL_CERT_FILE:-$("$VPY" -m certifi)}" REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-${SSL_CERT_FILE:-}}"
  # --check changes nothing, so the folders are made only for a real run.
  [ "$CHECK" = 1 ] || mkdir -p output/prices data/cache
  count() { find "$1" -maxdepth 1 -name "$2" 2>/dev/null | wc -l | tr -d ' '; }

  # Prices. Supabase holds what the cloud saved last night; a transfer pack's
  # copy is weeks old and is re-downloaded anyway, but it at least names the
  # tickers the nightly refresh should keep current.
  n="$(count output/prices '*.parquet')"
  if [ "$n" -gt 1000 ]; then ok "output/prices: $n parquet(s)"
  elif [ "$HAS_SUPABASE" = 1 ] && doing "restore output/prices from Supabase Storage"; then
    "$VPY" scripts/price_cache.py restore --prices-dir output/prices || warn "price restore failed; re-run, or see RECOVERY.md step 6"
    ok "output/prices: $(count output/prices '*.parquet') parquet(s) after the restore"
  elif [ -n "$TRANSFER" ] && [ -d "$TRANSFER/prices" ] && doing "copy the transfer pack's prices (they are refreshed on the first run)"; then
    cp -Rpn "$TRANSFER/prices/." output/prices/ 2>/dev/null
    ok "output/prices: $(count output/prices '*.parquet') parquet(s) from the pack"
  else
    warn "output/prices has $n file(s) and there is no Supabase key: run RECOVERY.md step 6 (hours) before the first nightly run"
  fi

  # Companyfacts. Never mix sources: the watermark in _state.json vouches for
  # every blob beside it, and a restore does not overwrite local files, so
  # old blobs under a newer watermark would be served as current.
  sec_blobs() { find data/cache/sec_facts -name '*.json.gz' 2>/dev/null | wc -l | tr -d ' '; }
  if [ -f data/cache/sec_facts/_state.json ]; then ok "data/cache/sec_facts present ($(sec_blobs) blobs)"
  else
    if [ "$HAS_SUPABASE" = 1 ] && doing "restore data/cache/sec_facts from Supabase Storage"; then
      "$VPY" scripts/sec_cache.py restore || warn "sec_facts restore failed (costs one slower first run)"
      if [ -f data/cache/sec_facts/_state.json ]; then ok "sec_facts restored from Storage ($(sec_blobs) blobs)"
      else echo "  info  Storage held no companyfacts cache (no watermark)"; fi
    fi
    # The pack only when Storage gave nothing: then there is nothing to mix it with.
    if [ ! -f data/cache/sec_facts/_state.json ] && [ "$(sec_blobs)" = 0 ] \
       && [ -n "$TRANSFER" ] && [ -f "$TRANSFER/sec_facts/_state.json" ] \
       && doing "copy the transfer pack's sec_facts"; then
      rm -rf data/cache/sec_facts && mkdir -p data/cache/sec_facts \
        && cp -Rp "$TRANSFER/sec_facts/." data/cache/sec_facts/ \
        && ok "sec_facts from the pack, $(sec_blobs) blobs (the first run's filing sweep catches it up)"
    fi
    [ -f data/cache/sec_facts/_state.json ] \
      || echo "  info  no companyfacts cache: the first run fetches it from SEC (slower, harmless)"
  fi

  # The newest snapshots and the blobs they reference, for carry-forward,
  # Yesterday's Rating and gate N/A deltas.
  have="$(count output 'results_*.json*')"
  if [ "$have" -ge "$SNAPSHOT_HISTORY" ]; then ok "output/: $have snapshot(s)"
  elif doing "copy the newest $SNAPSHOT_HISTORY snapshots and their blobs into output/"; then
    ls "$SNAP_WT"/results_*.json.gz 2>/dev/null | sort | tail -"$SNAPSHOT_HISTORY" | while IFS= read -r f; do
      [ -e "output/$(basename "$f")" ] || [ -e "output/$(basename "$f" .gz)" ] || cp "$f" output/
    done
    # cp -n, not rsync: macOS ships openrsync, which lacks some options.
    [ -d "$SNAP_WT/blobs" ] && mkdir -p output/blobs && cp -Rpn "$SNAP_WT/blobs/." output/blobs/ 2>/dev/null
    ok "output/: $(count output 'results_*.json*') snapshot(s)"
  fi

  # The state files the cloud run kept on the branch. Never over a local copy:
  # once this Mac has run, its own files are the newer ones.
  for pair in rating_history.json:output portfolio_nav.json:output screen_skip.json:data/cache; do
    f="${pair%%:*}"; dest="${pair#*:}"
    if [ -f "$dest/$f" ]; then ok "$dest/$f"
    elif [ -f "$SNAP_WT/$f" ] && doing "copy $f from data/snapshots"; then
      cp "$SNAP_WT/$f" "$dest/$f" && ok "$dest/$f from data/snapshots"
    fi
  done

  # The DuckDB index over the archive (RECOVERY.md step 7).
  if [ -f output/snapshots.duckdb ]; then ok "output/snapshots.duckdb"
  elif doing "build output/snapshots.duckdb from the archive (several minutes)"; then
    # --db: the whole archive, but into output/, where the nightly readers
    # look (without it the store lands in the worktree).
    "$VPY" scripts/ingest_snapshots.py --results-dir "$SNAP_WT" --db output/snapshots.duckdb >/dev/null \
      && ok "output/snapshots.duckdb built" \
      || warn "ingest failed; readers fall back to the JSON (RECOVERY.md step 7)"
  fi
fi

# ------------------------------------------------------------------ summary
step "Summary"
echo "  $OKS ok, $WARNS to fix"
[ -n "$TODO" ] && printf '  To fix, then re-run this script:%s\n' "$TODO"
if [ "$WARNS" = 0 ] && [ "$CHECK" = 0 ]; then
  echo "  Ready for MAC-MINI-SETUP.md section 5 (scheduling) and section 7 (cut-over)."
fi
if [ -n "$TRANSFER" ] && [ -f "$TRANSFER/secrets/env" ] && [ -f "$ENV_FILE" ]; then
  echo "  The transfer pack still holds your API keys: rm -rf \"$TRANSFER\" (and on the old Mac too)."
fi
[ "$WARNS" = 0 ]
