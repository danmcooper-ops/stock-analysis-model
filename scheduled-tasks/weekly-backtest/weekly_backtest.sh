#!/bin/bash
# Local weekly backtest + walk-forward calibration job
# (launchd: com.stockmodel.weekly). Runs from the canonical main checkout under
# ~/Projects (off iCloud — the old ~/Desktop copy had files evicted and its
# feature/backtesting worktree deleted, 2026-08; backtest.py now lives on main).
# output/ there is a symlink to the shared StockModelData/output (snapshots +
# price cache).
set -uo pipefail

REPO="/Users/danmcooper/Projects/Workspace Folder"
WT="$REPO"
# Same interpreter the nightly runbooks use. The repo-root .venv lacks scipy,
# which backtest.py now needs transitively via models/montecarlo.py on main.
VPY="$REPO/.claude/worktrees/phase-1-api/.venv/bin/python"

# Pin the CA bundle for all HTTPS. The python.org 3.14 build's default SSL
# trust file is a symlink that Install Certificates.command creates; a Python
# reinstall wipes it, and launchd never sources a shell profile that would
# otherwise repair the environment. SSL_CERT_FILE covers stdlib urllib (the
# Wikipedia universe fetch, FDIC/clinicaltrials/SEC-supply clients);
# REQUESTS_CA_BUNDLE covers the requests-based libs (yfinance, finnhub).
SSL_CERT_FILE="$("$VPY" -c 'import certifi; print(certifi.where())')"
export SSL_CERT_FILE
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"

LOGDIR="/Users/danmcooper/Library/Logs/StockModel"
mkdir -p "$LOGDIR"
TODAY=$(date +%F)
exec >>"$LOGDIR/weekly_$TODAY.log" 2>&1

echo "=== weekly backtest+calibrate start $(date) ==="
cd "$WT" || { echo "worktree not found"; exit 1; }
[ -e output ] || ln -s "$REPO/output" output

# 1. Refresh prices for the matured-snapshot tickers only (incremental — the
#    local parquet cache already exists, so --refresh re-fetches just the stale).
#    Clear any stale ticker list first so a crash here can't silently reuse an
#    old one; tolerate per-file read errors (iCloud-evicted snapshots have
#    failed reads with EDEADLK before) rather than aborting the whole run.
rm -f /tmp/sm_matured_tickers.txt
"$VPY" - <<'PYEOF'
import json, glob, os
from datetime import date, timedelta
today = date.today(); mat = set(); skipped = 0
for p in sorted(glob.glob('output/results_*.json')):
    d = date.fromisoformat(os.path.basename(p)[8:18])
    if d + timedelta(days=30) <= today:
        try:
            rows = json.load(open(p)).get('results', [])
        except Exception as e:
            print(f'  [warn] unreadable snapshot {p}: {e}')
            skipped += 1
            continue
        for r in rows:
            t = r.get('ticker')
            if t:
                mat.add(t)
# All four benchmark indices, not just SPY: QQQ/IWM/DIA feed the report's
# index-comparison lines and drifted 3+ weeks stale when only SPY was pinned
# here (caught 2026-08-17). The daily pipeline's Step 0 also refreshes them;
# this is the backstop for weeks where daily runs are skipped.
mat.update(('SPY', 'QQQ', 'IWM', 'DIA'))
open('/tmp/sm_matured_tickers.txt', 'w').write('\n'.join(sorted(mat)))
print(f'matured-snapshot tickers: {len(mat)}  (unreadable snapshots skipped: {skipped})')
PYEOF

TICKERS=$(cat /tmp/sm_matured_tickers.txt)
if [ -n "$TICKERS" ]; then
  "$VPY" scripts/download_prices.py --refresh --max-age-days 2 \
      --output-dir output/prices --tickers $TICKERS
fi

# 2. Backtest pipeline. annotate warms the return cache; measure + calibrate
#    refuse loudly (no file) when a horizon has no matured/usable data.
#    Track the worst stage exit code — a crashed stage must not report rc=0.
overall_rc=0
"$VPY" scripts/backtest.py annotate  --horizons 30,90,180 || overall_rc=$?
"$VPY" scripts/backtest.py measure   --horizons 30,90,180 || overall_rc=$?
# Calibrate each horizon SEPARATELY — pooling 30d and 90d returns into one
# rank-IC correlation mixes a near-zero-signal horizon into the one that
# matters and double-counts every stock. Default --max-evals enumerates the
# trimmed score-weight grid exhaustively; window spacing/overlap handling is
# inside walk_forward_calibrate (falls back to overlapped mode with an
# honest effective-N warning until enough spaced snapshots mature).
# TODAY comes from the top of the script (run-start date — midnight-safe).
"$VPY" scripts/backtest.py calibrate --horizons 30 --objective rank_ic \
    --output "output/calibration_h30_$TODAY.json" || overall_rc=$?
"$VPY" scripts/backtest.py calibrate --horizons 90 --objective rank_ic \
    --output "output/calibration_h90_$TODAY.json" || overall_rc=$?

# NOTE: capture rc BEFORE the echo — a $(date) substitution inside the echo
# line resets $? and previously masked failures as rc=0.
echo "=== weekly end $(date) rc=$overall_rc ==="
exit $overall_rc
