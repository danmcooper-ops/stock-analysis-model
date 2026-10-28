#!/usr/bin/env bash
# Build the two subprojects' own venvs, ~/.venvs/bonds-model and
# ~/.venvs/currencies-model (VENVS_DIR overrides ~/.venvs). They live outside
# the repo for the reason run_daily.sh gives for the stock model's own.
#
# They cannot share the stock model's venv. Their data/, models/ and scripts/
# have no __init__.py, so they are namespace packages, and the parent's
# editable-installed packages of the same names win over them on import.
# Each subproject therefore runs from its own folder on its own interpreter.
#
#   scripts/setup_submodels.sh            # create missing venvs, install deps
#   scripts/setup_submodels.sh --check    # exit 1 if either venv is missing
#
# Idempotent: an existing venv is only re-synced to its requirements.txt.
# run_daily.sh reads the same paths (BONDS_PYTHON / CURRENCIES_PYTHON there
# override them).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENVS="${VENVS_DIR:-$HOME/.venvs}"

# bonds/ pins Python 3.14 (bonds/CLAUDE.md); currencies/ needs 3.11+.
BONDS_PY="${BONDS_BASE_PYTHON:-python3.14}"
CURRENCIES_PY="${CURRENCIES_BASE_PYTHON:-python3}"

if [ "${1:-}" = "--check" ]; then
  rc=0
  for d in bonds currencies; do
    [ -x "$VENVS/$d-model/bin/python" ] || { echo "missing $VENVS/$d-model"; rc=1; }
  done
  exit $rc
fi

build() {  # build <dir> <python> <override variable>
  local venv="$VENVS/$1-model" py="$2"
  if [ ! -x "$venv/bin/python" ]; then
    command -v "$py" >/dev/null || { echo "$1: $py not found (set $3)" >&2; return 1; }
    echo "$1: creating $venv with $py"
    "$py" -m venv "$venv"
  fi
  "$venv/bin/python" -m pip install -q --disable-pip-version-check -r "$REPO/$1/requirements.txt"
  echo "$1: $("$venv/bin/python" --version) ready at $venv"
}

build bonds "$BONDS_PY" BONDS_BASE_PYTHON
build currencies "$CURRENCIES_PY" CURRENCIES_BASE_PYTHON
