#!/usr/bin/env bash
# Build the two subprojects' own venvs: bonds/.venv and currencies/.venv.
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
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"

# bonds/ pins Python 3.14 (bonds/CLAUDE.md); currencies/ needs 3.11+.
BONDS_PY="${BONDS_PYTHON:-python3.14}"
CURRENCIES_PY="${CURRENCIES_PYTHON:-python3}"

if [ "${1:-}" = "--check" ]; then
  rc=0
  for d in bonds currencies; do
    [ -x "$REPO/$d/.venv/bin/python" ] || { echo "missing $d/.venv"; rc=1; }
  done
  exit $rc
fi

build() {  # build <dir> <python> <override variable>
  local dir="$REPO/$1" py="$2"
  if [ ! -x "$dir/.venv/bin/python" ]; then
    command -v "$py" >/dev/null || { echo "$1: $py not found (set $3)" >&2; return 1; }
    echo "$1: creating .venv with $py"
    "$py" -m venv "$dir/.venv"
  fi
  "$dir/.venv/bin/python" -m pip install -q --disable-pip-version-check -r "$dir/requirements.txt"
  echo "$1: $("$dir/.venv/bin/python" --version) ready"
}

build bonds "$BONDS_PY" BONDS_PYTHON
build currencies "$CURRENCIES_PY" CURRENCIES_PYTHON
