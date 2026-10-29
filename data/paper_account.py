"""The paper account's own records (``portfolio/paper/``, gitignored, 0600).

``scripts/paper_trade.py`` writes them; ``report_html`` reads them to build
the public Paper account dashboard (``models/paper_dashboard.py``). Kept
here rather than in the script so the render can read them without the
script's import-time ``.env`` load.

* ``ledger.jsonl``: one JSON record per rebalance (targets, orders, halts)
  and per reconcile (fills, equity, positions).
* ``nav.json``: one entry per day, ``{d, equity, last_equity}``.
* ``STOP``: the kill switch.

``PAPER_DIR`` overrides the directory.
"""
from __future__ import annotations

import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def paper_dir():
    return os.environ.get('PAPER_DIR', os.path.join(REPO, 'portfolio', 'paper'))


def stop_path():
    return os.path.join(paper_dir(), 'STOP')


def ledger_path():
    return os.path.join(paper_dir(), 'ledger.jsonl')


def nav_path():
    return os.path.join(paper_dir(), 'nav.json')


def _private_dir():
    os.makedirs(paper_dir(), mode=0o700, exist_ok=True)


def append_ledger(record):
    _private_dir()
    fd = os.open(ledger_path(), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, separators=(',', ':'), default=str) + '\n')


def read_ledger():
    try:
        with open(ledger_path(), encoding='utf-8') as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []


def load_nav():
    try:
        with open(nav_path(), encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def save_nav(series):
    _private_dir()
    tmp = nav_path() + '.tmp'
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        json.dump(series, f, indent=1)
    os.replace(tmp, nav_path())
