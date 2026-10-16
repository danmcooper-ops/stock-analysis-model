"""scheduled-tasks/weekly-backtest/env_keys.py

Print `export KEY=VALUE` lines for the named keys of a .env file, for
mac_run.sh to eval. Only the keys asked for are exported (the weekly job needs
TIINGO_API_KEY; the rest of .env holds the Supabase service key, Cloudflare and
Anthropic secrets it has no use for). A key already in the environment is left
alone, and the first occurrence of a duplicated key wins, as in
analyze_stock.py. Values are shell-quoted, so the eval cannot run anything.

Usage: python env_keys.py <path/to/.env> KEY [KEY ...]
"""
import os
import re
import shlex
import sys

_KEY = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')


def export_lines(lines, wanted, environ=os.environ):
    """`export K=V` strings for *wanted* keys found in .env *lines*."""
    wanted, seen, out = set(wanted), set(), []
    for line in lines:
        line = line.strip()
        if line.startswith('export '):
            line = line[len('export '):].strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = (part.strip() for part in line.split('=', 1))
        if len(v) >= 2 and v[0] == v[-1] and v[0] in '"\'':
            v = v[1:-1]
        if _KEY.fullmatch(k) and k in wanted and k not in environ and k not in seen:
            seen.add(k)
            out.append(f'export {k}={shlex.quote(v)}')
    return out


def main(argv):
    if len(argv) < 2:
        return 0
    with open(argv[0], encoding='utf-8') as f:
        print('\n'.join(export_lines(f, argv[1:])))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
