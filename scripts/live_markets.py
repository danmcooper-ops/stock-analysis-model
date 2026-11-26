#!/usr/bin/env python3
"""Rebuild the Macro Markets board every 15 minutes in US market hours.

    .venv/bin/python scripts/live_markets.py --loop        # 9:30-16:15 ET, then exit
    .venv/bin/python scripts/live_markets.py --once        # one board now
    .venv/bin/python scripts/live_markets.py --once --dry-run   # write the file, skip upload

The nightly run publishes the board in ``macro.json`` at the close. During
the session this script rebuilds the same board (``macro_dashboard.BOARD``,
``_board_row``) from Yahoo Finance through yfinance, writes it to
``output/live/markets.json`` and uploads it to the site's ``LIVE`` KV
namespace, where the Pages Worker serves it as ``live/markets.json`` behind
the same Access login. The iOS app overlays it on the nightly board.

Rows by source:
  * Yahoo rows (equities, the dollar index, commodities): five years of
    daily closes fetched once a day, plus a five-day batch download each tick
    whose last bar is today's price so far.
  * FX rows: the FX store's history (as the nightly uses) plus today's
    ``<CCY>USD=X`` price from the same batch.
  * Rates rows: FRED publishes them once a day, so the nightly rows are
    copied unchanged.

The board leaves out each row's long ``hist`` (the app keeps the nightly
one), which keeps the file around 30 KB.

Each equity, currency and commodity row also carries ``intra``: its latest
session so far in 5-minute bars from Yahoo's chart endpoint (one call per
row), with the session's open and close times, the exchange's time zone and
the previous close. Futures and FX trade around the clock, so their session
is Yahoo's day (midnight ET for futures, midnight London for FX, drawn in
ET). Inverted FX rows (USD/JPY) use Yahoo's direct ``JPY=X`` quote, so no
precision is lost inverting a tiny number. The app draws it when a row is
opened in market hours. An intraday failure costs only that row's ``intra``.

Each tick also uploads ``prices`` (served as ``live/prices.json``): every
company in the app's universe (``app/manifest.json``) with Yahoo's latest
regular-session price and day change, from the batch quote endpoint (200
symbols a call, about 20 calls for the whole universe). The app shows them on
ticker and portfolio pages beside the close; scores and filters keep the
close. A price failure never costs the board, nor the board the prices.

Run by the Claude desktop-app task ``the-live-markets-board`` (weekdays
09:25 ET), which starts ``--loop`` detached. Needs CLOUDFLARE_API_TOKEN,
CLOUDFLARE_ACCOUNT_ID and CF_LIVE_KV_NAMESPACE (environment or .env).
"""
import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

ET = ZoneInfo('America/New_York')
OUT_DIR = REPO / 'output' / 'live'
OUT_FILE = OUT_DIR / 'markets.json'
NIGHTLY_MACRO = (REPO / '.claude' / 'worktrees' / 'pages-live' / 'docs' / 'macro.json',
                 REPO / 'output' / 'macro.json')
KV_KEY = 'markets'
PRICES_KV_KEY = 'prices'
PRICES_FILE = OUT_DIR / 'prices.json'
NIGHTLY_MANIFEST = (REPO / '.claude' / 'worktrees' / 'pages-live' / 'docs' / 'app' / 'manifest.json',
                    REPO / 'output' / 'app' / 'manifest.json')
QUOTE_URL = 'https://query1.finance.yahoo.com/v7/finance/quote'
QUOTE_CHUNK = 200
QUOTE_FIELDS = 'regularMarketPrice,regularMarketTime,regularMarketChangePercent'
KV_TTL_SECONDS = 36 * 3600          # a board left over from a dead loop expires
INTERVAL_MINUTES = 15
SESSION_START = (9, 30)
SESSION_END = (16, 15)              # one tick after the close settles the day
LIVE_PANELS = ('eq', 'fx', 'cm')    # rebuilt from Yahoo; the rest are copied
INTRADAY_PANELS = ('eq', 'fx', 'cm')  # rows that carry the session so far
CHART_URL = 'https://query1.finance.yahoo.com/v8/finance/chart/{}'
INTRADAY_INTERVAL = '5m'


def fx_symbol(ccy):
    """Yahoo's USD-per-unit symbol, the FX store's convention."""
    return f'{ccy}USD=X'


def board_metas():
    from scripts.macro_dashboard import BOARD
    return BOARD


def yahoo_symbols(board=None):
    """(Yahoo rows' symbols, {fx ccy: Yahoo symbol}) for the live panels."""
    yf_syms, fx = [], {}
    for p in board or board_metas():
        if p['k'] not in LIVE_PANELS:
            continue
        for m in p['rows']:
            if m['src'] == 'yf':
                yf_syms.append(m['sym'])
            elif m['src'] == 'fx':
                fx[m['sym']] = fx_symbol(m['sym'])
    return yf_syms, fx


def merge_closes(history, recent):
    """``history`` with ``recent``'s bars laid over it (newer data wins). Pure."""
    out = dict(history or {})
    out.update({d: v for d, v in (recent or {}).items() if v is not None and v > 0})
    return out


def _strip(row):
    return {k: v for k, v in row.items() if k != 'hist'}


def intraday_series(result):
    """One symbol's session so far from a Yahoo chart ``result``, or None. Pure.

    {tz, open, close (epoch s), prev (previous close), t (seconds after
    open), v}. Bars outside the session and empty bars are dropped."""
    try:
        meta = result.get('meta') or {}
        period = (meta.get('tradingPeriods') or [[None]])[0][0] or \
            (meta.get('currentTradingPeriod') or {}).get('regular') or {}
        start, end = int(period['start']), int(period['end'])
        ts = result.get('timestamp') or []
        closes = result['indicators']['quote'][0]['close']
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        return None
    t, v = [], []
    for stamp, c in zip(ts, closes):
        if isinstance(stamp, (int, float)) and isinstance(c, (int, float)) and c > 0 \
                and start <= stamp <= end:
            t.append(int(stamp) - start)
            v.append(float(f'{c:.7g}'))
    if len(v) < 2:
        return None
    prev = meta.get('chartPreviousClose') or meta.get('previousClose')
    return {'tz': meta.get('exchangeTimezoneName') or 'America/New_York',
            'open': start, 'close': end,
            'prev': float(f'{prev:.7g}') if isinstance(prev, (int, float)) and prev > 0 else None,
            't': t, 'v': v}


def intraday_symbol(meta):
    """The Yahoo chart symbol for a board row's session: its own symbol, or
    for FX the pair as the row reads it (``EURUSD=X``, or ``JPY=X`` for an
    inverted USD/JPY row)."""
    if meta['src'] == 'fx':
        return f'{meta["sym"]}=X' if meta.get('invert') else fx_symbol(meta['sym'])
    return meta['sym']


def intraday_symbols(board=None):
    """{row symbol: Yahoo chart symbol} for the rows that carry ``intra``."""
    return {m['sym']: intraday_symbol(m) for p in board or board_metas() if p['k'] in INTRADAY_PANELS
            for m in p['rows'] if m['src'] in ('yf', 'fx')}


def build_live_board(nightly, histories, fx_histories, today, updated, intraday=None):
    """The live board, or None when no row could be built. Pure.

    ``nightly`` is the published board (its rates panel is copied).
    ``histories`` maps a Yahoo symbol to {date: close} including today's bar;
    ``fx_histories`` maps an FX currency to {date: USD per unit};
    ``intraday`` maps a Yahoo symbol to its ``intraday_series``."""
    from scripts.macro_dashboard import _board_row
    copied = {p['k']: p for p in (nightly or {}).get('panels') or []}
    panels = []
    for p in board_metas():
        if p['k'] not in LIVE_PANELS:
            if p['k'] in copied:
                panels.append(dict(copied[p['k']],
                                   rows=[_strip(r) for r in copied[p['k']].get('rows') or []]))
            continue
        rows = []
        for meta in p['rows']:
            obs = (fx_histories if meta['src'] == 'fx' else histories).get(meta['sym']) or {}
            try:
                row = _board_row(meta, obs, today) if obs else None
            except Exception as e:       # one bad series costs one row
                print(f'  {meta["k"]} skipped ({e})')
                row = None
            if row:
                row = _strip(row)
                if p['k'] in INTRADAY_PANELS and (intraday or {}).get(meta['sym']):
                    intra = intraday[meta['sym']]
                    if meta['src'] == 'fx':      # London's day, read in New York time
                        intra = dict(intra, tz='America/New_York')
                    row['intra'] = intra
                rows.append(row)
        if rows:
            panels.append({'k': p['k'], 'l': p['l'], 'rows': rows})
    if not any(p['k'] in LIVE_PANELS for p in panels):
        return None
    return {'as_of': today.isoformat(), 'updated': updated, 'live': True,
            'night': (nightly or {}).get('as_of'), 'panels': panels}


def build_live_prices(quotes, today, updated, night):
    """The live prices file, or None when no quote is from ``today``. Pure.

    ``quotes`` is Yahoo's quote results ({symbol, regularMarketPrice,
    regularMarketTime (epoch s), regularMarketChangePercent (in percent)}).
    A quote whose last trade is not from today's session (a halted or
    delisted name, a holiday) is left out, so the app keeps the close.
    Each entry is [price, day change as a fraction, trade time in epoch s]."""
    q = {}
    for r in quotes or []:
        sym, px, t = r.get('symbol'), r.get('regularMarketPrice'), r.get('regularMarketTime')
        if not sym or not isinstance(px, (int, float)) or px <= 0 or not isinstance(t, (int, float)):
            continue
        if datetime.fromtimestamp(t, ET).date() != today:
            continue
        dp = r.get('regularMarketChangePercent')
        dp = round(dp / 100, 6) if isinstance(dp, (int, float)) else None
        q[sym] = [round(float(px), 4), dp, int(t)]
    if not q:
        return None
    return {'as_of': today.isoformat(), 'updated': updated, 'live': True, 'night': night, 'q': q}


# ---------------------------------------------------------------- I/O

def load_env():
    """Fill missing variables from the repo's .env (KEY=value lines)."""
    path = REPO / '.env'
    if not path.exists():
        return
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k.isidentifier() and k not in os.environ:
            os.environ[k] = v


def load_nightly_board():
    for path in NIGHTLY_MACRO:
        try:
            board = json.loads(path.read_text(encoding='utf-8')).get('board')
            if board:
                return board
        except (OSError, ValueError):
            continue
    return None


def history_cache(today, yf_syms):
    """Five years of daily closes per Yahoo symbol, fetched once a day."""
    from data.macro_client import fetch_yahoo_closes
    from scripts.macro_dashboard import BOARD_HISTORY_YEARS
    path = OUT_DIR / f'history-{today.isoformat()}.json'
    try:
        raw = json.loads(path.read_text(encoding='utf-8'))
        out = {s: {date.fromisoformat(d): v for d, v in obs.items()} for s, obs in raw.items()}
    except (OSError, ValueError):
        out = {}
    missing = [s for s in yf_syms if not out.get(s)]
    if not missing:
        return out
    # Symbols Yahoo returned nothing for are never cached, so a blip at the
    # open is retried next tick instead of costing the row its history all day.
    start = today - timedelta(days=int(BOARD_HISTORY_YEARS * 365.25))
    fetched = {s: fetch_yahoo_closes(s, start) for s in missing}
    out.update({s: obs for s, obs in fetched.items() if obs})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob('history-*.json'):
        if old != path:
            old.unlink(missing_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps({s: {d.isoformat(): v for d, v in obs.items()}
                               for s, obs in out.items() if obs}), encoding='utf-8')
    tmp.replace(path)
    return out


def recent_closes(symbols):
    """{symbol: {date: close}} for the last five sessions, today's so far last."""
    import yfinance as yf
    from data.macro_client import _yf_session
    df = yf.download(symbols, period='5d', interval='1d', progress=False,
                     auto_adjust=True, group_by='column', threads=True,
                     session=_yf_session())
    closes = df['Close'] if 'Close' in df else df
    out = {}
    for s in symbols:
        if s not in closes:
            continue
        col = closes[s].dropna()
        out[s] = {ts.date(): float(v) for ts, v in col.items() if v > 0}
    return out


def fx_histories(today, fx_syms, recent):
    from data.fx_store import FXStore
    from scripts.macro_dashboard import BOARD_HISTORY_YEARS, _fx_obs
    store = FXStore()
    start = today - timedelta(days=int(BOARD_HISTORY_YEARS * 365.25))
    out = {}
    for ccy, sym in fx_syms.items():
        stored = _fx_obs(store, ccy, start)
        latest = {d: v for d, v in (recent.get(sym) or {}).items() if d > max(stored, default=date.min)}
        out[ccy] = merge_closes(stored, latest)
    return out


def load_universe():
    """(tickers, night) from the published app manifest, or ([], None)."""
    for path in NIGHTLY_MANIFEST:
        try:
            m = json.loads(path.read_text(encoding='utf-8'))
            if m.get('tickers'):
                return list(m['tickers']), m.get('date')
        except (OSError, ValueError):
            continue
    return [], None


def fetch_quotes(symbols):
    """Yahoo's batch quotes for ``symbols``, QUOTE_CHUNK a call. A chunk that
    fails is skipped (its names keep the close); all failing raises."""
    from yfinance.data import YfData
    from data.macro_client import _yf_session
    data = YfData(session=_yf_session())
    out, errors = [], []
    for i in range(0, len(symbols), QUOTE_CHUNK):
        chunk = symbols[i:i + QUOTE_CHUNK]
        try:
            j = data.get_raw_json(QUOTE_URL, params={'symbols': ','.join(chunk), 'fields': QUOTE_FIELDS})
            out.extend((j.get('quoteResponse') or {}).get('result') or [])
        except Exception as e:
            errors.append(f'{type(e).__name__}: {e}')
    if errors and not out:
        raise RuntimeError(errors[0])
    return out


def fetch_intraday(symbols):
    """{row symbol: intraday_series} from Yahoo's chart endpoint, for
    ``symbols`` = {row symbol: chart symbol}; a symbol that fails is left out."""
    from yfinance.data import YfData
    from data.macro_client import _yf_session
    data = YfData(session=_yf_session())
    out = {}
    for key, sym in symbols.items():
        try:
            j = data.get_raw_json(CHART_URL.format(urllib.parse.quote(sym)),
                                  params={'range': '1d', 'interval': INTRADAY_INTERVAL})
            series = intraday_series(((j.get('chart') or {}).get('result') or [None])[0] or {})
        except Exception as e:
            print(f'  {sym} intraday skipped ({type(e).__name__}: {e})')
            continue
        if series:
            out[key] = series
    return out


def upload(body, key=KV_KEY):
    acct = os.environ.get('CLOUDFLARE_ACCOUNT_ID')
    ns = os.environ.get('CF_LIVE_KV_NAMESPACE')
    token = os.environ.get('CLOUDFLARE_API_TOKEN')
    if not (acct and ns and token):
        raise RuntimeError('CLOUDFLARE_ACCOUNT_ID, CF_LIVE_KV_NAMESPACE and CLOUDFLARE_API_TOKEN are needed to upload')
    url = (f'https://api.cloudflare.com/client/v4/accounts/{acct}/storage/kv/namespaces/'
           f'{ns}/values/{key}?expiration_ttl={KV_TTL_SECONDS}')
    req = urllib.request.Request(url, data=body.encode(), method='PUT', headers={
        'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=30) as resp:
        if not json.loads(resp.read()).get('success'):
            raise RuntimeError('KV write refused')


def write_atomic(path, body):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(body, encoding='utf-8')
    tmp.replace(path)


def prices_tick(dry_run=False):
    """Build, write and (unless dry_run) upload the live prices. True on success."""
    now = datetime.now(ET)
    tickers, night = load_universe()
    if not tickers:
        print(f'{now:%H:%M} prices failed: no app manifest')
        return False
    try:
        quotes = fetch_quotes(tickers)
    except Exception as e:
        print(f'{now:%H:%M} prices failed: Yahoo quotes: {e}')
        return False
    prices = build_live_prices(quotes, now.date(), datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), night)
    if not prices:
        print(f'{now:%H:%M} prices: no quote from today ({len(quotes)} returned)')
        return False
    body = json.dumps(prices, separators=(',', ':'))
    write_atomic(PRICES_FILE, body)
    n = len(prices['q'])
    if dry_run:
        print(f'{now:%H:%M} prices: wrote {PRICES_FILE} ({len(body)} bytes, {n}/{len(tickers)}); upload skipped')
        return True
    try:
        upload(body, PRICES_KV_KEY)
    except Exception as e:
        print(f'{now:%H:%M} prices upload failed: {e}')
        return False
    print(f'{now:%H:%M} prices: uploaded {len(body)} bytes, {n}/{len(tickers)} tickers')
    return True


def tick(dry_run=False):
    """The board and the prices; True when the board succeeded."""
    try:
        prices_tick(dry_run)
    except Exception as e:               # the prices never cost the board
        print(f'{datetime.now(ET):%H:%M} prices failed: {type(e).__name__}: {e}', flush=True)
    return board_tick(dry_run)


def board_tick(dry_run=False):
    """Build, write and (unless dry_run) upload one board. True on success."""
    now = datetime.now(ET)
    today = now.date()
    yf_syms, fx_syms = yahoo_symbols()
    nightly = load_nightly_board()
    try:
        recent = recent_closes(yf_syms + list(fx_syms.values()))
    except Exception as e:
        print(f'{now:%H:%M} Yahoo download failed: {e}')
        return False
    hist = history_cache(today, yf_syms)
    histories = {s: merge_closes(hist.get(s), recent.get(s)) for s in yf_syms}
    try:
        intraday = fetch_intraday(intraday_symbols())
    except Exception as e:               # the session chart never costs the board
        print(f'{now:%H:%M} intraday failed: {type(e).__name__}: {e}')
        intraday = {}
    board = build_live_board(nightly, histories, fx_histories(today, fx_syms, recent), today,
                             datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), intraday)
    if not board:
        print(f'{now:%H:%M} no rows built')
        return False
    body = json.dumps(board, separators=(',', ':'))
    write_atomic(OUT_FILE, body)
    n = sum(len(p['rows']) for p in board['panels'])
    if dry_run:
        print(f'{now:%H:%M} wrote {OUT_FILE} ({len(body)} bytes, {n} rows); upload skipped')
        return True
    try:
        upload(body)
    except Exception as e:
        print(f'{now:%H:%M} upload failed: {e}')
        return False
    print(f'{now:%H:%M} uploaded {len(body)} bytes, {n} rows')
    return True


def session_ticks(day):
    """Every 15-minute mark from the open to SESSION_END, in ET."""
    t = datetime(day.year, day.month, day.day, *SESSION_START, tzinfo=ET)
    end = datetime(day.year, day.month, day.day, *SESSION_END, tzinfo=ET)
    out = []
    while t <= end:
        out.append(t)
        t += timedelta(minutes=INTERVAL_MINUTES)
    return out


def loop(dry_run=False):
    from scripts.market_open import market_status
    today = datetime.now(ET).date()
    is_open, reason = market_status(today)
    if not is_open:
        print(reason)
        return 0
    ok = fails = 0
    for t in session_ticks(today):
        wait = (t - datetime.now(ET)).total_seconds()
        if wait < -INTERVAL_MINUTES * 60 + 5 and t != session_ticks(today)[-1]:
            continue                     # started late: skip marks long gone
        if wait > 0:
            time.sleep(wait)
        try:
            done = tick(dry_run)
        except Exception as e:           # one bad tick never ends the day's loop
            print(f'{datetime.now(ET):%H:%M} tick failed: {type(e).__name__}: {e}', flush=True)
            done = False
        if done:
            ok += 1
        else:
            fails += 1
    print(f'RESULT ok={ok} failed={fails}')
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--once', action='store_true', help='build one board now')
    g.add_argument('--loop', action='store_true', help='every 15 minutes until 16:15 ET')
    ap.add_argument('--dry-run', action='store_true', help='write the file, skip the upload')
    a = ap.parse_args(argv)
    load_env()
    if a.once:
        return 0 if tick(a.dry_run) else 1
    return loop(a.dry_run)


if __name__ == '__main__':
    sys.exit(main())
