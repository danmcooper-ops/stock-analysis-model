#!/usr/bin/env python3
"""Serve the published report with live intraday quotes on the home network.

    .venv/bin/python scripts/live_quotes.py            # http://<mini>.local:8765/

Runs on the Mac mini (launchd: scheduled-tasks/mac-mini/
com.stockmodel.livequotes.plist). It serves immutable copies of the published
``docs/`` folder and, from the same origin, ``api/quote?symbols=A,B`` — the
browser blocks an HTTPS page (Cloudflare, GitHub Pages) from calling a
plain-HTTP LAN address, so the report and its quotes must share one origin.
On the hosted sites the page's first quote request fails and it stays on the
nightly values.

Quotes come from Yahoo's pricing websocket, the stream yfinance's
``yf.AsyncWebSocket`` reads, but the connection loop is our own: yfinance
1.x never resets its socket after a drop (``_connect`` only connects when
``_ws is None``), so its ``listen()`` retries a dead socket every 3 s forever
and spins on a clean close, and its heartbeat task dies on the first send
error, leaving a reconnected socket with no subscriptions. The URL and the
``PricingData`` protobuf are still yfinance's.

The stream is one connection with no REST calls, open only 9:30-16:05 ET on
weekdays and never while ``output/.run_daily.lock`` names a live process, so
it stays clear of the nightly run's Yahoo traffic. ``LIVE_QUOTES=0`` keeps
the report served with streaming off.
"""
import argparse
import asyncio
import base64
import inspect
import json
import logging
import os
import re
import shutil
import socket
import sys
import threading
import time
from collections import OrderedDict
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts import config  # noqa: E402

logger = logging.getLogger(__name__)

ET = ZoneInfo('America/New_York')
CANARY = 'SPY'
SYMBOL_RE = re.compile(r'^[A-Z0-9.\-^=]{1,12}$')
# PricingData.market_hours: 0 pre-market (omitted on the wire), 1 regular.
REGULAR_MARKET = 1
DEFAULT_DOCS = REPO / '.claude' / 'worktrees' / 'pages-live' / 'docs'
DEFAULT_BUILDS = REPO / 'output' / 'live_quotes_builds'
RUN_LOCK = REPO / 'output' / '.run_daily.lock'
API_RE = re.compile(r'(?:^|/)api/quote(/health)?$')
BUILD_PATH_RE = re.compile(r'^/b/(\d+)/(.*)$')


def stream_url():
    """yfinance's streamer URL, read from its own default so a change there
    reaches us without a copy here going stale."""
    from yfinance.live import AsyncWebSocket
    return inspect.signature(AsyncWebSocket.__init__).parameters['url'].default


def decode_frame(raw):
    """One websocket text frame -> PricingData as a dict, or None."""
    from google.protobuf.json_format import MessageToDict
    from yfinance.pricing_pb2 import PricingData
    try:
        payload = json.loads(raw).get('message', '')
        msg = PricingData()
        msg.ParseFromString(base64.b64decode(payload))
        return MessageToDict(msg, preserving_proto_field_name=True) or None
    except Exception as e:
        logger.debug('undecodable frame: %s', e)
        return None


def _round_px(v):
    # PricingData.price is a float32; round off its noise.
    return round(v, 4 if abs(v) < 1 else 2)


def parse_tick(d):
    """PricingData dict -> (symbol, quote) or None.

    MessageToDict drops fields at their default value (change 0, market_hours
    0 = pre-market) and renders sint64 as strings, so every field is defaulted
    and cast. ``dp`` is a fraction, as the report's percentages are; the wire
    sends percent.
    """
    if not d:
        return None
    sym = d.get('id')
    try:
        price = float(d.get('price') or 0)
        t = int(d.get('time') or 0)
    except (TypeError, ValueError):
        return None
    if not sym or price <= 0 or t <= 0:
        return None
    pc = float(d.get('previous_close') or 0)
    return sym, {
        'c': _round_px(price),
        'd': round(float(d.get('change') or 0), 4),
        'dp': round(float(d.get('change_percent') or 0) / 100, 6),
        't': t,
        'pc': _round_px(pc) if pc > 0 else None,
        'mh': int(d.get('market_hours') or 0),
    }


def in_window(now_et):
    """Weekday between LIVE_QUOTES_OPEN_ET and LIVE_QUOTES_CLOSE_ET."""
    if now_et.weekday() >= 5:
        return False
    hm = (now_et.hour, now_et.minute)
    return config.LIVE_QUOTES_OPEN_ET <= hm < config.LIVE_QUOTES_CLOSE_ET


def run_lock_active(lock_path=RUN_LOCK):
    """True while the nightly run's lock names a live process."""
    try:
        pid = int(Path(lock_path).read_text(encoding='utf-8').strip())
    except (OSError, ValueError):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _et_date(ms):
    return datetime.fromtimestamp(ms / 1000, ET).date()


class QuoteBook:
    """Quotes and subscriptions, shared by the HTTP threads and the stream.

    HTTP threads only record demand here; the stream thread reads
    ``wanted()`` and owns the socket.
    """

    def __init__(self, pinned=(), max_symbols=None, idle_sec=None):
        self._lock = threading.Lock()
        self._quotes = {}
        self._requested = OrderedDict()   # symbol -> last request time, LRU first
        self._pinned = {CANARY, *pinned}
        self._max = max_symbols or config.LIVE_QUOTES_MAX_SYMBOLS
        self._idle = idle_sec or config.LIVE_QUOTES_IDLE_SEC
        self._session_day = None          # ET date the canary traded regular hours
        self.ticks = 0

    def request(self, symbols, now):
        with self._lock:
            for s in symbols:
                self._requested.pop(s, None)
                self._requested[s] = now
            room = self._max - len(self._pinned)
            while len(self._requested) > max(room, 0):
                self._requested.popitem(last=False)

    def expire(self, now):
        with self._lock:
            idle = [s for s, t in self._requested.items() if now - t > self._idle]
            for s in idle:
                del self._requested[s]
            return idle

    def wanted(self):
        with self._lock:
            return frozenset(self._pinned | set(self._requested))

    def update(self, sym, quote):
        with self._lock:
            self._quotes[sym] = quote
            self.ticks += 1
            if sym == CANARY and quote['mh'] == REGULAR_MARKET:
                self._session_day = _et_date(quote['t'])

    def get(self, symbols, today):
        """Quotes from today's (ET) session only: before the open, yesterday's
        last trade is the nightly close the page already shows."""
        with self._lock:
            out = {}
            for s in symbols:
                q = self._quotes.get(s)
                out[s] = ({k: q[k] for k in ('c', 'd', 'dp', 't')}
                          if q and _et_date(q['t']) == today else None)
            return out

    def session_open(self, today):
        with self._lock:
            return self._session_day == today

    def stats(self):
        with self._lock:
            last = max((q['t'] for q in self._quotes.values()), default=None)
            return {'subscribed': len(self._pinned | set(self._requested)),
                    'quotes': len(self._quotes), 'ticks': self.ticks,
                    'last_tick_ms': last}


def stall_delay(strikes):
    """Wait before the reconnect that follows the n-th consecutive stall."""
    return min(config.LIVE_QUOTES_STALL_SEC * 2 ** max(strikes - 1, 0),
               config.LIVE_QUOTES_STALL_DELAY_MAX)


class Streamer:
    """Owns the websocket: connect, subscribe, read, reconnect.

    Every (re)connect is a fresh socket that is sent the full subscription
    set, so nothing depends on state surviving a drop. ``gate()`` says
    whether to be connected at all (window, run lock, kill switch).
    """

    def __init__(self, book, gate, url=None, connect=None,
                 stall_sec=None, heartbeat_sec=None, gate_poll_sec=30,
                 error_delay=None, error_delay_max=None, refused_wait=None,
                 stall_delay_fn=stall_delay, sleep=asyncio.sleep):
        self.book = book
        self.gate = gate
        self.url = url
        self._connect = connect
        self.stall_sec = stall_sec or config.LIVE_QUOTES_STALL_SEC
        self.heartbeat_sec = heartbeat_sec or config.LIVE_QUOTES_HEARTBEAT_SEC
        self.gate_poll_sec = gate_poll_sec
        self.error_delay = error_delay or config.LIVE_QUOTES_ERROR_DELAY
        self.error_delay_max = error_delay_max or config.LIVE_QUOTES_ERROR_DELAY_MAX
        self.refused_wait = refused_wait or config.LIVE_QUOTES_REFUSED_WAIT
        self._stall_delay = stall_delay_fn
        self._sleep = sleep
        self._stop = asyncio.Event()
        self.counters = {'connects': 0, 'stalls': 0, 'errors': 0, 'refused': 0}

    def stop(self):
        self._stop.set()

    async def _wait(self, sec):
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=sec)
        except TimeoutError:
            pass

    async def run(self):
        from websockets.exceptions import InvalidStatus
        if self.url is None:
            self.url = stream_url()
        if self._connect is None:
            from websockets.asyncio.client import connect
            self._connect = connect
        err_delay = self.error_delay
        strikes = 0
        while not self._stop.is_set():
            if not self.gate():
                strikes = 0
                await self._wait(self.gate_poll_sec)
                continue
            try:
                async with self._connect(self.url) as ws:
                    self.counters['connects'] += 1
                    reason, got_any = await self._session(ws)
                if got_any:
                    strikes = 0
                    err_delay = self.error_delay
                if reason == 'stall':
                    strikes += 1
                    self.counters['stalls'] += 1
                    delay = self._stall_delay(strikes)
                    logger.info('stream silent %ss; reconnecting in %ss',
                                self.stall_sec, delay)
                    await self._wait(delay)
            except InvalidStatus as e:
                status = getattr(getattr(e, 'response', None), 'status_code', None)
                if status in (403, 429):
                    self.counters['refused'] += 1
                    logger.warning('stream refused (HTTP %s); waiting %ss',
                                   status, self.refused_wait)
                    await self._wait(self.refused_wait)
                else:
                    await self._error(e, err_delay)
                    err_delay = min(err_delay * 2, self.error_delay_max)
            except Exception as e:
                # ConnectionClosed (clean or not), DNS, TLS, timeouts: all
                # reconnect on the same back-off.
                await self._error(e, err_delay)
                err_delay = min(err_delay * 2, self.error_delay_max)

    async def _error(self, e, delay):
        self.counters['errors'] += 1
        logger.warning('stream error (%s: %s); reconnecting in %ss',
                       type(e).__name__, e, delay)
        await self._wait(delay)

    async def _session(self, ws):
        """Read until the gate closes or the stream stalls; a close or error
        raises out to run()."""
        loop = asyncio.get_running_loop()
        sent = frozenset()
        last_sub = last_msg = loop.time()
        got_any = False
        while not self._stop.is_set():
            if not self.gate():
                return 'gate', got_any
            now = loop.time()
            want = self.book.wanted()
            if want != sent or now - last_sub >= self.heartbeat_sec:
                gone = sent - want
                if gone:
                    await ws.send(json.dumps({'unsubscribe': sorted(gone)}))
                await ws.send(json.dumps({'subscribe': sorted(want)}))
                sent, last_sub = want, now
            if now - last_msg >= self.stall_sec:
                return 'stall', got_any
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except TimeoutError:
                continue
            last_msg = loop.time()
            got_any = True
            tick = parse_tick(decode_frame(raw))
            if tick:
                self.book.update(*tick)
        return 'stop', got_any


def _tree_mtime(root):
    """Newest mtime under *root* (directories included), or None."""
    newest = None
    stack = [str(root)]
    while stack:
        try:
            with os.scandir(stack.pop()) as it:
                for e in it:
                    try:
                        st = e.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    newest = st.st_mtime if newest is None else max(newest, st.st_mtime)
                    if e.is_dir(follow_symlinks=False):
                        stack.append(e.path)
        except OSError:
            continue
    return newest


class BuildSnapshots:
    """Immutable copies of the published docs/ folder.

    run_daily's publish rewrites docs/ in place, file by file, and a px/
    shard is an offset into prices_meta.json, so a page must load every
    sidecar from the build it was opened on. A copy is taken once the whole
    tree has been unchanged for ``settle_sec``; each lives at
    ``<store>/<build id>`` and ``keep`` are retained, so a page open across
    a publish still finds its shards.
    """

    def __init__(self, docs, store, settle_sec=None, keep=None):
        self.docs = Path(docs)
        self.store = Path(store)
        self.settle_sec = (config.LIVE_QUOTES_BUILD_SETTLE_SEC
                           if settle_sec is None else settle_sec)
        self.keep = keep or config.LIVE_QUOTES_BUILDS_KEEP
        self._lock = threading.Lock()

    def builds(self):
        try:
            return sorted((p.name for p in self.store.iterdir()
                           if p.is_dir() and p.name.isdigit()), key=int)
        except OSError:
            return []

    def latest(self):
        b = self.builds()
        return b[-1] if b else None

    def path(self, build):
        return self.store / build if build in self.builds() else None

    def refresh(self, now=None):
        """Take a copy if docs/ has changed and settled. Returns the latest id."""
        now = time.time() if now is None else now
        with self._lock:
            if not (self.docs / 'index.html').is_file():
                return self.latest()
            newest = _tree_mtime(self.docs)
            if newest is None or now - newest < self.settle_sec:
                return self.latest()
            build = str(int(newest))
            have = self.builds()
            if have and int(have[-1]) >= int(build):
                return have[-1]
            tmp = self.store / f'.{build}.tmp'
            shutil.rmtree(tmp, ignore_errors=True)
            self.store.mkdir(parents=True, exist_ok=True)
            shutil.copytree(self.docs, tmp)
            # The copy took time; if docs/ moved meanwhile, try again later.
            if (_tree_mtime(self.docs) or 0) > newest:
                shutil.rmtree(tmp, ignore_errors=True)
                return self.latest()
            os.replace(tmp, self.store / build)
            logger.info('serving build %s', build)
            for old in self.builds()[:-self.keep]:
                shutil.rmtree(self.store / old, ignore_errors=True)
            return build


def _primary_ip():
    # A UDP connect sends nothing; it only picks the outbound interface.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(('192.0.2.1', 9))
            return s.getsockname()[0]
    except OSError:
        return None


def allowed_hosts(extra=()):
    """Host names the server answers to; anything else is refused, so a page
    on another site cannot DNS-rebind its way to the report."""
    hn = socket.gethostname().lower()
    names = {'localhost', '127.0.0.1', '::1', '[::1]', hn,
             hn.split('.')[0], hn.split('.')[0] + '.local'}
    ip = _primary_ip()
    if ip:
        names.add(ip)
    env = os.environ.get('LIVE_QUOTES_ALLOWED_HOSTS', '')
    names.update(h.strip().lower() for h in env.split(',') if h.strip())
    names.update(h.lower() for h in extra)
    return names


def _host_name(header):
    h = (header or '').strip().lower()
    if h.startswith('['):
        return h.split(']')[0] + ']'
    return h.rsplit(':', 1)[0] if h.count(':') == 1 else h


class LiveQuotesServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, book, builds, streaming, hosts, streamer=None,
                 clock=time.time):
        self.book = book
        self.builds = builds
        self.streaming = streaming
        self.hosts = hosts
        self.streamer = streamer
        self.clock = clock
        super().__init__(addr, Handler)

    def market_open(self, now):
        now_et = datetime.fromtimestamp(now, ET)
        return (self.streaming and in_window(now_et)
                and self.book.session_open(now_et.date()))


class Handler(SimpleHTTPRequestHandler):
    server_version = 'LiveQuotes/1'

    def log_message(self, fmt, *args):
        logger.debug('%s %s', self.address_string(), fmt % args)

    def _send_json(self, code, obj):
        body = json.dumps(obj, separators=(',', ':')).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _host_ok(self):
        if _host_name(self.headers.get('Host')) in self.server.hosts:
            return True
        self.send_error(421, 'Unknown host')
        return False

    def do_HEAD(self):
        if self._host_ok():
            self._static(head=True)

    def do_GET(self):
        if not self._host_ok():
            return
        parts = urlsplit(self.path)
        m = API_RE.search(parts.path)
        if m:
            return self._health() if m.group(1) else self._quote(parts.query)
        self._static()

    def _quote(self, query):
        srv = self.server
        raw = ','.join(parse_qs(query).get('symbols', []))
        syms = list(dict.fromkeys(s.strip().upper() for s in raw.split(',') if s.strip()))
        if not syms or len(syms) > config.LIVE_QUOTES_MAX_PER_REQUEST \
                or not all(SYMBOL_RE.match(s) for s in syms):
            return self._send_json(400, {'error': 'symbols: 1-%d tickers'
                                         % config.LIVE_QUOTES_MAX_PER_REQUEST})
        now = srv.clock()
        if srv.streaming:
            srv.book.request(syms, now)
        today = datetime.fromtimestamp(now, ET).date()
        self._send_json(200, {
            'open': srv.market_open(now),
            'build': srv.builds.latest(),
            'asof': int(now * 1000),
            'symbols': srv.book.get(syms, today),
        })

    def _health(self):
        srv = self.server
        self._send_json(200, {
            'streaming': srv.streaming, 'open': srv.market_open(srv.clock()),
            'build': srv.builds.latest(), **srv.book.stats(),
            **(srv.streamer.counters if srv.streamer else {}),
        })

    def _static(self, head=False):
        path = unquote(urlsplit(self.path).path)
        if path in ('/', '/index.html'):
            build = self.server.builds.latest()
            if not build:
                return self.send_error(503, 'No published report yet')
            self.send_response(302)
            self.send_header('Location', f'/b/{build}/')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            return
        m = BUILD_PATH_RE.match(path)
        root = self.server.builds.path(m.group(1)) if m else None
        if root is None:
            return self.send_error(404)
        rel = m.group(2) or 'index.html'
        target = (root / rel).resolve()
        if not target.is_relative_to(root.resolve()):
            return self.send_error(404)
        if target.is_dir():
            target = target / 'index.html'
        if not target.is_file():
            return self.send_error(404)
        ctype = self.guess_type(str(target))
        try:
            f = open(target, 'rb')  # noqa: SIM115 -- copied out below
        except OSError:
            return self.send_error(404)
        with f:
            size = os.fstat(f.fileno()).st_size
            self.send_response(200)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(size))
            # A build directory never changes once written.
            self.send_header('Cache-Control', 'private, max-age=86400, immutable')
            self.end_headers()
            if not head:
                shutil.copyfileobj(f, self.wfile)


def _maintain(server, stop, interval=30):
    while not stop.wait(interval):
        try:
            server.book.expire(server.clock())
            server.builds.refresh()
        except Exception as e:
            logger.warning('maintenance failed: %s', e)


def _load_watch(spec):
    if not spec:
        return []
    p = Path(spec)
    text = p.read_text(encoding='utf-8') if p.is_file() else spec
    syms = [s.strip().upper() for s in re.split(r'[\s,]+', text) if s.strip()]
    return [s for s in syms if SYMBOL_RE.match(s)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=config.LIVE_QUOTES_PORT)
    ap.add_argument('--docs', default=str(DEFAULT_DOCS),
                    help='published report folder (default: the pages-live worktree)')
    ap.add_argument('--builds', default=str(DEFAULT_BUILDS),
                    help='where immutable copies of --docs are kept')
    ap.add_argument('--watch', default='',
                    help='tickers (or a file of them) kept subscribed all session')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s %(levelname)s %(name)s: %(message)s')

    streaming = os.environ.get('LIVE_QUOTES', '1') != '0'
    book = QuoteBook(pinned=_load_watch(args.watch))
    builds = BuildSnapshots(args.docs, args.builds)
    builds.refresh()
    if not builds.latest():
        logger.warning('no settled report in %s yet; / answers 503 until one is', args.docs)

    def gate():
        return (streaming and in_window(datetime.now(ET))
                and not run_lock_active())

    loop = asyncio.new_event_loop()
    streamer = None
    if streaming:
        streamer = Streamer(book, gate)
        threading.Thread(target=loop.run_until_complete, args=(streamer.run(),),
                         name='stream', daemon=True).start()
    else:
        logger.info('LIVE_QUOTES=0: serving the report without streaming')

    server = LiveQuotesServer((args.host, args.port), book, builds, streaming,
                              allowed_hosts(), streamer=streamer)
    stop = threading.Event()
    threading.Thread(target=_maintain, args=(server, stop), name='maintain',
                     daemon=True).start()
    logger.info('listening on http://%s:%d/ (hosts: %s)', args.host, args.port,
                ', '.join(sorted(server.hosts)))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        if streamer:
            loop.call_soon_threadsafe(streamer.stop)
        server.server_close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
