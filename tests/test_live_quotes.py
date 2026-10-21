"""scripts/live_quotes.py: the LAN quote service (parser, stream loop,
subscriptions, window and lock gating, immutable builds, HTTP)."""
import asyncio
import base64
import http.client
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from http import HTTPStatus
from pathlib import Path

import pytest

from scripts import config
from scripts import live_quotes as lq

REPO = Path(__file__).resolve().parents[1]
# A Wednesday, 10:00 ET.
WED_10 = datetime(2026, 10, 21, 10, 0, tzinfo=lq.ET)


def _frame(**fields):
    from yfinance.pricing_pb2 import PricingData
    raw = PricingData(**fields).SerializeToString()
    return json.dumps({'type': 'pricing', 'message': base64.b64encode(raw).decode()})


def _ms(dt):
    return int(dt.timestamp() * 1000)


# ---------------------------------------------------------------- parsing

def test_parse_tick_defaults_and_units():
    # change 0 and market_hours 0 are dropped on the wire; time is a string.
    d = lq.decode_frame(_frame(id='AAPL', price=231.4567, time=_ms(WED_10),
                               change=0.0, change_percent=1.25, market_hours=0))
    assert 'change' not in d and 'market_hours' not in d and isinstance(d['time'], str)
    sym, q = lq.parse_tick(d)
    assert sym == 'AAPL'
    assert q['c'] == 231.46
    assert q['d'] == 0
    assert q['dp'] == pytest.approx(0.0125)
    assert q['t'] == _ms(WED_10)
    assert q['mh'] == 0
    assert q['pc'] is None


def test_parse_tick_rounds_sub_dollar_to_four_places():
    _, q = lq.parse_tick(lq.decode_frame(_frame(id='X', price=0.123456, time=1)))
    assert q['c'] == 0.1235


@pytest.mark.parametrize('d', [None, {}, {'id': 'X', 'time': '1'},
                               {'id': 'X', 'price': 5.0},
                               {'id': 'X', 'price': 'nan?', 'time': '1'}])
def test_parse_tick_rejects_incomplete(d):
    assert lq.parse_tick(d) is None


def test_decode_frame_garbage_is_none():
    assert lq.decode_frame('not json') is None
    assert lq.decode_frame(json.dumps({'message': '!!!'})) is None


def test_stream_url_comes_from_yfinance():
    assert lq.stream_url().startswith('wss://streamer.finance.yahoo.com')


# ---------------------------------------------------------------- gating

@pytest.mark.parametrize('hm,ok', [((9, 29), False), ((9, 30), True),
                                   ((16, 4), True), ((16, 5), False)])
def test_window_edges(hm, ok):
    assert lq.in_window(WED_10.replace(hour=hm[0], minute=hm[1])) is ok


def test_window_closed_on_weekend():
    assert not lq.in_window(datetime(2026, 10, 24, 10, 0, tzinfo=lq.ET))


def test_run_lock(tmp_path):
    lock = tmp_path / '.run_daily.lock'
    assert not lq.run_lock_active(lock)
    lock.write_text(str(os.getpid()), encoding='utf-8')
    assert lq.run_lock_active(lock)
    p = subprocess.Popen([sys.executable, '-c', 'pass'])
    p.wait()
    lock.write_text(str(p.pid), encoding='utf-8')
    assert not lq.run_lock_active(lock)
    lock.write_text('garbage', encoding='utf-8')
    assert not lq.run_lock_active(lock)


def test_paths_resolve_from_repo_root_not_cwd():
    for p in (lq.RUN_LOCK, lq.DEFAULT_DOCS, lq.DEFAULT_BUILDS):
        assert p.is_absolute() and REPO in p.parents
    # launchd starts the agent with cwd /.
    r = subprocess.run([sys.executable, str(REPO / 'scripts' / 'live_quotes.py'), '--help'],
                       cwd='/', capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr


# ---------------------------------------------------------------- QuoteBook

def test_book_lru_cap_keeps_pinned():
    b = lq.QuoteBook(max_symbols=3, idle_sec=600)
    b.request(['A', 'B'], 0)
    b.request(['C'], 1)
    assert b.wanted() == {'SPY', 'B', 'C'}     # A, least recently asked, went
    b.request(['B'], 2)
    b.request(['D'], 3)
    assert b.wanted() == {'SPY', 'B', 'D'}


def test_book_idle_expiry():
    b = lq.QuoteBook(idle_sec=600)
    b.request(['A'], 0)
    b.request(['B'], 500)
    assert b.expire(700) == ['A']
    assert b.wanted() == {'SPY', 'B'}


def test_book_serves_only_todays_session():
    b = lq.QuoteBook()
    yday = WED_10.replace(day=20)
    b.update('A', {'c': 1.0, 'd': 0, 'dp': 0, 't': _ms(yday), 'pc': None, 'mh': 1})
    b.update('B', {'c': 2.0, 'd': 0, 'dp': 0, 't': _ms(WED_10), 'pc': None, 'mh': 1})
    got = b.get(['A', 'B', 'C'], WED_10.date())
    assert got['A'] is None and got['C'] is None
    assert got['B'] == {'c': 2.0, 'd': 0, 'dp': 0, 't': _ms(WED_10)}


def test_session_opens_only_on_regular_canary_tick():
    b = lq.QuoteBook()
    today = WED_10.date()
    b.update('AAPL', {'c': 1.0, 'd': 0, 'dp': 0, 't': _ms(WED_10), 'pc': None, 'mh': 1})
    b.update('SPY', {'c': 1.0, 'd': 0, 'dp': 0, 't': _ms(WED_10), 'pc': None, 'mh': 0})
    assert not b.session_open(today)
    b.update('SPY', {'c': 1.0, 'd': 0, 'dp': 0, 't': _ms(WED_10), 'pc': None, 'mh': 1})
    assert b.session_open(today)


def test_holiday_reconnects_stay_bounded():
    # A full weekday window with nothing trading: each stall costs
    # STALL_SEC of silence plus the doubling wait.
    o, c = config.LIVE_QUOTES_OPEN_ET, config.LIVE_QUOTES_CLOSE_ET
    window = (c[0] * 60 + c[1] - o[0] * 60 - o[1]) * 60
    t, strikes = 0, 0
    while True:
        t += config.LIVE_QUOTES_STALL_SEC
        if t >= window:
            break
        strikes += 1
        t += lq.stall_delay(strikes)
    assert [lq.stall_delay(n) for n in (1, 2, 3)] == [60, 120, 240]
    assert lq.stall_delay(99) == config.LIVE_QUOTES_STALL_DELAY_MAX
    assert strikes <= 20      # against ~390 with a flat one-minute retry


# ---------------------------------------------------------------- Streamer
# Against a real websocket server on localhost.

def _serve(handler, process_request=None):
    from websockets.asyncio.server import serve
    return serve(handler, '127.0.0.1', 0, process_request=process_request)


def _run_streamer(server_handler, *, until, timeout=5.0, process_request=None, **kw):
    """Run a Streamer against a local server until ``until(book, streamer)``."""
    book = lq.QuoteBook()

    async def go():
        async with _serve(server_handler, process_request) as srv:
            port = srv.sockets[0].getsockname()[1]
            st = lq.Streamer(book, gate=lambda: True, url=f'ws://127.0.0.1:{port}',
                             error_delay=0.01, error_delay_max=0.02, **kw)
            task = asyncio.create_task(st.run())
            deadline = asyncio.get_running_loop().time() + timeout
            while not until(book, st):
                assert asyncio.get_running_loop().time() < deadline, st.counters
                await asyncio.sleep(0.02)
            st.stop()
            await asyncio.wait_for(task, 2)
            return st

    return book, asyncio.run(go())


@pytest.mark.parametrize('how', ['clean', 'error'])
def test_reconnects_and_resubscribes_after_close(how):
    subs = []
    conns = [0]

    async def handler(ws):
        conns[0] += 1
        n = conns[0]
        msg = json.loads(await ws.recv())
        subs.append((n, msg))
        if n == 1:
            if how == 'clean':
                await ws.close()
            else:
                ws.transport.abort()
            return
        await ws.send(_frame(id='SPY', price=600.0, time=_ms(WED_10), market_hours=1))
        await ws.wait_closed()

    book, st = _run_streamer(handler, until=lambda b, s: b.ticks >= 1)
    assert st.counters['connects'] == 2 and st.counters['errors'] >= 1
    # The fresh socket was sent the subscription set again.
    assert [n for n, _ in subs] == [1, 2]
    assert all('SPY' in m['subscribe'] for _, m in subs)
    assert book.session_open(WED_10.date())


def test_stall_watchdog_reconnects_with_growing_wait():
    async def silent(ws):
        await ws.wait_closed()

    waits = []

    def delay(n):
        waits.append(n)
        return 0.01

    _, st = _run_streamer(silent, until=lambda b, s: s.counters['stalls'] >= 3,
                          stall_sec=0.2, stall_delay_fn=delay)
    assert waits[:3] == [1, 2, 3]       # strikes grow while nothing arrives
    assert st.counters['connects'] >= 3


def test_refused_handshake_waits_long():
    async def handler(ws):
        await ws.wait_closed()

    def refuse(conn, request):
        return conn.respond(HTTPStatus.TOO_MANY_REQUESTS, 'slow down\n')

    _, st = _run_streamer(handler, until=lambda b, s: s.counters['refused'] >= 1,
                          process_request=refuse, refused_wait=30)
    assert st.counters['connects'] == 0 and st.counters['errors'] == 0


def test_closed_gate_never_connects():
    async def go():
        st = lq.Streamer(lq.QuoteBook(), gate=lambda: False, url='ws://127.0.0.1:9',
                         gate_poll_sec=0.01)
        task = asyncio.create_task(st.run())
        await asyncio.sleep(0.1)
        st.stop()
        await asyncio.wait_for(task, 1)
        return st
    assert asyncio.run(go()).counters['connects'] == 0


def test_heartbeat_follows_demand():
    seen = []

    async def handler(ws):
        async for m in ws:
            seen.append(json.loads(m))

    book = lq.QuoteBook()

    async def go():
        async with _serve(handler) as srv:
            port = srv.sockets[0].getsockname()[1]
            st = lq.Streamer(book, gate=lambda: True, url=f'ws://127.0.0.1:{port}',
                             heartbeat_sec=0.1, stall_sec=30)
            task = asyncio.create_task(st.run())
            await asyncio.sleep(0.2)
            book.request(['AAPL'], time.time())
            await asyncio.sleep(1.3)  # the loop reads demand between 1 s recv waits
            book.expire(time.time() + 10_000)
            await asyncio.sleep(1.3)  # the loop reads demand between 1 s recv waits
            st.stop()
            await asyncio.wait_for(task, 2)

    asyncio.run(go())
    assert {'subscribe': ['SPY']} in seen
    assert {'subscribe': ['AAPL', 'SPY']} in seen
    assert {'unsubscribe': ['AAPL']} in seen


# ---------------------------------------------------------------- builds

def _docs(root, shard='A'):
    (root / 'px').mkdir(parents=True, exist_ok=True)
    (root / 'index.html').write_text('<html>report</html>', encoding='utf-8')
    (root / 'px' / 's.json').write_text(shard, encoding='utf-8')


def _touch_tree(root, ts):
    for p in [root, *root.rglob('*')]:
        os.utime(p, (ts, ts))


def test_builds_settle_copy_and_prune(tmp_path):
    docs, store = tmp_path / 'docs', tmp_path / 'store'
    _docs(docs)
    _touch_tree(docs, 1_000_000)
    b = lq.BuildSnapshots(docs, store, settle_sec=120, keep=2)
    assert b.refresh(now=1_000_060) is None          # still settling
    assert b.refresh(now=1_000_200) == '1000000'
    for n, ts in ((2, 2_000_000), (3, 3_000_000)):
        _docs(docs, shard=f'v{n}')
        _touch_tree(docs, ts)
        b.refresh(now=ts + 500)
    assert b.builds() == ['2000000', '3000000']
    assert (store / '2000000' / 'px' / 's.json').read_text(encoding='utf-8') == 'v2'


def test_builds_missing_docs(tmp_path):
    b = lq.BuildSnapshots(tmp_path / 'nope', tmp_path / 'store', settle_sec=0)
    assert b.refresh() is None and b.latest() is None


# ---------------------------------------------------------------- HTTP

@pytest.fixture
def server(tmp_path):
    docs, store = tmp_path / 'docs', tmp_path / 'store'
    _docs(docs)
    _touch_tree(docs, 1_000_000)
    builds = lq.BuildSnapshots(docs, store, settle_sec=0)
    builds.refresh(now=2_000_000)
    book = lq.QuoteBook()
    srv = lq.LiveQuotesServer(('127.0.0.1', 0), book, builds, True,
                              {'localhost', '127.0.0.1'},
                              clock=lambda: WED_10.timestamp())
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    srv.docs = docs
    yield srv
    srv.shutdown()
    srv.server_close()


def _get(srv, path, host='127.0.0.1'):
    c = http.client.HTTPConnection('127.0.0.1', srv.server_address[1], timeout=5)
    c.putrequest('GET', path, skip_host=True)
    c.putheader('Host', f'{host}:{srv.server_address[1]}')
    c.endheaders()
    r = c.getresponse()
    body = r.read()
    c.close()
    return r, body


def test_root_redirects_to_newest_build(server):
    r, _ = _get(server, '/')
    assert r.status == 302 and r.getheader('Location') == '/b/1000000/'
    r, body = _get(server, '/b/1000000/')
    assert r.status == 200 and body == b'<html>report</html>'


def test_page_keeps_its_builds_shards_after_a_publish(server):
    _docs(server.docs, shard='B')
    _touch_tree(server.docs, 1_500_000)
    server.builds.refresh(now=2_000_000)
    assert server.builds.latest() == '1500000'
    _, old = _get(server, '/b/1000000/px/s.json')
    _, new = _get(server, '/b/1500000/px/s.json')
    assert (old, new) == (b'A', b'B')


@pytest.mark.parametrize('path', ['/b/1000000/px/', '/b/1000000/../../x',
                                  '/b/1000000/%2e%2e/%2e%2e/x', '/b/999/index.html',
                                  '/px/s.json'])
def test_no_listing_traversal_or_unknown_build(server, path):
    assert _get(server, path)[0].status == 404


def test_foreign_host_refused(server):
    assert _get(server, '/', host='evil.example')[0].status == 421
    assert _get(server, '/api/quote?symbols=SPY', host='evil.example')[0].status == 421


def test_quote_api(server):
    server.book.update('AAPL', {'c': 231.0, 'd': 1.5, 'dp': 0.0065, 't': _ms(WED_10),
                                'pc': 229.5, 'mh': 1})
    r, body = _get(server, '/b/1000000/api/quote?symbols=aapl,MSFT')
    out = json.loads(body)
    assert r.status == 200 and r.getheader('Cache-Control') == 'no-store'
    assert out['build'] == '1000000'
    assert out['open'] is False                       # no regular SPY tick yet
    assert out['symbols'] == {'AAPL': {'c': 231.0, 'd': 1.5, 'dp': 0.0065,
                                       't': _ms(WED_10)}, 'MSFT': None}
    assert {'AAPL', 'MSFT'} <= server.book.wanted()
    server.book.update('SPY', {'c': 600.0, 'd': 0, 'dp': 0, 't': _ms(WED_10),
                               'pc': None, 'mh': 1})
    assert json.loads(_get(server, '/api/quote?symbols=SPY')[1])['open'] is True


@pytest.mark.parametrize('q', ['', 'symbols=', 'symbols=AA;PL', 'symbols=TOOLONGSYMBOL1',
                               'symbols=' + ','.join(f'T{i}' for i in range(101))])
def test_quote_api_rejects_bad_symbols(server, q):
    assert _get(server, '/api/quote?' + q)[0].status == 400


def test_health(server):
    r, body = _get(server, '/api/quote/health')
    h = json.loads(body)
    assert r.status == 200 and h['streaming'] is True and h['build'] == '1000000'


# ---------------------------------------------------------------- live

@pytest.mark.network
def test_live_stream_spy_tick():
    if not lq.in_window(datetime.now(lq.ET)):
        pytest.skip('outside the streaming window')
    book = lq.QuoteBook()

    async def go():
        st = lq.Streamer(book, gate=lambda: True)
        task = asyncio.create_task(st.run())
        for _ in range(600):
            if book.ticks:
                break
            await asyncio.sleep(0.1)
        st.stop()
        await asyncio.wait_for(task, 5)

    asyncio.run(go())
    assert book.ticks >= 1
