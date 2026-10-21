// Run with: cd cloudflare/pages && npm test
// Exercises the Pages Worker with real RS256 tokens against a stubbed JWKS
// endpoint and a stand-in for the Pages static-asset binding (env.ASSETS).
import { test, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import worker, {
  verifyAccessJwt, accessConfig, privateCacheControl, _resetJwksCache,
  quoteResponse, announceQuotes, finnhubSymbol, parseFinnhubQuote, inQuoteWindow, _resetQuotes,
} from '../_worker.js';

const TEAM = 'example.cloudflareaccess.com';
const AUD = 'aud-tag-123';
const cfg = { team: TEAM, aud: AUD };

const b64url = (buf) => Buffer.from(buf).toString('base64url');

async function keypair(kid) {
  const kp = await crypto.subtle.generateKey(
    { name: 'RSASSA-PKCS1-v1_5', modulusLength: 2048, publicExponent: new Uint8Array([1, 0, 1]), hash: 'SHA-256' },
    true, ['sign', 'verify'],
  );
  const jwk = { ...(await crypto.subtle.exportKey('jwk', kp.publicKey)), kid, alg: 'RS256' };
  return { kid, privateKey: kp.privateKey, jwk };
}

async function sign(kp, claims, header = {}) {
  const h = b64url(JSON.stringify({ alg: 'RS256', kid: kp.kid, typ: 'JWT', ...header }));
  const p = b64url(JSON.stringify(claims));
  const sig = await crypto.subtle.sign('RSASSA-PKCS1-v1_5', kp.privateKey, new TextEncoder().encode(`${h}.${p}`));
  return `${h}.${p}.${b64url(sig)}`;
}

const now = () => Math.floor(Date.now() / 1000);
const goodClaims = () => ({ aud: [AUD], iss: `https://${TEAM}`, exp: now() + 3600, email: 'me@example.com' });

let good, rogue, jwksFetches, assetRequests, finnhub, finnhubCalls;
beforeEach(async () => {
  _resetJwksCache();
  _resetQuotes();
  good ??= await keypair('k1');
  rogue ??= await keypair('k1'); // same kid, different key
  jwksFetches = 0;
  assetRequests = [];
  finnhubCalls = [];
  // symbol -> Finnhub /quote body, or a status number to answer with.
  finnhub = {};
  globalThis.fetch = async (url) => {
    if (url.startsWith('https://finnhub.io/')) {
      const u = new URL(url);
      assert.equal(u.pathname, '/api/v1/quote');
      assert.equal(u.searchParams.get('token'), 'fh-key');
      const sym = u.searchParams.get('symbol');
      finnhubCalls.push(sym);
      const a = finnhub[sym];
      if (typeof a === 'number') return new Response('', { status: a });
      return new Response(JSON.stringify(a ?? { c: 0, d: null, dp: null, t: 0 }));
    }
    assert.equal(url, `https://${TEAM}/cdn-cgi/access/certs`);
    jwksFetches++;
    return new Response(JSON.stringify({ keys: [good.jwk] }));
  };
});

const ASSETS = {
  async fetch(request) {
    assetRequests.push(new URL(request.url).pathname);
    return new Response('<html>2026-09-26</html>', {
      status: 200,
      headers: { 'content-type': 'text/html', 'cache-control': 'public, max-age=0, must-revalidate', etag: '"e1"' },
    });
  },
};
const env = (extra = {}) => ({ ACCESS_TEAM_DOMAIN: TEAM, ACCESS_AUD: AUD, ASSETS, ...extra });

async function get(path, token, e = env()) {
  const headers = token ? { 'cf-access-jwt-assertion': token } : {};
  return worker.fetch(new Request(`https://stock-analysis.pages.dev${path}`, { headers }), e);
}

test('valid token is passed through to the static assets, made private', async () => {
  const r = await get('/details_index.json', await sign(good, goodClaims()));
  assert.equal(r.status, 200);
  assert.match(await r.text(), /2026-09-26/);
  assert.deepEqual(assetRequests, ['/details_index.json']);
  assert.equal(r.headers.get('cache-control'), 'private, max-age=0, must-revalidate');
  assert.equal(r.headers.get('etag'), '"e1"');
  assert.equal(r.headers.get('x-content-type-options'), 'nosniff');
});

test('no token -> 403 and the assets are never touched', async () => {
  const r = await get('/');
  assert.equal(r.status, 403);
  assert.deepEqual(assetRequests, []);
});

test('rejects expired, wrong audience, wrong issuer, forged signature, alg none, garbage', async () => {
  const cases = [
    await sign(good, { ...goodClaims(), exp: now() - 10 }),
    await sign(good, { ...goodClaims(), aud: ['someone-else'] }),
    await sign(good, { ...goodClaims(), iss: 'https://evil.cloudflareaccess.com' }),
    await sign(rogue, goodClaims()),
    await sign(good, goodClaims(), { alg: 'none' }),
    'not.a.jwt',
  ];
  for (const t of cases) assert.equal(await verifyAccessJwt(t, cfg), false, t.slice(0, 40));
  assert.equal(await verifyAccessJwt(await sign(good, goodClaims()), cfg), true);
});

test('unconfigured (placeholders left in) fails closed with 500', async () => {
  const r = await get('/', await sign(good, goodClaims()), { ASSETS });
  assert.equal(r.status, 500);
  assert.deepEqual(assetRequests, []);
  assert.equal(accessConfig({}), null);
  assert.deepEqual(accessConfig({ ACCESS_TEAM_DOMAIN: TEAM, ACCESS_AUD: AUD }), cfg);
});

test('JWKS is cached across requests', async () => {
  const t = await sign(good, goodClaims());
  await get('/', t); await get('/hist_index.json', t);
  assert.equal(jwksFetches, 1);
});

test('cache-control is always private', () => {
  assert.equal(privateCacheControl(null), 'private, no-cache');
  assert.equal(privateCacheControl('max-age=600'), 'private, max-age=600');
  assert.equal(privateCacheControl('private, max-age=0'), 'private, max-age=0');
});


// --- Live quotes -------------------------------------------------------------
// Wednesday 2026-10-21, 10:00 ET.
const WED_10 = Date.UTC(2026, 9, 21, 14, 0);
const T_TODAY = WED_10 / 1000 - 60;
const T_YDAY = WED_10 / 1000 - 86400;
const fq = (c, t = T_TODAY) => ({ c, d: 1.5, dp: 0.75, h: c, l: c, o: c, pc: c - 1.5, t });
const qenv = { FINNHUB_API_KEY: 'fh-key' };
const ask = async (syms, now = WED_10, e = qenv) =>
  (await quoteResponse(new URL(`https://x.pages.dev/api/quote?symbols=${syms}`), e, now)).json();

test('quote route sits behind the login and needs the key', async () => {
  assert.equal((await get('/api/quote?symbols=AAPL')).status, 403);
  const t = await sign(good, goodClaims());
  const r = await get('/api/quote?symbols=AAPL', t);
  assert.equal(r.status, 404);                     // no FINNHUB_API_KEY
  assert.deepEqual(assetRequests, []);
  const r2 = await get('/b/1/api/quote?symbols=AAPL', t, env(qenv));
  assert.equal(r2.status, 200);
  assert.equal(r2.headers.get('cache-control'), 'private, no-store');
  assert.match(r2.headers.get('content-type'), /json/);
});

test('bad symbol lists are refused', async () => {
  for (const q of ['', 'AA;PL', 'TOOLONGSYMBOL1', Array.from({ length: 101 }, (_, i) => `T${i}`).join(',')]) {
    const r = await quoteResponse(new URL(`https://x/api/quote?symbols=${q}`), qenv, WED_10);
    assert.equal(r.status, 400, q.slice(0, 20));
  }
});

test('outside the window: closed, and Finnhub is not called', async () => {
  for (const at of [Date.UTC(2026, 9, 21, 13, 29), Date.UTC(2026, 9, 21, 20, 5), Date.UTC(2026, 9, 24, 15, 0)]) {
    const out = await ask('AAPL', at);
    assert.deepEqual([out.open, out.symbols], [false, {}]);
  }
  assert.deepEqual(finnhubCalls, []);
  assert.equal(inQuoteWindow(Date.UTC(2026, 9, 21, 13, 30)), true);
  assert.equal(inQuoteWindow(Date.UTC(2026, 9, 21, 20, 4)), true);
});

test('quotes map to the report shape; share classes use a dot', async () => {
  finnhub = { AAPL: fq(344), 'BRK.B': fq(539) };
  const out = await ask('AAPL,BRK-B,NOPE');
  assert.equal(out.open, true);
  assert.deepEqual(out.symbols.AAPL, { c: 344, d: 1.5, dp: 0.0075, t: T_TODAY * 1000 });
  assert.equal(out.symbols['BRK-B'].c, 539);
  assert.equal(out.symbols.NOPE, null);            // Finnhub's c: 0
  assert.deepEqual(finnhubCalls, ['AAPL', 'BRK.B', 'NOPE']);
  assert.equal(finnhubSymbol('BF-B'), 'BF.B');
  assert.equal(parseFinnhubQuote({ c: 0, t: 0 }), null);
  assert.equal(parseFinnhubQuote({ c: 1, dp: -0.1538, t: 1 }).dp, -0.001538);   // no float noise
});

test('at most 8 fetches a request; the rest are left out and rotate in', async () => {
  const syms = Array.from({ length: 20 }, (_, i) => `S${i}`);
  for (const s of syms) finnhub[s] = fq(10);
  let out = await ask(syms.join(','));
  assert.deepEqual(finnhubCalls, syms.slice(0, 8));
  assert.deepEqual(Object.keys(out.symbols), syms.slice(0, 8));   // others: keep what the page has
  out = await ask(syms.join(','), WED_10 + 15000);
  assert.deepEqual(finnhubCalls.slice(8), syms.slice(8, 16));
  // Cached for 30 s: once all are in, the stalest go first.
  await ask(syms.join(','), WED_10 + 20000);
  finnhubCalls = [];
  await ask(syms.join(','), WED_10 + 31000);
  assert.deepEqual(finnhubCalls, syms.slice(0, 8));
  assert.equal(Object.keys((await ask(syms.join(','), WED_10 + 32000)).symbols).length, 20);
});

test('a per-minute budget, and a 429 pauses fetching for a minute', async () => {
  for (let i = 0; i < 80; i++) finnhub[`S${i}`] = fq(10);
  const all = Array.from({ length: 80 }, (_, i) => `S${i}`).join(',');
  for (let k = 0; k < 8; k++) await ask(all, WED_10 + k * 1000);
  assert.equal(finnhubCalls.length, 50);
  _resetQuotes(); finnhubCalls = [];
  finnhub = { A: 429, B: fq(5) };
  await ask('A,B');
  finnhubCalls = [];
  await ask('A,B', WED_10 + 30000);
  assert.deepEqual(finnhubCalls, []);
  await ask('A,B', WED_10 + 61000);
  assert.deepEqual(finnhubCalls, ['A', 'B']);
});

test('a weekday with no trade today reads as closed (holiday)', async () => {
  finnhub = { AAPL: fq(344, T_YDAY), KO: fq(92, T_YDAY) };
  const out = await ask('AAPL,KO');
  assert.equal(out.open, false);
  assert.deepEqual(out.symbols, { AAPL: null, KO: null });
});

test('the page is told about the route only on HTML and only with a key', () => {
  const seen = [];
  globalThis.HTMLRewriter = class {
    on(sel, h) { this.sel = sel; this.h = h; return this; }
    transform(r) { const el = { append: (html) => seen.push([this.sel, html]) }; this.h.element(el); return r; }
  };
  try {
    const html = () => new Response('<html>', { headers: { 'content-type': 'text/html; charset=utf-8' } });
    announceQuotes(html(), qenv);
    announceQuotes(html(), {});
    announceQuotes(new Response('{}', { headers: { 'content-type': 'application/json' } }), qenv);
    assert.deepEqual(seen, [['head', '<meta name="live-quotes" content="finnhub">']]);
  } finally {
    delete globalThis.HTMLRewriter;
  }
});
