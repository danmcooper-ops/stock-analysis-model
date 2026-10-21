// Cloudflare Pages "advanced mode" Worker for the stock report: every request
// to the Pages site passes through here before a static file is served.
//
// Cloudflare Access (Zero Trust) does the login: it sits in front of the
// *.pages.dev hostname, lets only allowlisted emails (and the nightly run's
// service token) through, and adds a signed JWT in the Cf-Access-Jwt-Assertion
// header. This Worker re-checks that JWT against the team's public keys and
// the Access application's audience tag, then hands the request to the
// deployed static files (env.ASSETS). That makes the site fail CLOSED: if the
// Access application is deleted, disabled, or does not cover a hostname (a
// per-deployment <hash>.<project>.pages.dev URL, say), the answer is 403, not
// the report.
//
// It also answers GET /api/quote?symbols=A,B with live prices from Finnhub
// (behind the same login), when the Pages project has a FINNHUB_API_KEY
// secret; see "Live quotes" below.
//
// run.sh step 08b copies this file into the deploy directory as _worker.js
// via scripts/stage_pages_worker.py, which fills in the two placeholders
// below from CF_ACCESS_TEAM_DOMAIN and CF_ACCESS_AUD. Pages never serves
// _worker.js itself as a static file.

const ACCESS_TEAM_DOMAIN = '__ACCESS_TEAM_DOMAIN__';
const ACCESS_AUD = '__ACCESS_AUD__';

const JWKS_TTL_MS = 60 * 60 * 1000;
let jwksCache = { team: null, keys: null, at: 0 };

// The `_headers` file is not applied to responses that pass through an
// advanced-mode Worker, so its rules are repeated here.
const SECURITY_HEADERS = {
  'x-content-type-options': 'nosniff',
  'referrer-policy': 'strict-origin-when-cross-origin',
};

export function accessConfig(env = {}) {
  const team = env.ACCESS_TEAM_DOMAIN || ACCESS_TEAM_DOMAIN;
  const aud = env.ACCESS_AUD || ACCESS_AUD;
  // An unfilled placeholder counts as unset.
  if (!team || !aud || team.startsWith('__') || aud.startsWith('__')) return null;
  return { team, aud };
}

export default {
  async fetch(request, env) {
    const cfg = accessConfig(env);
    if (!cfg) return text(500, 'Access is not configured for this site');
    const token = request.headers.get('cf-access-jwt-assertion');
    if (!token || !(await verifyAccessJwt(token, cfg))) return text(403, 'Forbidden');

    const url = new URL(request.url);
    if (/(^|\/)api\/quote$/.test(url.pathname)) return quoteResponse(url, env);

    const resp = await env.ASSETS.fetch(request);
    const out = new Response(resp.body, resp);
    for (const [k, v] of Object.entries(SECURITY_HEADERS)) out.headers.set(k, v);
    // Authenticated content must never be kept by a shared cache. Pages'
    // own max-age=0/must-revalidate + ETag behaviour is otherwise kept (see
    // scheduled-tasks/cloud-daily-stock-analysis/pages_headers for why).
    out.headers.set('cache-control', privateCacheControl(out.headers.get('cache-control')));
    return announceQuotes(out, env);
  },
};

export function privateCacheControl(cc) {
  if (!cc) return 'private, no-cache';
  if (/\bprivate\b/.test(cc)) return cc;
  return /\bpublic\b/.test(cc) ? cc.replace(/\bpublic\b/, 'private') : `private, ${cc}`;
}

function text(status, body) {
  return new Response(body + '\n', {
    status,
    headers: { 'content-type': 'text/plain; charset=utf-8', 'cache-control': 'no-store', ...SECURITY_HEADERS },
  });
}

function b64urlDecode(s) {
  s = s.replace(/-/g, '+').replace(/_/g, '/');
  while (s.length % 4) s += '=';
  return Uint8Array.from(atob(s), (c) => c.charCodeAt(0));
}

async function teamKeys(team, force) {
  const now = Date.now();
  if (!force && jwksCache.team === team && jwksCache.keys && now - jwksCache.at < JWKS_TTL_MS) {
    return jwksCache.keys;
  }
  const resp = await fetch(`https://${team}/cdn-cgi/access/certs`);
  if (!resp.ok) throw new Error(`JWKS fetch ${resp.status}`);
  const { keys } = await resp.json();
  jwksCache = { team, keys, at: now };
  return keys;
}

// True when `token` is an RS256 JWT signed by the team's Access keys, for
// this application's audience, issued by this team and not expired.
export async function verifyAccessJwt(token, cfg, nowSeconds = Math.floor(Date.now() / 1000)) {
  try {
    const parts = token.split('.');
    if (parts.length !== 3) return false;
    const dec = new TextDecoder();
    const header = JSON.parse(dec.decode(b64urlDecode(parts[0])));
    const payload = JSON.parse(dec.decode(b64urlDecode(parts[1])));
    if (header.alg !== 'RS256' || !header.kid) return false;

    let jwk = (await teamKeys(cfg.team, false)).find((k) => k.kid === header.kid);
    if (!jwk) jwk = (await teamKeys(cfg.team, true)).find((k) => k.kid === header.kid); // key rotation
    if (!jwk) return false;

    const key = await crypto.subtle.importKey(
      'jwk', jwk, { name: 'RSASSA-PKCS1-v1_5', hash: 'SHA-256' }, false, ['verify'],
    );
    const ok = await crypto.subtle.verify(
      'RSASSA-PKCS1-v1_5', key, b64urlDecode(parts[2]),
      new TextEncoder().encode(`${parts[0]}.${parts[1]}`),
    );
    if (!ok) return false;

    const aud = Array.isArray(payload.aud) ? payload.aud : [payload.aud];
    if (!aud.includes(cfg.aud)) return false;
    if (payload.iss !== `https://${cfg.team}`) return false;
    if (typeof payload.exp !== 'number' || payload.exp <= nowSeconds) return false;
    if (typeof payload.nbf === 'number' && payload.nbf > nowSeconds + 60) return false;
    return true;
  } catch {
    return false;
  }
}

// Exposed for the tests only.
export function _resetJwksCache() {
  jwksCache = { team: null, keys: null, at: 0 };
}

// --- Live quotes -------------------------------------------------------------
// The report's _lq* block polls api/quote for what is on screen and shows a
// live price and MoS beside the nightly close (display only). The page only
// asks when it finds <meta name="live-quotes">, which is added here, to HTML,
// only when the key is set -- so GitHub Pages (no Worker) never asks.
//
// Finnhub's free tier allows 60 calls a minute and quotes one symbol per call,
// so quotes are kept QUOTE_TTL_MS per isolate, one request fetches at most
// QUOTE_MAX_FETCH of them (never-fetched first, then the stalest, so every row
// on screen refreshes in turn), each isolate spends at most QUOTE_BUDGET a
// minute, and a 429 pauses fetching for a minute. A symbol not fetched this
// time is left out of the answer, so the page keeps what it already shows.

const QUOTE_TTL_MS = 30 * 1000;
const QUOTE_MAX_FETCH = 8;
const QUOTE_BUDGET = 50;
const QUOTE_MAX_SYMBOLS = 100;
const QUOTE_SYMBOL_RE = /^[A-Z0-9.\-]{1,12}$/;
// Weekdays, America/New_York, as scripts/live_quotes.py.
const QUOTE_OPEN_ET = 9 * 60 + 30;
const QUOTE_CLOSE_ET = 16 * 60 + 5;

let quoteCache = new Map();   // symbol -> { v: quote|null, at: ms }
let quoteCalls = [];          // timestamps of Finnhub calls in the last minute
let quotePausedUntil = 0;

const ET_FMT = new Intl.DateTimeFormat('en-US', {
  timeZone: 'America/New_York', weekday: 'short', year: 'numeric', month: '2-digit',
  day: '2-digit', hour: '2-digit', minute: '2-digit', hourCycle: 'h23',
});

export function etClock(ms) {
  const p = Object.fromEntries(ET_FMT.formatToParts(new Date(ms)).map((x) => [x.type, x.value]));
  return { day: `${p.year}-${p.month}-${p.day}`, weekday: p.weekday, minutes: +p.hour * 60 + +p.minute };
}

export function inQuoteWindow(ms) {
  const c = etClock(ms);
  return c.weekday !== 'Sat' && c.weekday !== 'Sun'
    && c.minutes >= QUOTE_OPEN_ET && c.minutes < QUOTE_CLOSE_ET;
}

// The report uses Yahoo's symbols; Finnhub writes share classes with a dot.
export function finnhubSymbol(sym) {
  return sym.replace(/-/g, '.');
}

// Finnhub /quote -> the report's { c, d, dp, t }: dp as a fraction, t in ms.
// c of 0 is Finnhub's answer for a symbol it does not know.
export function parseFinnhubQuote(j) {
  if (!j || !(j.c > 0) || !(j.t > 0)) return null;
  return { c: j.c, d: j.d ?? 0, dp: +((j.dp ?? 0) / 100).toFixed(6), t: j.t * 1000 };
}

function quoteBudgetLeft(now) {
  quoteCalls = quoteCalls.filter((t) => now - t < 60 * 1000);
  return now < quotePausedUntil ? 0 : QUOTE_BUDGET - quoteCalls.length;
}

async function fetchQuote(sym, key, now) {
  quoteCalls.push(now);
  const resp = await fetch(
    `https://finnhub.io/api/v1/quote?symbol=${encodeURIComponent(finnhubSymbol(sym))}&token=${encodeURIComponent(key)}`,
  );
  if (resp.status === 429) {
    quotePausedUntil = now + 60 * 1000;
    return undefined;
  }
  if (!resp.ok) return undefined;
  const v = parseFinnhubQuote(await resp.json());
  quoteCache.set(sym, { v, at: now });
  return v;
}

export async function quoteResponse(url, env, now = Date.now()) {
  const key = env.FINNHUB_API_KEY;
  if (!key) return text(404, 'Not found');
  const raw = (url.searchParams.get('symbols') || '').split(',').map((s) => s.trim().toUpperCase()).filter(Boolean);
  const syms = [...new Set(raw)];
  if (!syms.length || syms.length > QUOTE_MAX_SYMBOLS || !syms.every((s) => QUOTE_SYMBOL_RE.test(s))) {
    return json(400, { error: `symbols: 1-${QUOTE_MAX_SYMBOLS} tickers` });
  }
  const today = etClock(now).day;
  if (!inQuoteWindow(now)) return json(200, { open: false, asof: now, symbols: {} });

  // Never fetched first (in the page's order: the open ticker leads), then
  // the stalest expired ones.
  const due = syms
    .filter((s) => { const c = quoteCache.get(s); return !c || now - c.at >= QUOTE_TTL_MS; })
    .sort((a, b) => (quoteCache.get(a)?.at ?? -1) - (quoteCache.get(b)?.at ?? -1));
  const n = Math.max(0, Math.min(QUOTE_MAX_FETCH, quoteBudgetLeft(now), due.length));
  await Promise.all(due.slice(0, n).map((s) => fetchQuote(s, key, now).catch(() => undefined)));

  const symbols = {};
  let known = 0, fromToday = 0;
  for (const s of syms) {
    const c = quoteCache.get(s);
    if (!c) continue;                      // not fetched yet: the page keeps what it has
    if (c.v && etClock(c.v.t).day === today) { symbols[s] = c.v; fromToday++; }
    else symbols[s] = null;                // unknown symbol, or no trade today
    known++;
  }
  // A weekday with no trade at all is a market holiday: say closed, so the
  // page polls slowly instead of spending the budget every 15 s.
  return json(200, { open: !(known > 0 && fromToday === 0), asof: now, symbols });
}

// Tell the page api/quote exists here, on HTML only and only with a key.
export function announceQuotes(resp, env) {
  if (!env.FINNHUB_API_KEY || typeof HTMLRewriter === 'undefined') return resp;
  if (!(resp.headers.get('content-type') || '').includes('text/html')) return resp;
  return new HTMLRewriter().on('head', {
    element(el) { el.append('<meta name="live-quotes" content="finnhub">', { html: true }); },
  }).transform(resp);
}

function json(status, obj) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { 'content-type': 'application/json', 'cache-control': 'private, no-store', ...SECURITY_HEADERS },
  });
}

// Exposed for the tests only.
export function _resetQuotes() {
  quoteCache = new Map();
  quoteCalls = [];
  quotePausedUntil = 0;
}
