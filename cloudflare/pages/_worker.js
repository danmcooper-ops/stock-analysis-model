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

    const resp = await env.ASSETS.fetch(request);
    const out = new Response(resp.body, resp);
    for (const [k, v] of Object.entries(SECURITY_HEADERS)) out.headers.set(k, v);
    // Authenticated content must never be kept by a shared cache. Pages'
    // own max-age=0/must-revalidate + ETag behaviour is otherwise kept (see
    // scheduled-tasks/cloud-daily-stock-analysis/pages_headers for why).
    out.headers.set('cache-control', privateCacheControl(out.headers.get('cache-control')));
    return out;
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
