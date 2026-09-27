// Serves the stock report from the R2 bucket, behind Cloudflare Access.
//
// Access (Zero Trust) does the login: it sits in front of this Worker's
// hostname and only lets allowlisted emails (or the nightly publish's service
// token) through, adding a signed JWT in the Cf-Access-Jwt-Assertion header.
// This Worker re-checks that JWT against the team's public keys and the
// Access application's audience tag, so the site fails CLOSED if the Access
// application is ever deleted, disabled or detached from this hostname —
// without the check, that mistake would quietly publish everything.
//
// Files are uploaded by scripts/publish_report.py. The page fetches its
// sidecars with relative URLs, so they ride the same Access session cookie.

const JWKS_TTL_MS = 60 * 60 * 1000;
let jwksCache = { team: null, keys: null, at: 0 };

export default {
  async fetch(request, env) {
    if (request.method !== 'GET' && request.method !== 'HEAD') {
      return text(405, 'Method not allowed', { allow: 'GET, HEAD' });
    }
    if (!env.ACCESS_TEAM_DOMAIN || !env.ACCESS_AUD) {
      return text(500, 'Access is not configured on this Worker (ACCESS_TEAM_DOMAIN / ACCESS_AUD)');
    }
    const token = request.headers.get('cf-access-jwt-assertion');
    if (!token || !(await verifyAccessJwt(token, env))) {
      return text(403, 'Forbidden');
    }

    const url = new URL(request.url);
    let key;
    try {
      key = decodeURIComponent(url.pathname).replace(/^\/+/, '');
    } catch {
      return text(400, 'Bad path');
    }
    if (key === '' || key.endsWith('/')) key += 'index.html';
    if (key.split('/').some((p) => p === '..' || p === '.')) return text(400, 'Bad path');

    const obj = request.method === 'HEAD'
      ? await env.BUCKET.head(key)
      : await env.BUCKET.get(key, { onlyIf: request.headers });
    if (obj === null) return text(404, 'Not found');

    const headers = new Headers();
    obj.writeHttpMetadata(headers);
    headers.set('etag', obj.httpEtag);
    // `private`: authenticated content must never sit in a shared cache.
    // Shards are per-ticker and re-fetched rarely; everything else changes
    // nightly and revalidates by ETag (a cheap 304).
    headers.set('cache-control', /^(px|vol)\//.test(key) ? 'private, max-age=600' : 'private, no-cache');
    headers.set('x-content-type-options', 'nosniff');

    if (request.method === 'HEAD') {
      headers.set('content-length', String(obj.size));
      return new Response(null, { status: 200, headers });
    }
    if (!('body' in obj)) return new Response(null, { status: 304, headers }); // If-None-Match hit
    return new Response(obj.body, { status: 200, headers });
  },
};

function text(status, body, extra = {}) {
  return new Response(body + '\n', {
    status,
    headers: { 'content-type': 'text/plain; charset=utf-8', 'cache-control': 'no-store', ...extra },
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
export async function verifyAccessJwt(token, env, nowSeconds = Math.floor(Date.now() / 1000)) {
  try {
    const parts = token.split('.');
    if (parts.length !== 3) return false;
    const dec = new TextDecoder();
    const header = JSON.parse(dec.decode(b64urlDecode(parts[0])));
    const payload = JSON.parse(dec.decode(b64urlDecode(parts[1])));
    if (header.alg !== 'RS256' || !header.kid) return false;

    const team = env.ACCESS_TEAM_DOMAIN;
    let jwk = (await teamKeys(team, false)).find((k) => k.kid === header.kid);
    if (!jwk) jwk = (await teamKeys(team, true)).find((k) => k.kid === header.kid); // key rotation
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
    if (!aud.includes(env.ACCESS_AUD)) return false;
    if (payload.iss !== `https://${team}`) return false;
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
