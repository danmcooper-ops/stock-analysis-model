// Run with: cd cloudflare/pages && npm test
// Exercises the Pages Worker with real RS256 tokens against a stubbed JWKS
// endpoint and a stand-in for the Pages static-asset binding (env.ASSETS).
import { test, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import worker, { verifyAccessJwt, accessConfig, privateCacheControl, _resetJwksCache } from '../_worker.js';

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

let good, rogue, jwksFetches, assetRequests;
beforeEach(async () => {
  _resetJwksCache();
  good ??= await keypair('k1');
  rogue ??= await keypair('k1'); // same kid, different key
  jwksFetches = 0;
  assetRequests = [];
  globalThis.fetch = async (url) => {
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
