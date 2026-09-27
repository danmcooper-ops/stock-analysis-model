// Run with: node --test cloudflare/worker/test/
// Exercises the Worker with real RS256 tokens against a stubbed JWKS
// endpoint and an in-memory stand-in for the R2 binding.
import { test, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import worker, { verifyAccessJwt, _resetJwksCache } from '../src/index.js';

const TEAM = 'example.cloudflareaccess.com';
const AUD = 'aud-tag-123';
const env0 = { ACCESS_TEAM_DOMAIN: TEAM, ACCESS_AUD: AUD };

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

let good, rogue, jwksFetches;
beforeEach(async () => {
  _resetJwksCache();
  good ??= await keypair('k1');
  rogue ??= await keypair('k1'); // same kid, different key
  jwksFetches = 0;
  globalThis.fetch = async (url) => {
    assert.equal(url, `https://${TEAM}/cdn-cgi/access/certs`);
    jwksFetches++;
    return new Response(JSON.stringify({ keys: [good.jwk] }));
  };
});

function bucket(objects) {
  const mk = (key, withBody) => {
    const { body, type } = objects[key];
    const o = {
      size: body.length,
      httpEtag: `"etag-${key}"`,
      writeHttpMetadata: (h) => h.set('content-type', type),
    };
    if (withBody) o.body = body;
    return o;
  };
  return {
    async get(key, opts) {
      if (!(key in objects)) return null;
      const inm = opts?.onlyIf?.get?.('if-none-match');
      return mk(key, inm !== `"etag-${key}"`);
    },
    async head(key) { return key in objects ? mk(key, false) : null; },
  };
}

const site = {
  'index.html': { body: '<html>2026-09-26</html>', type: 'text/html; charset=utf-8' },
  'details.json': { body: '{}', type: 'application/json; charset=utf-8' },
  'px/AAPL.json': { body: '[1,2]', type: 'application/json; charset=utf-8' },
};
const env = () => ({ ...env0, BUCKET: bucket(site) });

async function get(path, token, extra = {}, method = 'GET') {
  const headers = { ...extra };
  if (token) headers['cf-access-jwt-assertion'] = token;
  return worker.fetch(new Request(`https://stock-report.x.workers.dev${path}`, { method, headers }), env());
}

test('valid token serves index at / with private no-cache', async () => {
  const r = await get('/', await sign(good, goodClaims()));
  assert.equal(r.status, 200);
  assert.match(await r.text(), /2026-09-26/);
  assert.equal(r.headers.get('cache-control'), 'private, no-cache');
  assert.match(r.headers.get('content-type'), /text\/html/);
});

test('shards get a short private max-age', async () => {
  const r = await get('/px/AAPL.json', await sign(good, goodClaims()));
  assert.equal(r.status, 200);
  assert.equal(r.headers.get('cache-control'), 'private, max-age=600');
});

test('no token -> 403', async () => {
  assert.equal((await get('/details.json')).status, 403);
});

test('rejects expired, wrong audience, wrong issuer, forged signature, alg none', async () => {
  const cases = [
    await sign(good, { ...goodClaims(), exp: now() - 10 }),
    await sign(good, { ...goodClaims(), aud: ['someone-else'] }),
    await sign(good, { ...goodClaims(), iss: 'https://evil.cloudflareaccess.com' }),
    await sign(rogue, goodClaims()),
    (await sign(good, goodClaims(), { alg: 'none' })),
    'not.a.jwt',
  ];
  for (const t of cases) assert.equal(await verifyAccessJwt(t, env0), false, t.slice(0, 40));
  assert.equal(await verifyAccessJwt(await sign(good, goodClaims()), env0), true);
});

test('unconfigured Worker fails closed with 500', async () => {
  const r = await worker.fetch(
    new Request('https://x/', { headers: { 'cf-access-jwt-assertion': await sign(good, goodClaims()) } }),
    { BUCKET: bucket(site), ACCESS_TEAM_DOMAIN: '', ACCESS_AUD: '' },
  );
  assert.equal(r.status, 500);
});

test('missing key 404, traversal 400, POST 405', async () => {
  const t = await sign(good, goodClaims());
  assert.equal((await get('/nope.json', t)).status, 404);
  assert.equal((await get('/px%2F..%2Fdetails.json', t)).status, 400);
  assert.equal((await get('/', t, {}, 'POST')).status, 405);
});

test('If-None-Match hit -> 304; HEAD has length and no body', async () => {
  const t = await sign(good, goodClaims());
  const r = await get('/details.json', t, { 'if-none-match': '"etag-details.json"' });
  assert.equal(r.status, 304);
  const h = await get('/details.json', t, {}, 'HEAD');
  assert.equal(h.status, 200);
  assert.equal(h.headers.get('content-length'), '2');
});

test('JWKS is cached across requests', async () => {
  const t = await sign(good, goodClaims());
  await get('/', t); await get('/details.json', t);
  assert.equal(jwksFetches, 1);
});
