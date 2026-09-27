// P5 public-load test (design/supabase-migration.md, scalability check 3).
//
// Drives the published report on Cloudflare Pages: 500 req/s for 30 minutes,
// then a 2,000 req/s burst, over the same files a browser fetches (the page,
// its sidecars and the per-ticker shards). Postgres is not in this path at
// all; the check is that the CDN serves it fast and from cache.
//
//   k6 run -e BASE_URL=https://<project>.pages.dev tests/load/k6_public.js
//   k6 run -e BASE_URL=... -e SOAK=2m -e BURST=30s -e RATE=50 -e BURST_RATE=200 tests/load/k6_public.js  # smoke
//
// Thresholds (the plan's pass line): p95 < 100 ms, errors < 0.1%, and a
// cache hit rate > 95% read from cf-cache-status (HIT / REVALIDATED count as
// cached: the site keeps Pages' default revalidation, see P4c).
import http from 'k6/http';
import { check } from 'k6';
import { Rate } from 'k6/metrics';

const BASE = (__ENV.BASE_URL || '').replace(/\/$/, '');
const SOAK = __ENV.SOAK || '30m';
const BURST = __ENV.BURST || '2m';
const RATE = Number(__ENV.RATE || 500);
const BURST_RATE = Number(__ENV.BURST_RATE || 2000);
const cacheHit = new Rate('cdn_cache_hit');

export const options = {
  discardResponseBodies: true,
  scenarios: {
    soak: {
      executor: 'constant-arrival-rate', rate: RATE, timeUnit: '1s', duration: SOAK,
      preAllocatedVUs: 200, maxVUs: 1000,
    },
    burst: {
      executor: 'constant-arrival-rate', rate: BURST_RATE, timeUnit: '1s', duration: BURST,
      startTime: SOAK, preAllocatedVUs: 800, maxVUs: 4000,
    },
  },
  thresholds: {
    http_req_duration: ['p(95)<100'],
    http_req_failed: ['rate<0.001'],
    cdn_cache_hit: ['rate>0.95'],
  },
};

export function setup() {
  if (!BASE) throw new Error('set BASE_URL');
  const idx = http.get(`${BASE}/hist_index.json`, { responseType: 'text' });
  const meta = http.get(`${BASE}/prices_meta.json`, { responseType: 'text' });
  const hist = idx.status === 200 ? JSON.parse(idx.body).tickers : [];
  const px = meta.status === 200 ? (JSON.parse(meta.body).manifest || []) : [];
  if (!hist.length || !px.length) throw new Error(`no shard manifests at ${BASE}`);
  return { hist, px };
}

function pick(xs) { return xs[Math.floor(Math.random() * xs.length)]; }

// Roughly what a session asks for: the page and its small sidecars once, then
// a few shards per company opened.
export default function (data) {
  const r = Math.random();
  let path;
  if (r < 0.05) path = '/';
  else if (r < 0.10) path = '/prices_meta.json';
  else if (r < 0.12) path = '/hist_index.json';
  else if (r < 0.55) path = `/px/${encodeURIComponent(pick(data.px))}.json`;
  else path = `/hist/${encodeURIComponent(pick(data.hist))}.json`;
  const res = http.get(BASE + path, { tags: { kind: path.split('/')[1] || 'page' } });
  const cf = (res.headers['Cf-Cache-Status'] || res.headers['cf-cache-status'] || '').toUpperCase();
  cacheHit.add(cf === 'HIT' || cf === 'REVALIDATED');
  check(res, { 'status 200': (x) => x.status === 200 });
}
