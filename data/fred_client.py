"""FRED data client: Treasury yields, credit spreads, and macro series.

Adapted from the bond-analysis repo's FRED client. Two access paths:

  1. Keyed JSON API (api.stlouisfed.org) when FRED_API_KEY is set —
     explicit start/end parameters and first-class rate limits.
  2. Keyless fredgraph.csv fallback — same data, slower and less reliable
     for multi-year backfills, but the dashboard works without any key.

Values are stored RAW, exactly as FRED publishes them. FRED mixes units
freely across series — percent (DGS10), index levels (CPIAUCSL), thousands
of persons (PAYEMS), millions of dollars (WALCL) — so unit interpretation
belongs to the consumer's per-series metadata, not to this client. (The
bond repo's client divides by 100 because its universe is all-percent;
doing that here would corrupt every non-percent series.)

Note on the ICE BofA spread series (BAML*): FRED caps their history at a
rolling ~3-year window regardless of key — that is ICE's licensing, not a
key tier. Percentile/statistics windows for those series must say so.

Requires FRED_API_KEY environment variable for the keyed path (or pass via
constructor). Falls back gracefully when unavailable. Uses only stdlib.
"""

import csv
import io
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date

API_URL = 'https://api.stlouisfed.org/fred/series/observations'
CSV_URL = 'https://fred.stlouisfed.org/graph/fredgraph.csv'

# ICE BofA option-adjusted spreads by rating bucket, in percent. These are
# OAS, not effective yield — the *EY-suffixed series are yields and would be
# wrong here by the entire level of the risk-free curve.
OAS_SERIES = {
    'AAA': 'BAMLC0A1CAAA',
    'AA': 'BAMLC0A2CAA',
    'A': 'BAMLC0A3CA',
    'BBB': 'BAMLC0A4CBBB',
    'BB': 'BAMLH0A1HYBB',
    'B': 'BAMLH0A2HYB',
    'CCC': 'BAMLH0A3HYC',
}

IG_ALL = 'BAMLC0A0CM'
HY_ALL = 'BAMLH0A0HYM2'

# Constant-maturity Treasury yields, in percent.
CMT_SERIES = {
    '1M': 'DGS1MO', '3M': 'DGS3MO', '6M': 'DGS6MO', '1Y': 'DGS1',
    '2Y': 'DGS2', '3Y': 'DGS3', '5Y': 'DGS5', '7Y': 'DGS7',
    '10Y': 'DGS10', '20Y': 'DGS20', '30Y': 'DGS30',
}


class FREDClient:
    """Fetches FRED series with a per-series on-disk JSON cache."""

    def __init__(self, api_key=None, cache_dir=None, max_age_days=1,
                 request_delay=0.15):
        self.api_key = api_key or os.environ.get('FRED_API_KEY', '') or None
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'cache', 'fred')
        self.max_age_days = max_age_days
        self._memo = {}
        self._delay = request_delay
        self._last_req = 0
        self.history_source = 'keyed' if self.api_key else 'keyless'
        if not self.api_key:
            print('FRED: no FRED_API_KEY — using the keyless fredgraph.csv '
                  'endpoint (same data, slower for backfills).')

    @property
    def available(self):
        """True when the keyed API path is usable (keyless still works)."""
        return bool(self.api_key)

    # -- fetching -----------------------------------------------------------

    def _throttle(self):
        elapsed = time.time() - self._last_req
        if elapsed < self._delay:
            time.sleep(self._delay - elapsed)
        self._last_req = time.time()

    def _get(self, url, params, timeout=15):
        self._throttle()
        full = f'{url}?{urllib.parse.urlencode(params)}'
        try:
            req = urllib.request.Request(
                full, headers={'User-Agent': 'StockAnalyzer/1.0'})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode('utf-8', 'replace')
        except Exception:
            return None

    def _cache_path(self, series_id):
        return os.path.join(self.cache_dir, f'{series_id}.json')

    def _load_cache(self, series_id):
        path = self._cache_path(series_id)
        if not os.path.exists(path):
            return None
        try:
            age = (date.today()
                   - date.fromtimestamp(os.path.getmtime(path))).days
            if age >= self.max_age_days:
                return None
            with open(path, encoding='utf-8') as fh:
                raw = json.load(fh)
            return {date.fromisoformat(k): v for k, v in raw['obs'].items()}
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _save_cache(self, series_id, obs):
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            path = self._cache_path(series_id)
            tmp = f'{path}.tmp.{os.getpid()}'
            with open(tmp, 'w', encoding='utf-8') as fh:
                json.dump({'source': self.history_source,
                           'obs': {k.isoformat(): v for k, v in obs.items()}},
                          fh)
            os.replace(tmp, path)
        except OSError:
            pass

    def _fetch_keyed(self, series_id, start=None, end=None):
        params = {'series_id': series_id, 'api_key': self.api_key,
                  'file_type': 'json'}
        if start:
            params['observation_start'] = start.isoformat()
        if end:
            params['observation_end'] = end.isoformat()
        text = self._get(API_URL, params)
        if not text:
            return None
        try:
            payload = json.loads(text)
        except ValueError:
            return None
        if 'observations' not in payload:
            return None
        out = {}
        for row in payload['observations']:
            raw = row.get('value')
            if raw in (None, '', '.'):        # FRED marks holidays with '.'
                continue
            try:
                out[date.fromisoformat(row['date'])] = float(raw)
            except (ValueError, KeyError):
                continue
        return out

    def _fetch_keyless(self, series_id, start=None):
        params = {'id': series_id}
        if start:
            params['cosd'] = start.isoformat()
        text = self._get(CSV_URL, params)
        if not text or 'observation_date' not in text[:200]:
            return None
        out = {}
        for row in csv.DictReader(io.StringIO(text)):
            raw_date = row.get('observation_date')
            raw_val = row.get(series_id)
            if not raw_date or raw_val in (None, '', '.'):
                continue
            try:
                out[date.fromisoformat(raw_date)] = float(raw_val)
            except ValueError:
                continue
        return out

    def fetch_series(self, series_id, start=None, end=None, force=False):
        """Return {date: raw value} for a FRED series. {} on failure."""
        if not force and series_id in self._memo:
            return self._memo[series_id]
        if not force:
            cached = self._load_cache(series_id)
            if cached is not None:
                self._memo[series_id] = cached
                return cached

        obs = None
        if self.api_key:
            obs = self._fetch_keyed(series_id, start=start, end=end)
        if obs is None:
            obs = self._fetch_keyless(series_id, start=start)

        if not obs:
            print(f'FRED: no observations for {series_id}')
            self._memo[series_id] = {}
            return {}

        self._memo[series_id] = obs
        self._save_cache(series_id, obs)
        return obs

    # -- convenience --------------------------------------------------------

    @staticmethod
    def _as_of_value(obs, as_of, max_lookback_days=10):
        """Most recent observation at or before as_of. FRED lags a day or two
        and skips holidays, so an exact-date lookup would fail routinely."""
        if not obs:
            return None, None
        target = as_of or date.today()
        candidates = [d for d in obs if d <= target
                      and (target - d).days <= max_lookback_days]
        if not candidates:
            return None, None
        best = max(candidates)
        return best, obs[best]

    def fetch_bucket_oas(self, as_of=None):
        """{bucket: OAS percent} for the seven rating buckets."""
        out = {}
        for bucket, series_id in OAS_SERIES.items():
            _, value = self._as_of_value(self.fetch_series(series_id), as_of)
            if value is not None:
                out[bucket] = value
        return out

    def fetch_cmt_curve(self, as_of=None, with_dates=False):
        """Constant-maturity Treasury curve as {tenor: percent}."""
        out = {}
        for tenor, series_id in CMT_SERIES.items():
            obs = self.fetch_series(series_id)
            obs_date, value = self._as_of_value(obs, as_of)
            if value is not None:
                out[tenor] = (obs_date, value) if with_dates else value
        return out

    def coverage(self):
        """{series: {first, last, n}} for everything fetched this session."""
        return {sid: {'first': min(obs).isoformat(),
                      'last': max(obs).isoformat(), 'n': len(obs)}
                for sid, obs in self._memo.items() if obs}
