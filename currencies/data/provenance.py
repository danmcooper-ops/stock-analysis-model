"""Data lineage recording.

Adapted from the sibling stock-analysis-model's provenance module. Records
which source served each data *slot* per currency, plus a run-level event log
for anything that degraded silently.

The governing rule, inherited from upstream: **every method swallows its own
exceptions**. A provenance bug must never be the reason a currency drops out
of the run.

Slots are coarse (per data family, not per field) because that is the
granularity at which a source actually fails: the ECB series either came back
or it didn't.
"""

import json
import os
import platform
from datetime import datetime, timezone

SCHEMA_VERSION = 1

# Cached macro data older than this is worth surfacing in the UI. World Bank
# indicators are annual, so 90 days is not alarming in itself — but a value
# older than that usually means the endpoint has been 502ing for a while.
STALE_CACHE_DAYS = 90

SLOTS = ('spot', 'reer', 'policy_rate', 'macro')

EVENT_TYPES = (
    'source_failed',      # endpoint returned nothing and no cache existed
    'stale_cache',        # served an expired cache entry after network failure
    'series_missing',     # source responded but had no series for this currency
    'short_history',      # series exists but is too short to score
    'indicator_failed',   # a single World Bank indicator is down
)


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


class ProvenanceRecorder:
    def __init__(self, run_date):
        self.run_date = run_date
        self._sources = {}   # {code: {slot: {...}}}
        self._events = []

    def record_source(self, code, slot, source, meta=None, **extra):
        """Record that `source` served `slot` for `code`.

        `meta` is the dict populated by cache.fetch, so a call site can pass
        it straight through without unpacking cache_hit/age itself.
        """
        try:
            entry = {'source': source, 'fetched_at': now_iso()}
            if meta:
                for k in ('cache_hit', 'cache_age_days', 'stale', 'attempts'):
                    if k in meta:
                        entry[k] = meta[k]
            entry.update(extra)
            self._sources.setdefault(code, {})[slot] = entry

            # Auto-raise a stale event so call sites don't each have to.
            if meta and meta.get('stale'):
                self.record_event('stale_cache', code, source,
                                  {'slot': slot,
                                   'cache_age_days': meta.get('cache_age_days')})
        except Exception:
            pass

    def record_event(self, etype, code=None, source=None, detail=None):
        try:
            self._events.append({
                'type': etype, 'code': code, 'source': source,
                'detail': detail, 'at': now_iso(),
            })
        except Exception:
            pass

    def currency_block(self, code):
        """Per-row provenance, attached to the result dict as `_provenance`."""
        try:
            return dict(self._sources.get(code, {}))
        except Exception:
            return {}

    def event_counts(self):
        try:
            out = {}
            for e in self._events:
                out[e['type']] = out.get(e['type'], 0) + 1
            return out
        except Exception:
            return {}

    def run_block(self, rows=None):
        """Run-level provenance for the results JSON and the report footer."""
        try:
            stale_slots = sum(
                1 for slots in self._sources.values()
                for s in slots.values() if s.get('stale')
            )
            return {
                'schema_version': SCHEMA_VERSION,
                'run_date': str(self.run_date),
                'generated_at': now_iso(),
                'python': platform.python_version(),
                'currencies': len(self._sources),
                'stale_slots': stale_slots,
                'event_counts': self.event_counts(),
            }
        except Exception:
            return {'schema_version': SCHEMA_VERSION}

    def write_events(self, output_dir):
        """Write the unbounded event log to a sidecar.

        Kept out of the committed results JSON deliberately: results are a
        daily snapshot series, the event log is diagnostic and unbounded.
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, 'events_%s.json' % self.run_date)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump({'run_date': str(self.run_date),
                           'events': self._events}, f, indent=1)
            return path
        except Exception:
            return None
