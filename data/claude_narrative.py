"""Claude-generated macro narrative for the Macro Outlook tab.

Serializes the macro.json sidecar's numeric facts (regime model output,
FRED series with changes/percentiles, the Treasury curve, credit spreads,
and per-sector ETF momentum) into a prompt for the Claude API and returns
a structured narrative: economy-wide paragraphs, headwind/tailwind bullets,
and one outlook per GICS sector. The LLM call is network I/O, so this
lives in data/ rather than models/.

Fails soft everywhere: missing ANTHROPIC_API_KEY, the `anthropic` package
not installed, API errors, refusals, truncation, or unparseable output all
log a warning and return None — the dashboard simply renders without prose.
Results are cached per as_of date on disk (data/cache/claude_narrative/) so
the daily re-render (scripts/rescore_and_render.py) never re-pays for a day
the main run already generated.
"""

import json
import logging
import os
from datetime import datetime, timezone

from models.narrative import _SECTOR_MACRO_DRIVERS

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'claude-opus-5'
# Headroom, not a target. The grammar cannot pin array lengths, so output
# size is model-determined: an 11-sector reply runs ~1.9k tokens, but a
# 2026-08-31 run returned 26 sector entries at 4,959 tokens. At 6000 that
# draw tripped stop_reason='max_tokens' and the whole narrative was
# discarded, silently un-shipping the card. Cost is per-use, not per-cap.
DEFAULT_MAX_TOKENS = 12000

# Bumped whenever the narrative's shape changes. The day cache is keyed by
# as_of alone and a hit short-circuits every post-parse check below, so
# without this a run on the day of a shape change replays yesterday's shape
# — bullet-less sector sections — until the date rolls over.
SCHEMA_VERSION = 2

# Advisory bullet band per sector; the ceiling is enforced in
# _clamp_sector_bullets, the floor is only ever counted (never padded).
MIN_SECTOR_BULLETS = 3
MAX_SECTOR_BULLETS = 5

# The 11 GICS sectors under the yfinance naming this repo uses everywhere
# (rows, SECTOR_CONFIG, sector ETF maps). The narrative must cover all 11.
GICS_SECTORS = [
    'Technology', 'Financial Services', 'Healthcare', 'Consumer Cyclical',
    'Consumer Defensive', 'Communication Services', 'Industrials', 'Energy',
    'Basic Materials', 'Utilities', 'Real Estate',
]

# Structured-output schema: the API guarantees the response validates, so
# the render side can trust the shape (content strings still get escaped).
# Array lengths are NOT pinned here — the structured-outputs grammar rejects
# minItems other than 0/1 (400 invalid_request_error, seen live 2026-08-31),
# so counts are enforced by the prompt and checked post-parse in generate().
NARRATIVE_SCHEMA = {
    'type': 'object',
    'properties': {
        # Three named paragraphs rather than an array: the grammar CAN pin a
        # fixed set of required object keys, which is how "exactly 3" is
        # actually enforced (the prompt alone was ignored — a live run
        # returned 5). generate() flattens them to the list the page renders.
        'paragraphs': {
            'type': 'object',
            'properties': {
                'growth_labor': {'type': 'string'},
                'inflation_rates': {'type': 'string'},
                'credit_conditions': {'type': 'string'},
            },
            'required': ['growth_labor', 'inflation_rates',
                         'credit_conditions'],
            'additionalProperties': False,
        },
        'headwinds': {'type': 'array', 'items': {'type': 'string'}},
        'tailwinds': {'type': 'array', 'items': {'type': 'string'}},
        'sectors': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'sector': {'type': 'string', 'enum': GICS_SECTORS},
                    'stance': {'type': 'string',
                               'enum': ['tailwind', 'neutral', 'headwind']},
                    'headline': {'type': 'string', 'maxLength': 60},
                    'outlook': {'type': 'string'},
                    # Per-sector bullets, distinct from the economy-wide
                    # arrays above. Same no-length-pinning rule applies —
                    # and it applies to these NESTED arrays too, which is
                    # the easy place to forget it.
                    'tailwinds': {'type': 'array',
                                  'items': {'type': 'string'}},
                    'headwinds': {'type': 'array',
                                  'items': {'type': 'string'}},
                },
                'required': ['sector', 'stance', 'headline', 'outlook',
                             'tailwinds', 'headwinds'],
                'additionalProperties': False,
            },
        },
    },
    'required': ['paragraphs', 'headwinds', 'tailwinds', 'sectors'],
    'additionalProperties': False,
}

SYSTEM_PROMPT = (
    'You are the macro strategist for a value-investing equity research '
    'report. Write a narrative assessment of the US economy from the '
    'indicator data provided: regime model output, FRED series with recent '
    'changes and historical percentiles, the Treasury yield curve, credit '
    'spreads by rating bucket, and sector ETF momentum.\n'
    'Rules:\n'
    '- Use ONLY the numbers provided, and cite specific figures inline '
    "(e.g. 'core PCE at 2.8%'). Never invent a data point.\n"
    '- Declarative plain-English prose for a long-horizon value investor; '
    'no hedging boilerplate, no first person, no investment advice.\n'
    '- paragraphs: three named paragraphs — growth_labor, inflation_rates, '
    'credit_conditions — each a single flowing paragraph.\n'
    '- headwinds / tailwinds: AT MOST 5 of each — only the sharpest '
    'economy-wide risks and supports, one clause each.\n'
    '- sectors: one entry for EVERY GICS sector listed in the data (all '
    '11, including any without ETF metrics). Style: The Economist — pithy '
    'but dense with information. For each sector write:\n'
    '  headline: a 3-6 word kicker leading the entry, wordplay in the '
    "paper's tradition (e.g. 'Banks bank the curve', 'Rates tax the "
    "growth premium'); sentence case, no terminal period.\n"
    '  outlook: ONE declarative active-voice sentence, 25 words maximum, '
    'in which every clause carries a figure from the data (an ETF return '
    'or relative strength, a yield, a spread, an indicator level), tying '
    "the sector's macro sensitivities (rate sensitivity, cyclicality, "
    'commodity linkage, defensiveness) to those numbers. Dry wit is '
    'welcome; filler and hedging are not — never write "may", "could", '
    '"likely", "remains to be seen", or "bears watching".\n'
    '  tailwinds / headwinds: the macro forces acting on THIS sector, '
    'split by direction — not the economy-wide headwinds/tailwinds above, '
    'which are a separate top-level field. Write 3 to 5 bullets in TOTAL '
    'across the two lists (not 3-5 each), putting at least one on each '
    'side when the data supports it. Never pad to reach three: a sector '
    'the data pushes one way gets a lopsided split, and a bullet with no '
    'figure behind it should not exist.\n'
    '    Each bullet: ONE clause, 20 words maximum, naming its indicator '
    "and its figure — 'core PCE at 2.8%, 71st percentile of 10 years', "
    "'BBB spreads 18bp wider in a month'. Same ban on filler and hedging "
    'as outlook.\n'
    "    Spread a sector's bullets across DIFFERENT indicator families — "
    "each series carries a 'sec' key (rates, inflation, growth, credit, "
    'housing) and the yield curve, the OAS buckets by rating, the regime '
    "scores and the sector's own ETF relative strength are all fair game. "
    'Four restatements of the 10-year yield is the failure to avoid.\n'
    "    Each series also carries 'good' — the direction the report treats "
    'as favourable for the economy. Use it as a starting point, not the '
    "answer: the bullet belongs in the list matching THIS sector's "
    'exposure. A rising 10-year is a tailwind for banks and a headwind '
    'for utilities and REITs.\n'
    '  stance: the net read across those bullets — tailwind, neutral, or '
    'headwind.'
)

# Per-series keys worth showing the model; 'hist' (hundreds of points per
# series) is deliberately excluded to keep the prompt a few thousand tokens.
# 'good' and 'sec' are cheap and load-bearing for the per-sector bullets:
# 'good' is the repo's own view of which direction is favourable — headwind
# vs tailwind polarity, stated rather than inferred from the label — and
# 'sec' is the indicator family (rates / inflation / growth / credit /
# housing), which is what lets the model spread a sector's bullets across
# families instead of restating one rate four times.
_SERIES_FACT_KEYS = ('l', 'sec', 'latest', 'chg_1m', 'chg_1y', 'pctile',
                     'pct_win', 'z', 'suffix', 'good')


def _driver_for(sector):
    """Static macro sensitivities for a sector, tolerating the legacy
    'Financials' key in _SECTOR_MACRO_DRIVERS."""
    if sector == 'Financial Services':
        return (_SECTOR_MACRO_DRIVERS.get('Financial Services')
                or _SECTOR_MACRO_DRIVERS.get('Financials') or {})
    return _SECTOR_MACRO_DRIVERS.get(sector, {})


def build_macro_facts(sidecar):
    """Compact, prompt-ready facts dict from a macro.json sidecar. Pure."""
    sidecar = sidecar or {}
    series = sidecar.get('series') or {}
    facts_series = {}
    for sid, s in series.items():
        entry = {k: s[k] for k in _SERIES_FACT_KEYS if s.get(k) not in (None, '')}
        if entry:
            facts_series[sid] = entry

    curve = sidecar.get('curve') or None
    if curve:
        curve = {k: curve[k] for k in ('tenors', 'now', 'm1', 'y1')
                 if curve.get(k)}

    sector_data = sidecar.get('sector_data') or {}
    sectors = {}
    for sector in GICS_SECTORS:
        entry = dict(sector_data.get(sector) or {})
        drivers = _driver_for(sector)
        if drivers:
            entry['macro_sensitivities'] = drivers
        sectors[sector] = entry

    return {
        'as_of': sidecar.get('as_of'),
        'regime': sidecar.get('regime'),
        'series': facts_series,
        'yield_curve': curve,
        'credit_oas_by_rating': sidecar.get('oas_buckets'),
        'sectors': sectors,
    }


def _clamp_sector_bullets(sectors):
    """Hold each sector to at most MAX_SECTOR_BULLETS bullets in total.

    The grammar pins no array length at any depth, so the prompt's "3 to 5
    in total" is advisory in exactly the way the all-11-sectors rule is.
    Trim from the longer list first so a 6-1 draw comes back 4-1 rather
    than losing the lone bullet on the other side. Under-filled sectors are
    left alone and merely counted: padding would mean inventing a bullet
    with no figure behind it, which is the one thing the prompt forbids.
    Mutates in place; pure otherwise.
    """
    short = 0
    for entry in sectors or []:
        if not isinstance(entry, dict):
            continue
        tw = [b for b in (entry.get('tailwinds') or []) if b]
        hw = [b for b in (entry.get('headwinds') or []) if b]
        while len(tw) + len(hw) > MAX_SECTOR_BULLETS:
            (tw if len(tw) >= len(hw) else hw).pop()
        entry['tailwinds'], entry['headwinds'] = tw, hw
        if len(tw) + len(hw) < MIN_SECTOR_BULLETS:
            short += 1
    if short:
        logger.warning('macro narrative: %d sectors under %d bullets',
                       short, MIN_SECTOR_BULLETS)


class ClaudeNarrativeClient:
    """Generates the macro narrative via the Claude API, with a per-day
    on-disk cache (same pattern as FREDClient's per-series cache)."""

    def __init__(self, api_key=None, cache_dir=None, model=None,
                 max_tokens=None):
        self.api_key = (api_key or os.environ.get('ANTHROPIC_API_KEY', '')
                        or None)
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'cache',
            'claude_narrative')
        self.model = model or DEFAULT_MODEL
        self.max_tokens = max_tokens or DEFAULT_MAX_TOKENS

    @property
    def available(self):
        return bool(self.api_key)

    # -- cache ---------------------------------------------------------------

    def _cache_path(self, as_of):
        return os.path.join(self.cache_dir, f'{as_of}.json')

    def _read_cache(self, as_of):
        try:
            with open(self._cache_path(as_of), encoding='utf-8') as fh:
                cached = json.load(fh)
        except (OSError, ValueError):
            return None
        # A cached narrative written under an older shape is unusable: the
        # hit below short-circuits every post-parse check, so it would be
        # served straight to the page missing whatever the new shape added.
        # Treat a version mismatch as a miss and regenerate.
        if not isinstance(cached, dict) or not cached.get('paragraphs'):
            return None
        if cached.get('schema_version') != SCHEMA_VERSION:
            logger.info('macro narrative cache for %s is schema v%s, want '
                        'v%s — regenerating', as_of,
                        cached.get('schema_version'), SCHEMA_VERSION)
            return None
        return cached

    def _write_cache(self, as_of, narrative):
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(self._cache_path(as_of), 'w', encoding='utf-8') as fh:
                json.dump(narrative, fh)
        except OSError as e:
            logger.debug('macro narrative cache write failed: %s', e)

    # -- generation ----------------------------------------------------------

    def generate(self, sidecar):
        """Narrative dict for a sidecar, or None when unavailable/failed."""
        as_of = (sidecar or {}).get('as_of')
        if not as_of:
            return None
        cached = self._read_cache(as_of)
        if cached:
            logger.info('macro narrative: cache hit for %s', as_of)
            return cached
        if not self.available:
            logger.warning('macro narrative skipped: no ANTHROPIC_API_KEY')
            return None
        try:
            import anthropic
        except ImportError:
            logger.warning("macro narrative skipped: `anthropic` package "
                           'not installed')
            return None

        facts = build_macro_facts(sidecar)
        user_msg = (f'Today is {as_of}. Macro data (JSON):\n'
                    + json.dumps(facts, sort_keys=True))
        try:
            response = anthropic.Anthropic(api_key=self.api_key).messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                output_config={'format': {'type': 'json_schema',
                                          'schema': NARRATIVE_SCHEMA}},
                messages=[{'role': 'user', 'content': user_msg}],
            )
        except anthropic.RateLimitError as e:
            logger.warning('macro narrative skipped: rate limited (%s)', e)
            return None
        except anthropic.APIStatusError as e:
            logger.warning('macro narrative skipped: API error %s (%s)',
                           getattr(e, 'status_code', '?'), e)
            return None
        except anthropic.APIConnectionError as e:
            logger.warning('macro narrative skipped: connection error (%s)', e)
            return None

        if response.stop_reason == 'refusal':
            logger.warning('macro narrative skipped: model refused')
            return None
        if response.stop_reason == 'max_tokens':
            logger.warning('macro narrative skipped: output truncated at '
                           '%d tokens', self.max_tokens)
            return None

        text = next((b.text for b in response.content
                     if getattr(b, 'type', None) == 'text'), None)
        try:
            narrative = json.loads(text)
        except (TypeError, ValueError) as e:
            logger.warning('macro narrative skipped: unparseable response '
                           '(%s)', e)
            return None
        if isinstance(narrative, dict) and \
                isinstance(narrative.get('paragraphs'), dict):
            p = narrative['paragraphs']
            narrative['paragraphs'] = [p[k] for k in
                                       ('growth_labor', 'inflation_rates',
                                        'credit_conditions') if p.get(k)]
        if not isinstance(narrative, dict) or not narrative.get('paragraphs'):
            logger.warning('macro narrative skipped: empty response')
            return None
        # The grammar pins no array length (minItems>1 AND maxItems both 400
        # live, verified 2026-09-01), so the model may repeat sectors — one
        # run returned 26 entries spanning the 11 names. Keep the first
        # outlook per sector, in canonical GICS order, so the page renders one
        # card each instead of duplicates; then police the count as before.
        raw_sectors = narrative.get('sectors') or []
        first_by_sector = {}
        for entry in raw_sectors:
            name = (entry or {}).get('sector')
            if name and name not in first_by_sector:
                first_by_sector[name] = entry
        if len(first_by_sector) != len(raw_sectors):
            logger.warning('macro narrative: %d sector entries collapsed to '
                           '%d unique', len(raw_sectors), len(first_by_sector))
        narrative['sectors'] = [first_by_sector[name] for name in GICS_SECTORS
                                if name in first_by_sector]
        n_sectors = len(narrative['sectors'])
        if n_sectors != len(GICS_SECTORS):
            logger.warning('macro narrative: %d sector outlooks (expected %d)',
                           n_sectors, len(GICS_SECTORS))
        _clamp_sector_bullets(narrative['sectors'])

        narrative['model'] = self.model
        narrative['generated_at'] = datetime.now(timezone.utc).isoformat()
        narrative['schema_version'] = SCHEMA_VERSION
        self._write_cache(as_of, narrative)
        logger.info('macro narrative: generated for %s (%d sectors)',
                    as_of, len(narrative.get('sectors') or []))
        return narrative
