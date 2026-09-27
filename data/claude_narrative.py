"""Claude-generated macro narrative for the Macro Outlook tab.

Serializes the macro.json sidecar's numeric facts (regime model output,
FRED series with changes/percentiles, the Treasury curve, credit spreads,
and per-sector ETF momentum) into a prompt for the Claude API and returns
a structured narrative: economy-wide paragraphs, headwind/tailwind bullets,
and one entry per GICS sector — an `influence` paragraph, a kicker and a
stance — which the Overview's "Key sector influences" section sets as
full prose. The LLM call is network I/O, so this lives in data/ rather
than models/.

The key is read from MACRO_ANTHROPIC_API_KEY, falling back to
ANTHROPIC_API_KEY. The cloud container that runs the nightly routine is a
Claude Code session, which owns the ANTHROPIC_* namespace for its own auth
(ANTHROPIC_BASE_URL arrives pre-set, and an ANTHROPIC_API_KEY configured on
the environment never reached the container over four consecutive nightly
runs, 2026-09-15..18, while every other key did). MACRO_ANTHROPIC_API_KEY is
a name the platform has no claim on; the fallback keeps the Mac runbook's
.env and any existing shell export working unchanged.

Fails soft everywhere: no key, the `anthropic` package
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
# The cap also covers the model's adaptive thinking, which is the bigger
# share: the first schema-v3 run (per-sector influence paragraphs) spent
# 10.3k thinking + ~6k text = 16.3k and was truncated at a 16k cap.
# Past ~21k the SDK refuses a non-streaming request unless it is given an
# explicit timeout, hence REQUEST_TIMEOUT_S below.
DEFAULT_MAX_TOKENS = 32000
REQUEST_TIMEOUT_S = 900

# Bumped whenever the narrative's shape changes. The day cache is keyed by
# as_of alone and a hit short-circuits every post-parse check below, so
# without this a run on the day of a shape change replays yesterday's shape
# — bullet-less sector sections — until the date rolls over.
# v3: per-sector `influence` paragraph; tighter paragraph rules.
# v4: the Overview cut to a ~750-word budget.
# v5: per-sector outlook and bullets dropped (nothing renders them since
#     the sector tabs lost their Macro Outlook section).
SCHEMA_VERSION = 5

# Advisory floor for a sector's influence paragraph, only ever counted:
# under it the Overview is back to reading like a kicker, which is what
# the paragraph replaced.
MIN_INFLUENCE_WORDS = 20

# Target length of the Overview read (paragraphs + economy-wide winds +
# sector headlines and influences), ~3 minutes. The per-field ceilings in
# the prompt sum to it: 3 x (20 + 3 x 20) + 8 x 12 + 11 x (4 + 40) ~= 820
# at the ceilings, ~750 as drawn. Counted post-parse, never enforced.
OVERVIEW_WORD_BUDGET = 750

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
_PARAGRAPH_SCHEMA = {
    'type': 'object',
    'properties': {
        'lead': {'type': 'string'},
        'points': {'type': 'array', 'items': {'type': 'string'}},
    },
    'required': ['lead', 'points'],
    'additionalProperties': False,
}
PARAGRAPH_KEYS = ('growth_labor', 'inflation_rates', 'credit_conditions')

NARRATIVE_SCHEMA = {
    'type': 'object',
    'properties': {
        # Three named paragraphs rather than an array: the grammar CAN pin a
        # fixed set of required object keys, which is how "exactly 3" is
        # actually enforced (the prompt alone was ignored — a live run
        # returned 5). generate() flattens them to the list the page renders.
        # Each is a lead plus its supporting points rather than one string:
        # the first v3 run, asked in prose for "a lead then 4 to 6
        # sentences", returned the lead alone for all three.
        'paragraphs': {
            'type': 'object',
            'properties': {
                'growth_labor': _PARAGRAPH_SCHEMA,
                'inflation_rates': _PARAGRAPH_SCHEMA,
                'credit_conditions': _PARAGRAPH_SCHEMA,
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
                # Generated in this order: the influence paragraph first,
                # then the kicker and the stance that summarise it, so
                # neither is committed to before the argument is made.
                'properties': {
                    'sector': {'type': 'string', 'enum': GICS_SECTORS},
                    'influence': {'type': 'string'},
                    'headline': {'type': 'string', 'maxLength': 60},
                    'stance': {'type': 'string',
                               'enum': ['tailwind', 'neutral', 'headwind']},
                },
                'required': ['sector', 'influence', 'headline', 'stance'],
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
    'no hedging boilerplate, no first person, no investment advice. Never '
    'write "may", "could", "likely", "remains to be seen", or "bears '
    'watching".\n'
    '- Figures: round for a reader, not a terminal — at most two decimals '
    "('3.35%', not '3.353%'), whole-number ordinal percentiles ('2nd "
    "percentile', never '1.5th'), basis points as 'bp', and name each "
    "series the way a reader knows it ('the 10-year Treasury yield', not "
    "'DGS10'). Give a percentile its window once ('71st percentile of 10 "
    "years') rather than on every mention.\n"
    '- Length: the Overview prints the paragraphs, the economy-wide '
    'headwinds/tailwinds, and every sector headline and influence, and '
    f'that read must total about {OVERVIEW_WORD_BUDGET} words. The word '
    'limits below add up to that budget; treat each as a ceiling and '
    'spend the words on meaning, not on more figures.\n'
    '- paragraphs: three named sections — growth_labor, inflation_rates, '
    'credit_conditions — each a lead and its points. The page sets the '
    'lead as a paragraph and each point as a bullet under it.\n'
    '  lead: ONE sentence, the verdict on that part of the economy, 20 '
    'words maximum, stated plainly with at most one figure.\n'
    '  points: exactly 3 supporting sentences, each ONE complete sentence '
    'ending in a full stop, one point in 20 words or fewer, carrying at '
    'most two figures, and each saying what its figure '
    "means — 'claims of 197,000, the 2nd percentile of a decade, show "
    "employers still hoarding workers' — rather than listing more numbers. "
    'No semicolon chains, no parenthetical asides, no dashes stacking a '
    'second clause of figures onto the first, no mixed metaphors.\n'
    '  Each sentence must stand alone as a bullet: no sentence that opens '
    "with 'This', 'That' or 'It' pointing back at the one before.\n"
    '- headwinds / tailwinds: AT MOST 4 of each — only the sharpest '
    'economy-wide risks and supports. Each one clause, 12 words maximum, '
    "in the form indicator, figure, consequence: 'initial claims at "
    "197,000, the tightest in a decade, keep incomes growing'. No two "
    'items on the same indicator.\n'
    '- sectors: one entry for EVERY GICS sector listed in the data (all '
    '11, including any without ETF metrics), each written in this order:\n'
    '  influence: the read the Overview prints under "Key sector '
    'influences", where it is the only thing a reader sees about the '
    'sector. 2 or 3 complete sentences, 30 to 40 words, plain declarative '
    'prose rather than wordplay. Explain the MECHANISM: the one or two '
    'macro forces that matter most for this sector now and the channel '
    'each works through (financing costs, consumer or business demand, '
    'input and commodity prices, pricing power and margins, or the '
    'discount rate on long-dated earnings), then the net effect as the '
    "sector ETF has priced it. Cite each force's indicator and figure, "
    'and explain it rather than list it.\n'
    '    When two forces matter, draw them from DIFFERENT indicator '
    "families — each series carries a 'sec' key (rates, inflation, "
    'growth, credit, housing), and the yield curve, the OAS buckets by '
    "rating, the regime scores and the sector's own ETF relative strength "
    'are all fair game.\n'
    "    Each series also carries 'good' — the direction the report treats "
    'as favourable for the economy. Use it as a starting point, not the '
    "answer: judge each force by THIS sector's exposure. A rising 10-year "
    'helps banks and hurts utilities and REITs.\n'
    '  headline: a 3-6 word kicker for that paragraph, wordplay in The '
    "Economist's tradition (e.g. 'Banks bank the curve', 'Rates tax the "
    "growth premium'); sentence case, no terminal period.\n"
    '  stance: the net effect the influence describes — tailwind, '
    'neutral, or headwind.'
)

# Per-series keys worth showing the model; 'hist' (hundreds of points per
# series) is deliberately excluded to keep the prompt a few thousand tokens.
# 'good' and 'sec' are cheap and load-bearing for the per-sector reads:
# 'good' is the repo's own view of which direction is favourable — headwind
# vs tailwind polarity, stated rather than inferred from the label — and
# 'sec' is the indicator family (rates / inflation / growth / credit /
# housing), which is what lets the model draw a sector's forces from
# different families instead of restating one rate twice.
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


def overview_word_count(narrative):
    """Words the Overview prints — the same fields the page's reading-time
    footer counts. Pure."""
    n = narrative or {}

    def wc(x):
        return len(str(x).split()) if x else 0
    total = sum(wc(x) for k in ('paragraphs', 'tailwinds', 'headwinds')
                for x in (n.get(k) or []))
    for e in n.get('sectors') or []:
        if isinstance(e, dict):
            total += wc(e.get('headline')) + wc(e.get('influence'))
    return total


def _flatten_paragraph(p):
    """One paragraph string from the schema's {lead, points}: the page
    splits it back into lead + bullets by sentence (_macSentences), which
    is also how every cached narrative before this shape renders. Each
    piece gets a terminal stop so the split lands between them. A bare
    string passes through."""
    if isinstance(p, str):
        return p.strip()
    if not isinstance(p, dict):
        return ''
    parts = [str(x).strip() for x in [p.get('lead')] + list(p.get('points')
                                                            or []) if x]
    parts = [x if x[-1] in '.!?' else x + '.' for x in parts if x]
    if len(parts) < 3:
        logger.warning('macro narrative: a paragraph came back with %d '
                       'sentences', len(parts))
    return ' '.join(parts)


class ClaudeNarrativeClient:
    """Generates the macro narrative via the Claude API, with a per-day
    on-disk cache (same pattern as FREDClient's per-series cache)."""

    def __init__(self, api_key=None, cache_dir=None, model=None,
                 max_tokens=None):
        # MACRO_ANTHROPIC_API_KEY first, ANTHROPIC_API_KEY second (see the
        # module docstring: the cloud routine cannot receive the latter).
        self.api_key = (api_key
                        or os.environ.get('MACRO_ANTHROPIC_API_KEY', '')
                        or os.environ.get('ANTHROPIC_API_KEY', '')
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
            logger.warning('macro narrative skipped: no '
                           'MACRO_ANTHROPIC_API_KEY (or ANTHROPIC_API_KEY)')
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
                timeout=REQUEST_TIMEOUT_S,
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
            narrative['paragraphs'] = [t for t in (_flatten_paragraph(p.get(k))
                                                   for k in PARAGRAPH_KEYS)
                                       if t]
        if not isinstance(narrative, dict) or not narrative.get('paragraphs'):
            logger.warning('macro narrative skipped: empty response')
            return None
        # The grammar pins no array length (minItems>1 AND maxItems both 400
        # live, verified 2026-09-01), so the model may repeat sectors — one
        # run returned 26 entries spanning the 11 names. Keep the first
        # entry per sector, in canonical GICS order, so the page renders one
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
            logger.warning('macro narrative: %d sector entries (expected %d)',
                           n_sectors, len(GICS_SECTORS))
        thin = sum(1 for e in narrative['sectors']
                   if len(str((e or {}).get('influence') or '').split())
                   < MIN_INFLUENCE_WORDS)
        if thin:
            logger.warning('macro narrative: %d sector influences under %d '
                           'words', thin, MIN_INFLUENCE_WORDS)
        words = overview_word_count(narrative)
        log = logger.warning if words > 1.25 * OVERVIEW_WORD_BUDGET \
            else logger.info
        log('macro narrative: overview is %d words (budget %d)', words,
            OVERVIEW_WORD_BUDGET)

        narrative['model'] = self.model
        narrative['generated_at'] = datetime.now(timezone.utc).isoformat()
        narrative['schema_version'] = SCHEMA_VERSION
        self._write_cache(as_of, narrative)
        logger.info('macro narrative: generated for %s (%d sectors)',
                    as_of, len(narrative.get('sectors') or []))
        return narrative
