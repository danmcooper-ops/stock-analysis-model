"""Claude-generated macro narrative for the Macro Outlook tab.

Serializes the macro.json sidecar's numeric facts (regime model output,
FRED series with changes/percentiles, the Treasury curve, credit spreads,
and per-sector ETF momentum) into a prompt for the Claude API and returns
a structured narrative: a bottom line, economy-wide paragraphs, what to
watch, the international backdrop, and one entry per GICS sector — an `influence` paragraph, a kicker and a
stance — which the Overview's "Sectors" chips open as
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

from models.narrative import (
    _SECTOR_MACRO_DRIVERS,
    _SECTOR_THESIS_RISKS,
    _SECTOR_THESIS_TAILWINDS,
)

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
# v6: each influence blends the sector's structural headwinds/tailwinds
#     (the Sector Analysis tab's list) with today's macro forces. Same
#     shape; bumped so a same-day v5 cache is not replayed.
# v7: the Overview reorganised — a `bottom_line` that leads (and reconciles
#     the regime model's label with the data), `watch` (what would change
#     the call) in place of the economy-wide headwinds/tailwinds, which
#     restated the sections' bullets, and a rule against repeating figures.
# v8: `international` — a lead, one point per region and a spillover line
#     for the Overview's International Economy card, read from the `intl`
#     series family the facts now hand over apart from the US series.
SCHEMA_VERSION = 8

# Ceilings for the v7 fields, in words; counted post-parse, never enforced.
BOTTOM_LINE_MAX_WORDS = 60
WATCH_ITEMS = 3
INTL_SPILLOVER_MAX_WORDS = 25

# The International card's regions, the series' `region` keys, in the order
# scripts/macro_dashboard.REGIONS draws them (data/ never imports scripts/).
INTL_REGIONS = ['ea', 'uk', 'jp', 'cn', 'em']

# Advisory floor for a sector's influence paragraph, only ever counted:
# under it the Overview is back to reading like a kicker, which is what
# the paragraph replaced.
MIN_INFLUENCE_WORDS = 20

# Target length of the Overview read (bottom line + paragraphs + what to
# watch + international + sector headlines and influences), ~3.5 minutes.
# The per-field ceilings in the prompt sum to it: 60 + 3 x (20 + 3 x 20) +
# 3 x 20 + (20 + 5 x 20 + 25) + 11 x (4 + 40) ~= 990 at the ceilings, ~860
# as drawn (v7's ~750 plus ~110 for the international block). Influences are
# drawn at their ceiling: a 45-word one put the v6 Overview at 820 words.
# Counted post-parse, never enforced.
OVERVIEW_WORD_BUDGET = 860

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
        'bottom_line': {'type': 'string'},
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
        'watch': {'type': 'array', 'items': {'type': 'string'}},
        # The International Economy card: a verdict, one point per region
        # (the region an enum, so the page can label it), and what the world
        # means for US equities.
        'international': {
            'type': 'object',
            'properties': {
                'lead': {'type': 'string'},
                'regions': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'region': {'type': 'string',
                                       'enum': INTL_REGIONS},
                            'point': {'type': 'string'},
                        },
                        'required': ['region', 'point'],
                        'additionalProperties': False,
                    },
                },
                'spillover': {'type': 'string'},
            },
            'required': ['lead', 'regions', 'spillover'],
            'additionalProperties': False,
        },
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
    'required': ['bottom_line', 'paragraphs', 'watch', 'international',
                 'sectors'],
    'additionalProperties': False,
}

SYSTEM_PROMPT = (
    'You are the macro strategist for a fundamental equity research '
    'report. Write a narrative assessment of the US economy and its global '
    'backdrop from the indicator data provided: regime model output, FRED '
    'series with recent changes and historical percentiles, the Treasury '
    'yield curve, credit spreads by rating bucket, sector ETF momentum, and '
    "an 'international' block of foreign policy rates, yields, inflation, "
    'equity markets and currencies grouped by region.\n'
    'Rules:\n'
    '- Use ONLY the numbers provided, and cite specific figures inline '
    "(e.g. 'core PCE at 2.8%'). Never invent a data point.\n"
    '- Declarative plain-English prose for a long-horizon investor; '
    'no hedging boilerplate, no first person, no investment advice. Never '
    'write "may", "could", "likely", "remains to be seen", or "bears '
    'watching".\n'
    '- Figures: round for a reader, not a terminal — at most two decimals '
    "('3.35%', not '3.353%'), whole-number ordinal percentiles ('2nd "
    "percentile', never '1.5th'), basis points as 'bp', and name each "
    "series the way a reader knows it ('the 10-year Treasury yield', not "
    "'DGS10'). Give a percentile its window once ('71st percentile of 10 "
    "years') rather than on every mention.\n"
    '- Length: the Overview prints the bottom line, the paragraphs, what '
    'to watch, and every sector headline and influence, and '
    f'that read must total about {OVERVIEW_WORD_BUDGET} words. The word '
    'limits below add up to that budget; treat each as a ceiling and '
    'spend the words on meaning, not on more figures.\n'
    '- Say each figure once. Across the bottom line, the paragraphs, '
    'what to watch and international, cite each indicator\'s number at most once; later '
    'mentions name the indicator without restating the figure. A sector '
    'influence may repeat a number only for the force that drives that '
    'sector.\n'
    f'- bottom_line: 2 or 3 sentences, {BOTTOM_LINE_MAX_WORDS} words '
    'maximum, read first and alone: where the economy stands, the one '
    'tension that matters most, and what that means for equity investors. '
    "It must square with the regime model's label and composite in the "
    'data: when the indicators read weaker or stronger than the label, say '
    'so and why the market signals still read as they do. At most two '
    'figures. It is about the US: the world goes in international.\n'
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
    f'- watch: exactly {WATCH_ITEMS} items, what would change the call — '
    'each the indicator, the level or move that would matter, and what it '
    "would signal, one sentence of 20 words or fewer: 'Initial claims "
    "above 250,000 would say layoffs have started, ending the labor "
    "story.' Each on a different indicator family; thresholds may be new "
    'numbers, but never restate a current reading already given.\n'
    "- international: the global backdrop, read from the data's "
    "'international' block only (region keys: ea euro area, uk United "
    'Kingdom, jp Japan, cn China, em emerging markets).\n'
    '  lead: ONE sentence, 20 words maximum, the verdict on the world '
    'economy as it bears on the US: where growth and policy are '
    'diverging or moving together.\n'
    '  regions: one entry per region in the data, in the order ea, uk, '
    'jp, cn, em; point is ONE sentence of 20 words or fewer with at most '
    'two figures, saying what its central bank, bond market, currency or '
    'equity market says about that economy. Equity series are 1-year price '
    'returns. A series marked stale may be cited only with its month.\n'
    f'  spillover: ONE sentence, {INTL_SPILLOVER_MAX_WORDS} words maximum, '
    'what the world means for US equities now, through the dollar, US '
    'yields, commodity prices or overseas earnings.\n'
    '- sectors: one entry for EVERY GICS sector listed in the data (all '
    '11, including any without ETF metrics), each written in this order:\n'
    '  influence: the read a sector\'s chip opens under the Overview\'s '
    '"Sectors", where it is the only thing a reader sees about the '
    'sector. 3 short complete sentences, 30 to 40 words in all, plain '
    'declarative '
    'prose rather than wordplay. Explain the MECHANISM: the one or two '
    'macro forces that matter most for this sector now and the channel '
    'each works through (financing costs, consumer or business demand, '
    'input and commodity prices, pricing power and margins, or the '
    'discount rate on long-dated earnings). Then weigh them against the '
    "sector's standing forces — its 'structural' headwinds and tailwinds "
    'in the data, the list its own Sector Analysis tab shows — naming the '
    'ONE structural force that most reinforces or offsets them, and say '
    'which dominates now, as the sector ETF has priced it. Cite each macro '
    "force's indicator and figure and explain it rather than list it; "
    'paraphrase the structural force in a few words, never quote it.\n'
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
    "  stance: the net effect of today's macro backdrop on the sector, read "
    'against its structural position — tailwind, neutral, or headwind.'
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
                     'pct_win', 'z', 'suffix', 'good', 'stale')


def _driver_for(sector):
    """Static macro sensitivities for a sector."""
    return _SECTOR_MACRO_DRIVERS.get(sector, {})


def build_macro_facts(sidecar):
    """Compact, prompt-ready facts dict from a macro.json sidecar. Pure."""
    sidecar = sidecar or {}
    series = sidecar.get('series') or {}
    facts_series = {}
    # The intl family goes in its own block, grouped by region, so the US
    # sections are written from US series and the International card from
    # the rest.
    international = {}
    for sid, s in series.items():
        entry = {k: s[k] for k in _SERIES_FACT_KEYS if s.get(k) not in (None, '')}
        if not entry:
            continue
        if s.get('sec') == 'intl':
            international.setdefault(s.get('region') or 'other', {})[sid] = entry
        else:
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
        # The standing forces the Sector Analysis tab lists under Sector
        # Headwinds & Tailwinds, so each influence can weigh today's macro
        # against them rather than repeat them.
        structural = {k: list(src.get(sector) or [])
                      for k, src in (('headwinds', _SECTOR_THESIS_RISKS),
                                     ('tailwinds', _SECTOR_THESIS_TAILWINDS))}
        if any(structural.values()):
            entry['structural'] = structural
        sectors[sector] = entry

    return {
        'as_of': sidecar.get('as_of'),
        'regime': sidecar.get('regime'),
        'series': facts_series,
        'yield_curve': curve,
        'credit_oas_by_rating': sidecar.get('oas_buckets'),
        'international': international,
        'sectors': sectors,
    }


def overview_word_count(narrative):
    """Words the Overview prints — the same fields the page's reading-time
    footer counts. Pure."""
    n = narrative or {}

    def wc(x):
        return len(str(x).split()) if x else 0
    total = wc(n.get('bottom_line'))
    total += sum(wc(x) for k in ('paragraphs', 'watch', 'tailwinds',
                                 'headwinds')
                 for x in (n.get(k) or []))
    intl = n.get('international') or {}
    if isinstance(intl, dict):
        total += wc(intl.get('lead')) + wc(intl.get('spillover'))
        total += sum(wc(r.get('point')) for r in intl.get('regions') or []
                     if isinstance(r, dict))
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
        if len(narrative.get('watch') or []) != WATCH_ITEMS:
            logger.warning('macro narrative: %d watch items (expected %d)',
                           len(narrative.get('watch') or []), WATCH_ITEMS)
        # Regions in the card's order, one entry each (the grammar pins no
        # array length, so a repeat is possible, as with the sectors).
        intl = narrative.get('international')
        if isinstance(intl, dict):
            by_region = {}
            for r in intl.get('regions') or []:
                k = (r or {}).get('region')
                if k and k not in by_region and str(r.get('point') or '').strip():
                    by_region[k] = r
            intl['regions'] = [by_region[k] for k in INTL_REGIONS
                               if k in by_region]
            missing = set(facts['international']) & set(INTL_REGIONS) \
                - set(by_region)
            if missing:
                logger.warning('macro narrative: international omits %s',
                               ', '.join(sorted(missing)))
        if len(str(narrative.get('bottom_line') or '').split()) > \
                1.25 * BOTTOM_LINE_MAX_WORDS:
            logger.warning('macro narrative: bottom line runs %d words',
                           len(str(narrative.get('bottom_line')).split()))
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
