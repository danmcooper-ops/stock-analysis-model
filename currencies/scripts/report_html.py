"""Render the screener results into a single self-contained HTML page.

Jinja2 is used purely as a JSON-injection mechanism: the template contains no
loops or conditionals, just a handful of `{{ }}` placeholders. All rendering
happens client-side in vanilla JS against the inlined `DATA` array and the
`GM` gate-metadata object.

That split is what makes the page work from `file://` with no network, no
build step and no CDN. At 30 currencies the whole payload is small enough to
inline outright — the sibling stock model needs lazy-loaded JSON sidecars
only because it carries 2,200 tickers.

The world basemap lives in `templates/world_svg.js` (Natural Earth 110m,
public domain) and is inlined here rather than fetched, for the same reason.
"""

import json
import math
import os
from datetime import date

import jinja2

from scripts.config import PILLAR_ORDER
from scripts.safe_json import dumps_for_script
from scripts.scoring import gate_metadata

_TEMPLATE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'templates')

# Fields carried into the page. Everything else in a row (raw series, working
# values) stays in the results JSON and out of the HTML.
_BASE_FIELDS = [
    'code', 'name', 'regime', 'map_iso3', 'bis_area',
    'rating', 'rating_raw', '_rating_cap', '_rating_cap_reasons',
    '_composite_score', '_data_coverage_score',
    '_gates_passed', '_gates_passed_num', '_gates_applicable',
    '_gates_inapplicable',
    # Raw metrics shown in the detail panel and sortable table columns.
    'reer_z20', 'reer_dev10', 'reer_dev5', 'reer_level', 'reer_period',
    'reer_months',
    'policy_rate', 'real_policy_rate', 'nominal_carry', 'real_carry',
    'carry_to_vol',
    'current_account_pct_gdp', 'reserves_months_imports', 'inflation_pct',
    'inflation_gap', 'gdp_growth_pct', 'gov_debt_pct_gdp',
    'external_debt_pct_gni', 'macro_year',
    'ret_3m', 'ret_6m', 'ret_12m', 'ret_12m_1m', 'spot_vs_ma200',
    'reer_trend_6m', 'spot', 'spot_date', 'spot_days',
    'vol_1y', 'vol_3m', 'vol_ratio', 'max_dd_3y', 'inflation_vol',
    'spark', '_provenance',
]


def _sanitize(value):
    """Recursively replace non-finite floats with None.

    NaN and Infinity are not valid JSON; json.dumps emits them as bare
    NaN/Infinity tokens which JSON.parse rejects and which silently poison an
    inline `var DATA = ...` assignment. Strings are checked too, because a
    value that round-tripped through str() arrives as the literal 'Infinity'.
    """
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, str) and value in ('Infinity', '-Infinity', 'NaN'):
        return None
    return value


def _load_world_svg():
    path = os.path.join(_TEMPLATE_DIR, 'world_svg.js')
    try:
        with open(path, encoding='utf-8') as f:
            return f.read().strip()
    except OSError:
        # The map view degrades to an empty panel rather than breaking the
        # page; the table and matrix are the primary deliverables.
        return 'var WORLD_SVG = "";'


def build_html(rows, filename, run_date=None, run_provenance=None,
               params=None):
    """Write the report. Returns the path written."""
    gm = gate_metadata(params)
    gate_field_keys = [k for g in gm['gates']
                       for k in (g['key'], g['gpKey'], g['scoreKey'])]
    pillar_keys = ['_score_pillar_' + c.lower() for c in PILLAR_ORDER]

    records = []
    for r in rows:
        rec = {k: r.get(k) for k in _BASE_FIELDS}
        rec.update({k: r.get(k) for k in gate_field_keys})
        rec.update({k: r.get(k) for k in pillar_keys})
        records.append(_sanitize(rec))

    run_date = run_date or date.today()
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(_TEMPLATE_DIR), autoescape=False)
    template = env.get_template('report.html')

    html = template.render(
        data_json=dumps_for_script(records, separators=(',', ':')),
        gate_meta=dumps_for_script(_sanitize(gm), separators=(',', ':')),
        provenance_json=dumps_for_script(_sanitize(run_provenance or {}),
                                         separators=(',', ':')),
        world_svg=_load_world_svg(),
        total=len(records),
        run_date_iso=run_date.isoformat(),
        generated_at=(run_provenance or {}).get('generated_at', ''),
    )

    os.makedirs(os.path.dirname(os.path.abspath(filename)) or '.', exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html)
    return filename


def load_results(path):
    """Read a results_<date>.json snapshot -> (rows, meta)."""
    with open(path, encoding='utf-8') as f:
        payload = json.load(f)
    return payload.get('results', []), payload
