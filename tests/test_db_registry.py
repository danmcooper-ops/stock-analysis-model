"""Column registry for core.results (data/db/columns.py) and its migration."""
import glob
import os

import pytest

from data.db.codec import PG_TYPES
from data.db.columns import COLUMNS, EXTRA_KEYS, GENERATED_FROM
from data.db.schema import generated_block, quote_ident, results_columns_sql
from data.snapshot_store import BLOB_KEYS, DEFAULT_EXCLUDE_KEYS
from scripts.db_gen_registry import build_registry, decide, Observations
from scripts.report_html import _PREV_DRIVER_KEYS
from scripts.scoring import APPLICABILITY_FIELDS, GATES, _gate_key, _gp_key, _score_key

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXED_COLUMNS = {'run_date', 'ticker_id', 'extra', 'edgar_history_sha'}

# Scalar keys the pipeline's store readers and the backtest read; they must be
# queryable typed columns, not buried in extra.
SCALAR_READER_KEYS = (
    'rating', 'rating_raw', '_rating_cap', '_composite_score', '_data_coverage_score', 'mos',
    'shares_out', 'mcap', 'data_source',                       # carry-forward
    '_gates_passed', '_gates_passed_num', 'dcf_fv', 'price',   # backtest
    'sector', 'pp_multiple', 'trap_score',
    'beneish_flag', 'altman_z_zone', 'pool_share_cagr',
    # backtest.py also reads trap_score_ex_momentum, but no pipeline step
    # writes it, so it never appears in a snapshot.
)


def _gate_keys():
    keys = set()
    for g in GATES:
        keys.update({_gate_key(g.name), _gp_key(g.name), _score_key(g.name), g.field})
    return keys


def test_every_key_readers_and_scorers_use_is_registered():
    wanted = set(_PREV_DRIVER_KEYS) | set(APPLICABILITY_FIELDS) | _gate_keys() | set(SCALAR_READER_KEYS)
    known = set(COLUMNS) | EXTRA_KEYS | set(BLOB_KEYS)
    assert sorted(wanted - known) == []


def test_scalar_reader_keys_are_typed_columns():
    assert sorted(k for k in SCALAR_READER_KEYS if k not in COLUMNS) == []


def test_registry_is_well_formed():
    assert COLUMNS, 'empty registry'
    assert set(COLUMNS.values()) <= set(PG_TYPES)
    assert not set(COLUMNS) & EXTRA_KEYS
    assert not set(COLUMNS) & set(DEFAULT_EXCLUDE_KEYS)
    assert not set(COLUMNS) & set(BLOB_KEYS)
    assert not {k.lower() for k in COLUMNS} & (FIXED_COLUMNS | {'ticker'})
    assert len({k.lower() for k in COLUMNS}) == len(COLUMNS)   # no case-only duplicates
    for k in COLUMNS:
        quote_ident(k)
    assert GENERATED_FROM['dates'] > 0
    # well under Postgres's 1,600-column limit, leaving room for promotions
    assert len(COLUMNS) + len(FIXED_COLUMNS) < 1000


def test_core_migration_matches_registry():
    paths = glob.glob(os.path.join(ROOT, 'supabase', 'migrations', '*_core_schema.sql'))
    assert len(paths) == 1
    with open(paths[0], encoding='utf-8') as f:
        sql = f.read()
    assert generated_block(sql) == results_columns_sql(COLUMNS), \
        'core.results columns differ from data/db/columns.py; regenerate with scripts/db_gen_registry.py'


@pytest.mark.parametrize('kinds,max_int,expected', [
    ({'bool'}, 0, 'boolean'),
    ({'int'}, 10, 'bigint'),
    ({'int'}, 2 ** 64, None),
    ({'int', 'float'}, 10, 'double precision'),
    ({'float'}, 0, 'double precision'),
    ({'int', 'float'}, 2 ** 53 + 1, None),
    ({'str'}, 0, 'text'),
    ({'str', 'float'}, 0, None),
    ({'bool', 'int'}, 1, None),
    ({'container'}, 0, None),
    (set(), 0, None),
])
def test_type_rules(kinds, max_int, expected):
    assert decide(kinds, max_int) == expected


def test_build_registry_from_rows():
    obs = Observations()
    obs.add_rows('2026-01-02', [
        {'ticker': 'A', 'n': 1, 'x': 1, 'pe': 'Infinity', 'flag': True, 'name': 'a', 'lst': [1],
         'mixed': 1, 'never': None, 'edgar_history': {}, 'news_headlines': []},
        {'ticker': 'B', 'n': 2, 'x': 2.5, 'pe': 12.0, 'flag': None, 'name': 'b', 'lst': None, 'mixed': 'x'},
    ])
    obs.add_rows('2026-01-05', [{'ticker': 'A', 'n': 3, 'bad key%': 1.0, 'old': None}])
    obs.add_rows('2026-01-06', [{'ticker': 'A', 'n': 4}])
    columns, extra, mixed = build_registry(obs, recent=2)
    assert columns == {'n': 'bigint'}
    # x, pe, flag, name were last seen outside the 2 newest snapshots
    assert extra == {'bad key%', 'flag', 'lst', 'mixed', 'name', 'never', 'old', 'pe', 'x'}
    assert mixed['mixed'] == ['int', 'str']
    assert mixed['bad key%'] == 'not an identifier'
    assert mixed['x'].startswith('retired')
    columns, _, _ = build_registry(obs, recent=10)
    assert columns == {'flag': 'boolean', 'n': 'bigint', 'name': 'text',
                       'pe': 'double precision', 'x': 'double precision'}
