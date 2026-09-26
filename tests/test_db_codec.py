"""Row codec for the Supabase database (data/db/codec.py)."""
import json
import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from data.db.codec import (CastError, cast, decode, dumps, encode, join_row, rows_equivalent, split_row,
                           values_equal)

# Text including NUL and lone surrogates, which jsonb rejects unescaped.
_text = st.text(alphabet=st.characters(codec=None, exclude_categories=()), max_size=12)
_safe_key = st.text(max_size=8).filter(lambda s: '\x00' not in s)
_scalar = st.one_of(
    st.none(), st.booleans(),
    st.integers(min_value=-2 ** 70, max_value=2 ** 70),
    st.floats(allow_nan=True, allow_infinity=True),
    _text,
    st.sampled_from(['$nf', '$s', '$lit', 'NaN', 'Infinity']),
)
_value = st.recursive(
    _scalar,
    lambda ch: st.one_of(
        st.lists(ch, max_size=4),
        st.dictionaries(st.one_of(_safe_key, st.sampled_from(['$nf', '$s', '$lit', '$blob'])), ch, max_size=4),
    ),
    max_leaves=20,
)


def _via_jsonb(v):
    """What Postgres hands back for a jsonb value written with dumps()."""
    return decode(json.loads(dumps(v)))


@settings(max_examples=400)
@given(_value)
def test_jsonb_round_trip_is_lossless(v):
    assert values_equal(_via_jsonb(v), v)


@given(_value)
def test_dumps_output_is_jsonb_safe(v):
    text = dumps(v)
    json.loads(text)                     # strict JSON, no bare NaN/Infinity
    assert '\x00' not in text
    text.encode('utf-8')                 # no lone surrogates


@pytest.mark.parametrize('tagged', [
    {'$nf': 'NaN'}, {'$s': '"x"'}, {'$lit': {'a': 1}}, {'$blob': 'abc'}, {'a': 1, '$nf': 'Inf'},
])
def test_real_dicts_that_look_like_tags_survive(tagged):
    assert _via_jsonb(tagged) == tagged
    assert list(encode(tagged)) == ['$lit']


def test_nonfinite_and_nul_encodings():
    assert encode(math.inf) == {'$nf': 'Inf'}
    assert encode(-math.inf) == {'$nf': '-Inf'}
    assert encode(math.nan) == {'$nf': 'NaN'}
    assert _via_jsonb('a\x00b') == 'a\x00b'
    assert _via_jsonb('\ud800') == '\ud800'
    assert _via_jsonb('café') == 'café'


def test_pinned_quirks():
    # jsonb drops the sign of -0.0; the contract compares numerically.
    assert values_equal(_via_jsonb(-0.0), -0.0)
    # "Infinity" strings in a double column come back as floats.
    assert cast('Infinity', 'double precision') == math.inf
    assert values_equal('Infinity', math.inf)


@pytest.mark.parametrize('value,pg_type', [
    (1.5, 'bigint'), (True, 'bigint'), (2 ** 63, 'bigint'),
    (True, 'double precision'), (2 ** 53 + 1, 'double precision'), ('1.5', 'double precision'),
    (1, 'boolean'), ('true', 'boolean'),
    (5, 'text'), ('a\x00b', 'text'), ('\ud800', 'text'), ({'a': 1}, 'text'),
])
def test_cast_rejects_lossy_values(value, pg_type):
    with pytest.raises(CastError):
        cast(value, pg_type)


def test_cast_accepts_fitting_values():
    assert cast(None, 'bigint') is None
    assert cast(3, 'double precision') == 3.0 and isinstance(cast(3, 'double precision'), float)
    assert math.isnan(cast(math.nan, 'double precision'))
    assert cast(-2 ** 63, 'bigint') == -2 ** 63
    assert cast(False, 'boolean') is False
    assert cast('x', 'text') == 'x'
    with pytest.raises(ValueError):
        cast(1, 'numeric')


COLUMNS = {'a': 'double precision', 'b': 'bigint', 'c': 'text', 'd': 'boolean'}


def _rebuild(row, columns=COLUMNS):
    sr = split_row(row, columns)
    extra = json.loads(dumps(sr.extra)) if sr.extra else None
    blob = json.loads(dumps(sr.blob)) if sr.blob else None
    return sr, join_row(sr.ticker, list(columns), sr.typed, extra, blob)


def test_split_row_parts():
    row = {'ticker': 'X', 'a': 1.5, 'b': 2.5, 'c': None, 'd': True, 'lst': [1, math.nan],
           'edgar_history': {'rev': [1, 2]}, 'news_headlines': ['dropped']}
    sr, rebuilt = _rebuild(row)
    assert sr.typed == [1.5, None, None, True]
    assert sr.cast_failures == ['b']
    assert values_equal(sr.extra, {'b': 2.5, 'lst': [1, math.nan]})
    assert sr.blob == {'edgar_history': {'rev': [1, 2]}}
    assert 'news_headlines' not in rebuilt          # report-only key, never stored
    assert rows_equivalent(row, rebuilt) == []


def test_null_typed_value_is_left_out():
    _, rebuilt = _rebuild({'ticker': 'X', 'a': None})
    assert 'a' not in rebuilt and rebuilt.get('a') is None


@settings(max_examples=300)
@given(st.fixed_dictionaries({}, optional={
    'a': _scalar, 'b': _scalar, 'c': _scalar, 'd': _scalar,
    'other': _value, 'edgar_history': _value,
}))
def test_split_join_honours_fidelity_contract(fields):
    row = {'ticker': 'T', **fields}
    _, rebuilt = _rebuild(row)
    assert rows_equivalent(row, rebuilt) == []


def test_values_equal_rules():
    assert values_equal(math.nan, math.nan)
    assert values_equal(1, 1.0)
    assert not values_equal(True, 1)
    assert not values_equal(None, 0)
    assert values_equal({'a': [1, math.nan]}, {'a': [1.0, math.nan]})
    assert not values_equal({'a': 1}, {'a': 1, 'b': 2})
