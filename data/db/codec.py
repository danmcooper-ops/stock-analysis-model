"""Lossless conversion between a snapshot result row and its Postgres form.

A row becomes three parts (see ``design/supabase-migration.md``):

* **typed values** — one per registry column (``data/db/columns.COLUMNS``), in
  registry order. ``double precision`` keeps NaN and ±Infinity natively, so
  the scoring rule "missing is not NaN" survives.
* **extra** — every other key (dicts, lists, keys not yet promoted, and any
  value that would not fit its typed column losslessly), stored as ``jsonb``.
* **blob** — the filing-driven ``BLOB_KEYS`` (``edgar_history``), stored once
  per distinct value and referenced by SHA-256.

``jsonb`` cannot hold NaN/Infinity or NUL characters, so :func:`encode` rewrites
them into tagged objects that :func:`decode` reverses:

* ``{"$nf": "NaN" | "Inf" | "-Inf"}`` — a non-finite float
* ``{"$s": "<JSON string literal>"}`` — a string holding NUL or a lone
  surrogate, kept as its ASCII-escaped JSON literal
* ``{"$lit": {...}}`` — a real dict with a ``$``-prefixed key, so data can
  never be mistaken for a tag

Fidelity contract: the rebuilt row is ``.get()``-equivalent to the source —
a typed column that is NULL is left out of the rebuilt row, ints and floats
compare numerically, and NaN equals NaN (:func:`rows_equivalent`). Two
normalisations are pinned on purpose: the strings ``"Infinity"``,
``"-Infinity"`` and ``"NaN"`` in a ``double precision`` column come back as
floats (the DuckDB store already treated them so), and ``-0.0`` in ``jsonb``
comes back as ``0``.
"""
import hashlib
import json
import math

from data.snapshot_store import BLOB_KEYS, DEFAULT_EXCLUDE_KEYS

PG_TYPES = ('boolean', 'bigint', 'double precision', 'text')

_INT64_MIN, _INT64_MAX = -2 ** 63, 2 ** 63 - 1
# Integers beyond this lose precision as a float, so they never go into a
# double column.
_FLOAT_EXACT_INT = 2 ** 53
NONFINITE_STRINGS = {'Infinity': math.inf, '-Infinity': -math.inf, 'NaN': math.nan}

_NF_TAG, _STR_TAG, _LIT_TAG = '$nf', '$s', '$lit'


class CastError(ValueError):
    """A value that cannot be stored losslessly in its typed column."""


def _needs_escape(s):
    if '\x00' in s:
        return True
    try:
        s.encode('utf-8')
    except UnicodeEncodeError:  # lone surrogate from a JSON "\udXXX" escape
        return True
    return False


def encode(v):
    """*v* rewritten so ``json.dumps(..., allow_nan=False)`` and ``jsonb`` accept it."""
    if isinstance(v, float):
        if math.isfinite(v):
            return v
        return {_NF_TAG: 'NaN' if math.isnan(v) else ('Inf' if v > 0 else '-Inf')}
    if isinstance(v, str):
        return {_STR_TAG: json.dumps(v)} if _needs_escape(v) else v
    if isinstance(v, dict):
        out = {}
        for k, x in v.items():
            k = k if isinstance(k, str) else json.dumps(k) if k is None or isinstance(k, bool) else str(k)
            if _needs_escape(k):
                raise CastError(f'dict key {k!r} cannot be stored in jsonb')
            out[k] = encode(x)
        return {_LIT_TAG: out} if any(k.startswith('$') for k in out) else out
    if isinstance(v, (list, tuple)):
        return [encode(x) for x in v]
    return v


def decode(v):
    """Inverse of :func:`encode` on a parsed JSON value."""
    if isinstance(v, dict):
        if len(v) == 1:
            (k, x), = v.items()
            if k == _NF_TAG:
                return {'NaN': math.nan, 'Inf': math.inf, '-Inf': -math.inf}[x]
            if k == _STR_TAG:
                return json.loads(x)
            if k == _LIT_TAG:
                return {kk: decode(xx) for kk, xx in x.items()}
        return {k: decode(x) for k, x in v.items()}
    if isinstance(v, list):
        return [decode(x) for x in v]
    return v


def dumps(v):
    """Compact ``jsonb`` text for *v*."""
    return json.dumps(encode(v), separators=(',', ':'), allow_nan=False, ensure_ascii=False)


def blob_sha(text):
    """Content address of an encoded blob."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def cast(v, pg_type):
    """*v* as stored in a *pg_type* column, or raise :class:`CastError`."""
    if v is None:
        return None
    if pg_type == 'double precision':
        if isinstance(v, float):
            return v
        if isinstance(v, int) and not isinstance(v, bool) and abs(v) <= _FLOAT_EXACT_INT:
            return float(v)
        if isinstance(v, str) and v in NONFINITE_STRINGS:
            return NONFINITE_STRINGS[v]
    elif pg_type == 'bigint':
        if isinstance(v, int) and not isinstance(v, bool) and _INT64_MIN <= v <= _INT64_MAX:
            return v
    elif pg_type == 'boolean':
        if isinstance(v, bool):
            return v
    elif pg_type == 'text':
        if isinstance(v, str) and not _needs_escape(v):
            return v
    else:
        raise ValueError(f'unknown column type {pg_type!r}')
    raise CastError(f'{type(v).__name__} does not fit {pg_type}')


class SplitRow:
    """A row in database form. ``typed`` follows the registry's key order."""
    __slots__ = ('ticker', 'typed', 'extra', 'blob', 'cast_failures')

    def __init__(self, ticker, typed, extra, blob, cast_failures):
        self.ticker = ticker
        self.typed = typed
        self.extra = extra
        self.blob = blob
        self.cast_failures = cast_failures


def split_row(row, columns, blob_keys=BLOB_KEYS, exclude=DEFAULT_EXCLUDE_KEYS):
    """Split a result *row* against the registry *columns* (``{key: pg_type}``).

    Report-only keys in *exclude* are dropped (they live only in the archived
    JSON). A value that does not fit its column goes to ``extra`` under its own
    key and is listed in ``cast_failures``; ``extra`` and ``blob`` are returned
    un-encoded (use :func:`dumps`).
    """
    skip = set(exclude) | set(blob_keys) | {'ticker'}
    typed, extra, failures = [], {}, []
    for k, t in columns.items():
        v = row.get(k)
        try:
            typed.append(cast(v, t))
        except CastError:
            typed.append(None)
            extra[k] = v
            failures.append(k)
    for k, v in row.items():
        if k not in columns and k not in skip:
            extra[k] = v
    blob = {k: row[k] for k in blob_keys if row.get(k) is not None}
    return SplitRow(row['ticker'], typed, extra, blob or None, failures)


def join_row(ticker, columns, typed, extra, blob=None):
    """Rebuild a row from its database form.

    *typed* are the column values in *columns* order (as read back from
    Postgres); *extra* and *blob* are the parsed ``jsonb`` values (or None).
    """
    row = {'ticker': ticker}
    for k, v in zip(columns, typed, strict=True):
        if v is not None:
            row[k] = v
    if extra:
        row.update(decode(extra))
    if blob:
        row.update(decode(blob))
    return row


def values_equal(a, b):
    """Equality under the codec's fidelity contract (NaN == NaN, 1 == 1.0)."""
    if a is b:
        return True
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, str) and a in NONFINITE_STRINGS and isinstance(b, float):
        a = NONFINITE_STRINGS[a]
    if isinstance(b, str) and b in NONFINITE_STRINGS and isinstance(a, float):
        b = NONFINITE_STRINGS[b]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            return True
        return a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(values_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(values_equal(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def rows_equivalent(source, rebuilt, exclude=DEFAULT_EXCLUDE_KEYS):
    """Keys whose ``.get()`` values differ between *source* and *rebuilt*.

    Empty means the rebuilt row honours the fidelity contract. Keys in
    *exclude* are ignored, since the database never stores them. Dict keys are
    compared as JSON would write them (strings).
    """
    skip = set(exclude)
    src = json.loads(json.dumps(encode(source), allow_nan=False))
    src = decode(src)
    keys = (set(src) | set(rebuilt)) - skip
    return sorted(k for k in keys if not values_equal(src.get(k), rebuilt.get(k)))
