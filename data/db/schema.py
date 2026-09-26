"""SQL generated from the column registry.

The typed columns of ``core.results`` are written into the core migration
between two marker lines by ``scripts/db_gen_registry.py``, and
``tests/test_db_registry.py`` checks the block still matches
``data/db/columns.COLUMNS``. Promoting a key later is a new migration
(``ALTER TABLE ... ADD COLUMN``), never an edit of an applied one.
"""
import re

BEGIN_MARK = '  -- BEGIN GENERATED registry columns (scripts/db_gen_registry.py)'
END_MARK = '  -- END GENERATED registry columns'

_IDENT = re.compile(r'^[A-Za-z_][A-Za-z0-9_]{0,62}$')


def is_safe_ident(name):
    """True for a plain identifier that is safe as a column name."""
    return bool(_IDENT.match(name))


def quote_ident(name):
    """Double-quoted identifier; registry keys are plain identifiers."""
    if not _IDENT.match(name):
        raise ValueError(f'not a safe column name: {name!r}')
    return f'"{name}"'


def results_columns_sql(columns):
    """The ``core.results`` column definitions for *columns* (``{key: type}``)."""
    return '\n'.join(f'  {quote_ident(k)} {t},' for k, t in columns.items())


def generated_block(sql_text):
    """The text between the markers in a migration, or None."""
    try:
        start = sql_text.index(BEGIN_MARK) + len(BEGIN_MARK) + 1
        end = sql_text.index(END_MARK)
    except ValueError:
        return None
    return sql_text[start:end].rstrip('\n')


def replace_generated_block(sql_text, columns):
    """*sql_text* with the marked block regenerated from *columns*."""
    start = sql_text.index(BEGIN_MARK) + len(BEGIN_MARK) + 1
    end = sql_text.index(END_MARK)
    return sql_text[:start] + results_columns_sql(columns) + '\n' + sql_text[end:]
