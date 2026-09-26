"""Direct Postgres connections (dev machines, CI, admin work).

The cloud pipeline cannot open raw TCP to Postgres and talks to Supabase over
HTTPS instead (design/supabase-migration.md, A1); this is for everything else.

Every connection sets:

* ``connect_timeout=5`` so an unreachable database costs seconds, not minutes
  (plan item R10);
* ``extra_float_digits=3`` because the Supabase image defaults it to 0, which
  makes Postgres *print* doubles with 15 significant digits. Values are stored
  exactly, but a read would round them (0.03932028370017462 came back as
  0.0393202837001746), breaking the codec's round trip.
"""
import psycopg

CONNECT_TIMEOUT_S = 5
SESSION_OPTIONS = '-c extra_float_digits=3'


def connect(dsn, **kwargs):
    """A psycopg connection with the settings every direct connection needs."""
    kwargs.setdefault('connect_timeout', CONNECT_TIMEOUT_S)
    options = kwargs.pop('options', '')
    kwargs['options'] = f'{SESSION_OPTIONS} {options}'.strip()
    return psycopg.connect(dsn, **kwargs)
