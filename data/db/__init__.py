"""Supabase Postgres layer: column registry, row codec and schema helpers.

See ``design/supabase-migration.md``. The JSON snapshots stay canonical;
``codec`` converts a result row to and from its database form without loss.
"""
