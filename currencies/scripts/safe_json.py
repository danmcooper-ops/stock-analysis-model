"""Helpers for safely embedding JSON inside inline JavaScript."""

import json


def dumps_for_script(obj, **kwargs):
    """Serialize JSON for use inside a ``<script>`` tag.

    ``json.dumps`` alone can emit ``</script>`` inside a string value, and the
    HTML parser ends the script block there even though it is inside a JS
    string literal. Escaping the HTML-significant characters afterwards keeps
    the payload inert; the escapes are ordinary JS string escapes, so the
    parsed values are unchanged.
    """
    text = json.dumps(obj, **kwargs)
    return (text
            .replace('&', '\\u0026')
            .replace('<', '\\u003c')
            .replace('>', '\\u003e')
            .replace(' ', '\\u2028')
            .replace(' ', '\\u2029'))
