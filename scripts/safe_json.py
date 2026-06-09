"""Helpers for safely embedding JSON inside inline JavaScript."""

import json


def dumps_for_script(obj, **kwargs):
    """Serialize JSON for use inside a ``<script>`` tag.

    ``json.dumps`` alone can emit ``</script>`` inside string values. Browsers
    parse that as the end of the script block even when it appears inside a JS
    string, so escape HTML-significant characters after serialization.
    """
    text = json.dumps(obj, **kwargs)
    return (
        text
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
