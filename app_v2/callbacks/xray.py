"""X-ray activity monitor — now served as a standalone window.

The X-ray panel has moved to a separate browser window (/xray route).
All rendering, polling, and filtering is handled client-side in the
standalone HTML page served by Flask.  This module only keeps the
register function (called from __init__.py) as a no-op.
"""


def register_xray_callbacks(app):
    """No-op — X-ray is now a standalone window at /xray."""
    pass
