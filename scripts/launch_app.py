#!/usr/bin/env python
"""Launch Chronos app v2."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app_v2.main import create_app

app = create_app()
print("Starting Chronos v2 at http://localhost:8050", flush=True)
app.run(debug=False, host="0.0.0.0", port=8050)
