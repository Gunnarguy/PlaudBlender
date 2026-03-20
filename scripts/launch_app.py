#!/usr/bin/env python
"""Launch Chronos app v2."""
import sys
import os
import signal
import subprocess
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PORT = 8050


def _kill_stale_server():
    """Kill any existing process on PORT so we get a clean start."""
    try:
        result = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{PORT}", "-sTCP:LISTEN", "-t"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.strip().splitlines():
            pid = int(line.strip())
            if pid != os.getpid():
                print(f"Killing stale server on port {PORT} (PID {pid})", flush=True)
                os.kill(pid, signal.SIGTERM)
    except (subprocess.TimeoutExpired, ValueError, ProcessLookupError, OSError):
        pass


_kill_stale_server()

from app_v2.main import create_app

app = create_app()

# Use HTTPS if certs exist (fixes Safari mixed-content OAuth block)
cert_dir = Path(__file__).resolve().parent.parent / ".certs"
cert_file = cert_dir / "localhost.crt"
key_file = cert_dir / "localhost.key"

if cert_file.exists() and key_file.exists():
    ssl_ctx = (str(cert_file), str(key_file))
    print("Starting Chronos v2 at https://localhost:8050 (HTTPS)", flush=True)
    app.run(debug=False, host="0.0.0.0", port=8050, ssl_context=ssl_ctx)
else:
    print("Starting Chronos v2 at http://localhost:8050 (no certs found)", flush=True)
    app.run(debug=False, host="0.0.0.0", port=8050)
