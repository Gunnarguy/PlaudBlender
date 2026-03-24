"""Quick smoke test for the Chronos API — starts server, hits endpoints, exits."""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

load_dotenv(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
)

from api.main import app
import uvicorn, threading, time, urllib.request, json

ENDPOINTS = [
    "/api/v1/health",
    "/api/v1/costs/session",
    "/api/v1/costs/pricing",
    "/api/v1/timeline/days",
    "/api/v1/topics",
    "/api/v1/stats",
    "/api/v1/stats/db",
    "/api/v1/xray/events",
    "/api/v1/sync/status",
    "/api/v1/auth/plaud/status",
    "/api/v1/auth/notion/status",
]


def run_tests():
    time.sleep(3)
    ok = 0
    for path in ENDPOINTS:
        try:
            resp = urllib.request.urlopen(f"http://127.0.0.1:8004{path}", timeout=5)
            data = json.loads(resp.read())
            print(f"  {resp.status} {path}: {json.dumps(data)[:80]}")
            ok += 1
        except Exception as e:
            print(f"  ERR {path}: {e}")
    print(f"\n  {ok}/{len(ENDPOINTS)} endpoints OK")
    os._exit(0 if ok == len(ENDPOINTS) else 1)


t = threading.Thread(target=run_tests, daemon=True)
t.start()
uvicorn.run(app, host="127.0.0.1", port=8004, log_level="warning")
