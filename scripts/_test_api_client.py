"""Test endpoints against running API server on port 8000."""

import urllib.request, json, sys

PORT = 8000
ENDPOINTS = [
    "/api/v1/health",
    "/api/v1/costs/session",
    "/api/v1/costs/pricing",
    "/api/v1/xray/events",
    "/api/v1/sync/status",
    "/api/v1/auth/plaud/status",
    "/api/v1/auth/notion/status",
    "/api/v1/timeline/days",
    "/api/v1/topics",
    "/api/v1/stats",
    "/api/v1/stats/db",
    "/api/v1/stats/workflows",
    "/api/v1/graph",
    "/api/v1/xray/throughput",
]

ok = 0
for path in ENDPOINTS:
    try:
        resp = urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=30)
        data = json.loads(resp.read())
        print(f"{resp.status} {path}: {json.dumps(data)[:90]}")
        ok += 1
    except Exception as e:
        print(f"ERR {path}: {e}")

print(f"\n{ok}/{len(ENDPOINTS)} OK")
sys.exit(0 if ok > 0 else 1)
