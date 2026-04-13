#!/usr/bin/env bash
# Chronos Pi Verify — prints a red/green report for the Pi runtime.

set -euo pipefail

ROOT="${CHRONOS_ROOT:-$HOME/PlaudBlender}"
PYTHON_BIN="$ROOT/venv/bin/python"

PASS_COUNT=0
WARN_COUNT=0
FAIL_COUNT=0

pass() {
    echo "PASS  $1"
    PASS_COUNT=$((PASS_COUNT + 1))
}

warn() {
    echo "WARN  $1"
    WARN_COUNT=$((WARN_COUNT + 1))
}

fail() {
    echo "FAIL  $1"
    FAIL_COUNT=$((FAIL_COUNT + 1))
}

check_unit() {
    local unit="$1"
    local label="$2"
    local active
    local enabled
    active=$(systemctl is-active "$unit" 2>/dev/null || true)
    enabled=$(systemctl is-enabled "$unit" 2>/dev/null || true)
    if [[ "$active" == "active" || "$active" == "activating" || "$active" == "reloading" ]]; then
        pass "$label: $active ($enabled)"
    else
        fail "$label: $active ($enabled)"
    fi
}

check_http() {
    local url="$1"
    local label="$2"
    if curl -fsS --max-time 10 -o /dev/null "$url"; then
        pass "$label reachable at $url"
    else
        fail "$label unreachable at $url"
    fi
}

echo "═══════════════════════════════════════════════"
echo "  Chronos Pi Verify"
echo "═══════════════════════════════════════════════"

if ! command -v systemctl >/dev/null 2>&1; then
    echo "FAIL  systemctl not found — this script is for the Pi/systemd host"
    exit 1
fi

echo "[1/5] Checking services and timer..."
check_unit chronos-qdrant.service "Qdrant service"
check_unit chronos-ui.service "UI service"
check_unit chronos-auto-sync.service "Auto-sync service"
check_unit chronos-api.service "API service"
check_unit chronos-mcp.service "MCP service"
check_unit chronos-watchdog.timer "Watchdog timer"

echo "[2/5] Checking ports and health endpoints..."
check_http http://127.0.0.1:8050/ "Dash UI"
check_http http://127.0.0.1:8000/api/v1/health "FastAPI"
check_http http://127.0.0.1:6333/healthz "Qdrant"
check_http http://127.0.0.1:8090/health "Webhook listener"

echo "[3/5] Checking Plaud auth..."
if auth_result=$(cd "$ROOT" && "$PYTHON_BIN" - <<'PY' 2>&1
from src.plaud_oauth import PlaudOAuthClient

client = PlaudOAuthClient()
status = client.token_status

if not status["has_access_token"] and not status["has_refresh_token"]:
    raise SystemExit("missing Plaud OAuth tokens")

if status["needs_refresh"]:
    client.ensure_valid_token()
    status = client.token_status

if status.get("is_authenticated"):
    mins = status.get("expires_in_minutes")
    if mins is None:
        print("Plaud token active")
    else:
        print(f"Plaud token valid for ~{int(mins)} min")
else:
    raise SystemExit("Plaud token exists but is not authenticated")
PY
); then
    pass "$auth_result"
else
    fail "Plaud auth unhealthy: $auth_result"
fi

echo "[4/5] Checking disk space..."
avail_kb=$(df /home | tail -1 | awk '{print $4}')
if [[ "$avail_kb" =~ ^[0-9]+$ ]]; then
    avail_gb=$((avail_kb / 1024 / 1024))
    if (( avail_kb < 1048576 )); then
        fail "Low disk space: ${avail_gb}GB free"
    elif (( avail_kb < 2097152 )); then
        warn "Disk getting tight: ${avail_gb}GB free"
    else
        pass "Disk space healthy: ${avail_gb}GB free"
    fi
else
    fail "Could not determine disk space"
fi

echo "[5/5] Recent watchdog activity..."
if [[ -f "$ROOT/logs/watchdog.log" ]]; then
    tail -n 5 "$ROOT/logs/watchdog.log" | sed 's/^/      /'
    pass "watchdog.log present"
else
    warn "watchdog.log not found yet"
fi

echo ""
echo "Summary: $PASS_COUNT pass, $WARN_COUNT warn, $FAIL_COUNT fail"

if (( FAIL_COUNT > 0 )); then
    exit 1
fi

exit 0
