#!/usr/bin/env bash
# Chronos Health Watchdog — checks all services and restarts broken ones.
# Run by systemd timer every 5 minutes.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="$ROOT/venv/bin/python"
LOG_PREFIX="[watchdog $(date '+%Y-%m-%d %H:%M:%S')]"

ok=true

restart_unit() {
    local unit="$1"
    if systemctl restart "$unit"; then
        echo "$LOG_PREFIX restarted $unit"
    else
        echo "$LOG_PREFIX ERROR: failed to restart $unit"
    fi
    ok=false
}

unit_enabled() {
    local unit="$1"
    local enabled
    enabled=$(systemctl is-enabled "$unit" 2>/dev/null || true)
    [[ "$enabled" == "enabled" || "$enabled" == "enabled-runtime" || "$enabled" == "linked" || "$enabled" == "alias" ]]
}

http_healthy() {
    local url="$1"
    curl -fsS --max-time 10 -o /dev/null "$url"
}

# --- 1. Qdrant health ---
if ! http_healthy http://localhost:6333/healthz >/dev/null 2>&1; then
    echo "$LOG_PREFIX Qdrant unhealthy — restarting chronos-qdrant"
    restart_unit chronos-qdrant
    sleep 10
fi

# --- 2. Dash UI responds ---
# Dash UI may serve HTTPS (self-signed, .certs/) or plain HTTP depending on
# whether certs are present. Accept either so a protocol switch cannot
# trigger a restart loop. -k is safe: this is a loopback self-signed cert.
if ! curl -fsSk --max-time 10 -o /dev/null https://localhost:8050/ >/dev/null 2>&1 \
   && ! http_healthy http://localhost:8050/ >/dev/null 2>&1; then
    echo "$LOG_PREFIX Dash UI unreachable — restarting chronos-ui"
    restart_unit chronos-ui
fi

# --- 3. FastAPI responds ---
if ! http_healthy http://localhost:8000/api/v1/health >/dev/null 2>&1; then
    echo "$LOG_PREFIX FastAPI unreachable — restarting chronos-api"
    restart_unit chronos-api
fi

# --- 4. Auto-sync + webhook listener health ---
if ! systemctl is-active --quiet chronos-auto-sync; then
    echo "$LOG_PREFIX Auto-sync not active — restarting"
    restart_unit chronos-auto-sync
    sleep 10
elif ! http_healthy http://localhost:8090/health >/dev/null 2>&1; then
    echo "$LOG_PREFIX Webhook listener unhealthy — restarting chronos-auto-sync"
    restart_unit chronos-auto-sync
    sleep 10
fi

# --- 5. MCP server process is alive ---
if unit_enabled chronos-mcp.service && ! systemctl is-active --quiet chronos-mcp; then
    echo "$LOG_PREFIX MCP server not active — restarting"
    restart_unit chronos-mcp
fi

# --- 6. Plaud auth tokens present and refreshable ---
if ! auth_status=$(cd "$ROOT" && "$PYTHON_BIN" - <<'PY' 2>&1
from src.plaud_oauth import PlaudOAuthClient

client = PlaudOAuthClient()
status = client.token_status

if not status["has_access_token"] and not status["has_refresh_token"]:
    raise SystemExit("Plaud OAuth tokens are missing")

client.ensure_valid_token()

print("ok")
PY
); then
    echo "$LOG_PREFIX WARNING: Plaud auth unhealthy — ${auth_status}"
    ok=false
fi

# --- 7. Disk space check (warn if <1GB free) ---
avail_kb=$(df /home | tail -1 | awk '{print $4}')
if [ "$avail_kb" -lt 1048576 ] 2>/dev/null; then
    echo "$LOG_PREFIX WARNING: Low disk space — ${avail_kb}KB available"
    # Trim old logs to free space
    find "$ROOT/logs" -name "*.log" -size +50M -exec truncate -s 10M {} \;
    ok=false
fi

# --- 8. Docker container running ---
if command -v docker >/dev/null 2>&1 && ! docker ps --format '{{.Names}}' | grep -q qdrant; then
    echo "$LOG_PREFIX Qdrant Docker container not running — restarting chronos-qdrant"
    restart_unit chronos-qdrant
fi

if $ok; then
    echo "$LOG_PREFIX All services healthy"
fi
