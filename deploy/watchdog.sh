#!/usr/bin/env bash
# Chronos Health Watchdog — checks all services and restarts broken ones.
# Run by systemd timer every 5 minutes.

set -euo pipefail

LOG_PREFIX="[watchdog $(date '+%Y-%m-%d %H:%M:%S')]"

ok=true

# --- 1. Qdrant health ---
if ! curl -sf http://localhost:6333/healthz >/dev/null 2>&1; then
    echo "$LOG_PREFIX Qdrant unhealthy — restarting chronos-qdrant"
    sudo systemctl restart chronos-qdrant
    sleep 10
    ok=false
fi

# --- 2. Dash UI responds ---
if ! curl -sf -o /dev/null http://localhost:8050/ 2>/dev/null; then
    echo "$LOG_PREFIX Dash UI unreachable — restarting chronos-ui"
    sudo systemctl restart chronos-ui
    ok=false
fi

# --- 3. FastAPI responds ---
if ! curl -sf -o /dev/null http://localhost:8000/api/v1/health 2>/dev/null; then
    echo "$LOG_PREFIX FastAPI unreachable — restarting chronos-api"
    sudo systemctl restart chronos-api
    ok=false
fi

# --- 4. Auto-sync is alive ---
if ! systemctl is-active --quiet chronos-auto-sync; then
    echo "$LOG_PREFIX Auto-sync not active — restarting"
    sudo systemctl restart chronos-auto-sync
    ok=false
fi

# --- 5. Disk space check (warn if <1GB free) ---
avail_kb=$(df /home | tail -1 | awk '{print $4}')
if [ "$avail_kb" -lt 1048576 ] 2>/dev/null; then
    echo "$LOG_PREFIX WARNING: Low disk space — ${avail_kb}KB available"
    # Trim old logs to free space
    find /home/gunnarhostetler/PlaudBlender/logs -name "*.log" -size +50M -exec truncate -s 10M {} \;
    ok=false
fi

# --- 6. Docker container running ---
if ! docker ps --format '{{.Names}}' | grep -q qdrant; then
    echo "$LOG_PREFIX Qdrant Docker container not running — restarting chronos-qdrant"
    sudo systemctl restart chronos-qdrant
    ok=false
fi

if $ok; then
    echo "$LOG_PREFIX All services healthy"
fi
