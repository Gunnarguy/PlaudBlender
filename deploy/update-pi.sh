#!/usr/bin/env bash
# Chronos Pi Update — pull latest code, refresh deps, reload systemd, restart services.

set -euo pipefail

REPO_DIR="$HOME/PlaudBlender"
VENV="$REPO_DIR/venv"
SYSTEMD_DIR="/etc/systemd/system"

restart_if_enabled() {
    local unit="$1"
    if systemctl list-unit-files "$unit" --no-legend 2>/dev/null | grep -q "$unit"; then
        if systemctl is-enabled "$unit" >/dev/null 2>&1; then
            echo "  → restarting $unit"
            sudo systemctl restart "$unit"
        else
            echo "  → $unit is installed but not enabled; leaving it alone"
        fi
    fi
}

echo "═══════════════════════════════════════════════"
echo "  Chronos Pi Update"
echo "═══════════════════════════════════════════════"

cd "$REPO_DIR"

echo "[1/6] Pulling latest code..."
git fetch origin
git pull --ff-only
echo "  ✓ Code updated"

echo "[2/6] Refreshing Python dependencies..."
"$VENV/bin/pip" install --upgrade pip setuptools wheel -q
"$VENV/bin/pip" install -r requirements.txt -q
echo "  ✓ Python dependencies refreshed"

echo "[3/6] Installing systemd unit files..."
for svc in "$REPO_DIR"/deploy/systemd/chronos-*.service; do
    sudo install -m 0644 "$svc" "$SYSTEMD_DIR/$(basename "$svc")"
done
for timer in "$REPO_DIR"/deploy/systemd/chronos-*.timer; do
    [ -e "$timer" ] || continue
    sudo install -m 0644 "$timer" "$SYSTEMD_DIR/$(basename "$timer")"
done
sudo systemctl daemon-reload
echo "  ✓ Systemd units reloaded"

echo "[4/6] Ensuring watchdog timer is enabled..."
sudo systemctl enable --now chronos-watchdog.timer
echo "  ✓ Watchdog timer active"

echo "[5/6] Restarting enabled Chronos services..."
restart_if_enabled chronos-qdrant.service
restart_if_enabled chronos-ui.service
restart_if_enabled chronos-auto-sync.service
restart_if_enabled chronos-api.service
restart_if_enabled chronos-mcp.service
restart_if_enabled chronos-ngrok.service
echo "  ✓ Service restart pass complete"

echo "[6/6] Quick health check..."
curl -fsS --max-time 10 http://localhost:8050/ >/dev/null && echo "  ✓ UI reachable" || echo "  ⚠ UI not reachable yet"
curl -fsS --max-time 10 http://localhost:8000/api/v1/health >/dev/null && echo "  ✓ API reachable" || echo "  ⚠ API not reachable yet"
systemctl is-active --quiet chronos-auto-sync && echo "  ✓ Auto-sync active" || echo "  ⚠ Auto-sync inactive"

echo ""
echo "Update complete."
