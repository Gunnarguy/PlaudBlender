#!/usr/bin/env bash
# Chronos Pi Update — pull latest code, refresh deps, reload systemd, restart services.

set -euo pipefail

REPO_DIR="$HOME/PlaudBlender"
VENV="$REPO_DIR/venv"
SYSTEMD_DIR="/etc/systemd/system"
RUN_DIR="$REPO_DIR/.run"
LOCK_FILE="$RUN_DIR/update-pi.lock"

acquire_update_lock() {
    mkdir -p "$RUN_DIR"

    if command -v flock >/dev/null 2>&1; then
        exec 9>"$LOCK_FILE"
        if ! flock -n 9; then
            echo "Another Chronos update is already running; skipping."
            exit 0
        fi
        return
    fi

    if ! mkdir "$LOCK_FILE.d" 2>/dev/null; then
        echo "Another Chronos update is already running; skipping."
        exit 0
    fi

    trap 'rmdir "$LOCK_FILE.d" 2>/dev/null || true' EXIT
}

unit_is_enabled() {
    local unit="$1"
    systemctl is-enabled "$unit" >/dev/null 2>&1
}

restart_if_enabled() {
    local unit="$1"
    if systemctl list-unit-files "$unit" --no-legend 2>/dev/null | grep -q "$unit"; then
        if unit_is_enabled "$unit"; then
            echo "  → restarting $unit"
            sudo systemctl restart "$unit"
        else
            echo "  → $unit is installed but not enabled; leaving it alone"
        fi
    fi
}

wait_for_unit_active() {
    local unit="$1"
    local timeout_seconds="$2"
    local waited=0

    while (( waited < timeout_seconds )); do
        if systemctl is-active --quiet "$unit"; then
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done

    return 1
}

wait_for_http() {
    local url="$1"
    local timeout_seconds="$2"
    local waited=0

    while (( waited < timeout_seconds )); do
        if curl -fsS --max-time 5 -o /dev/null "$url"; then
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done

    return 1
}

heal_service() {
    local unit="$1"
    local label="$2"
    local url="$3"
    local timeout_seconds="${4:-45}"

    if ! systemctl list-unit-files "$unit" --no-legend 2>/dev/null | grep -q "$unit"; then
        return 0
    fi

    if ! unit_is_enabled "$unit"; then
        return 0
    fi

    echo "  → waiting for $label"
    if wait_for_unit_active "$unit" "$timeout_seconds" && wait_for_http "$url" "$timeout_seconds"; then
        echo "    ✓ $label healthy"
        return 0
    fi

    echo "    ↻ $label still not healthy; restarting $unit once more"
    sudo systemctl restart "$unit"

    if wait_for_unit_active "$unit" "$timeout_seconds" && wait_for_http "$url" "$timeout_seconds"; then
        echo "    ✓ $label recovered after retry"
        return 0
    fi

    echo "    ⚠ $label still unhealthy after retry"
    return 1
}

echo "═══════════════════════════════════════════════"
echo "  Chronos Pi Update"
echo "═══════════════════════════════════════════════"

acquire_update_lock

cd "$REPO_DIR"

echo "[1/6] Pulling latest code..."
# Services rewrite telemetry into these tracked files while running, which makes
# --ff-only refuse to pull. Discard the local churn; it regenerates on the next
# probe. Kept in step with RUNTIME_TRACKED_FILES in auto-update.sh.
for runtime_file in "plaud-capability-manifest.json"; do
    [[ -e "$runtime_file" ]] || continue
    if ! git diff --quiet --ignore-submodules -- "$runtime_file"; then
        echo "  · discarding runtime-generated changes in $runtime_file"
        git checkout -- "$runtime_file" || true
    fi
done
git fetch origin
git pull --ff-only
echo "  ✓ Code updated"

echo "[2/6] Refreshing Python dependencies..."
"$VENV/bin/pip" install --upgrade pip setuptools wheel -q
"$VENV/bin/pip" install -r requirements.txt -q
echo "  ✓ Python dependencies refreshed"

echo "[3/6] Installing systemd unit files..."
CURRENT_USER=$(whoami)
for svc in "$REPO_DIR"/deploy/systemd/chronos-*.service; do
    svc_name=$(basename "$svc")
    sed "s|your-pi-username|$CURRENT_USER|g" "$svc" | sudo tee "$SYSTEMD_DIR/$svc_name" > /dev/null
    sudo chmod 0644 "$SYSTEMD_DIR/$svc_name"
done
for timer in "$REPO_DIR"/deploy/systemd/chronos-*.timer; do
    [ -e "$timer" ] || continue
    timer_name=$(basename "$timer")
    sed "s|your-pi-username|$CURRENT_USER|g" "$timer" | sudo tee "$SYSTEMD_DIR/$timer_name" > /dev/null
    sudo chmod 0644 "$SYSTEMD_DIR/$timer_name"
done
sudo systemctl daemon-reload
echo "  ✓ Systemd units reloaded"

echo "[4/6] Ensuring maintenance timers are enabled..."
sudo systemctl enable --now chronos-watchdog.timer
sudo systemctl enable --now chronos-auto-update.timer
echo "  ✓ Maintenance timers active"

echo "[5/6] Restarting enabled Chronos services..."
restart_if_enabled chronos-qdrant.service
restart_if_enabled chronos-ui.service
restart_if_enabled chronos-auto-sync.service
restart_if_enabled chronos-api.service
restart_if_enabled chronos-mcp.service
restart_if_enabled chronos-ngrok.service

echo "  → waiting for required services to become healthy"
heal_service chronos-qdrant.service "Qdrant" "http://127.0.0.1:6333/healthz" 60 || true
heal_service chronos-api.service "API" "http://127.0.0.1:8000/api/v1/health" 45 || true
heal_service chronos-ui.service "UI" "http://127.0.0.1:8050/" 45 || true
heal_service chronos-auto-sync.service "Webhook listener" "http://127.0.0.1:8090/health" 45 || true
echo "  ✓ Service restart pass complete"

echo "[6/6] Verifying Pi health..."
"$REPO_DIR/deploy/verify-pi.sh" || true

echo ""
echo "Update complete."
