#!/usr/bin/env bash
# Chronos Pi Bootstrap — Installs dependencies and configures services
# Run on the Raspberry Pi as the deploy user (gunnarhostetler)
set -euo pipefail

REPO_DIR="$HOME/PlaudBlender"
VENV="$REPO_DIR/venv"

echo "═══════════════════════════════════════════════"
echo "  Chronos Pi Bootstrap"
echo "═══════════════════════════════════════════════"

# ── 1. System packages ──────────────────────────────────────
echo "[1/7] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3-dev python3-venv python3-pip \
    libffi-dev libssl-dev \
    git curl jq \
    ffmpeg \
    libopenblas-dev libatlas-base-dev \
    iproute2 \
    2>/dev/null
echo "  ✓ System packages installed"

# ── 2. Python venv ──────────────────────────────────────────
echo "[2/7] Setting up Python venv..."
cd "$REPO_DIR"
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
    echo "  ✓ Created venv"
else
    echo "  ✓ Venv already exists"
fi

# ── 3. Python dependencies ──────────────────────────────────
echo "[3/7] Installing Python dependencies..."
"$VENV/bin/pip" install --upgrade pip setuptools wheel -q
"$VENV/bin/pip" install -r requirements.txt -q
echo "  ✓ Python dependencies installed"

# ── 4. Create directories ───────────────────────────────────
echo "[4/7] Creating data directories..."
mkdir -p "$REPO_DIR"/{data/{raw,processed,cache/graphs,audio},logs,.run}
echo "  ✓ Directories ready"

# ── 5. Qdrant via Docker ────────────────────────────────────
echo "[5/7] Setting up Qdrant..."
if command -v docker &>/dev/null; then
    # Pull arm64 image
    docker pull qdrant/qdrant:latest 2>/dev/null || true
    echo "  ✓ Qdrant image pulled"
else
    echo "  ⚠ Docker not found — install Docker first"
fi

# ── 6. Install systemd services ─────────────────────────────
echo "[6/7] Installing systemd services..."
SYSTEMD_DIR="/etc/systemd/system"
for svc in "$REPO_DIR"/deploy/systemd/chronos-*.service; do
    svc_name=$(basename "$svc")
    sudo install -m 0644 "$svc" "$SYSTEMD_DIR/$svc_name"
    echo "  → $svc_name"
done
for timer in "$REPO_DIR"/deploy/systemd/chronos-*.timer; do
    [ -e "$timer" ] || continue
    timer_name=$(basename "$timer")
    sudo install -m 0644 "$timer" "$SYSTEMD_DIR/$timer_name"
    echo "  → $timer_name"
done
sudo systemctl daemon-reload
echo "  ✓ Systemd services installed"

# ── 7. Timezone check ───────────────────────────────────────
echo "[7/7] Checking timezone..."
TZ_CURRENT=$(timedatectl show --property=Timezone --value 2>/dev/null || echo "unknown")
echo "  Current timezone: $TZ_CURRENT"
if [[ "$TZ_CURRENT" == "Etc/UTC" || "$TZ_CURRENT" == "unknown" ]]; then
    echo "  ⚠ Consider: sudo timedatectl set-timezone America/Los_Angeles"
fi

echo ""
echo "═══════════════════════════════════════════════"
echo "  Bootstrap complete!"
echo "═══════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  1. Copy .env file:  scp .env pi:~/PlaudBlender/.env"
echo "  2. Enable services:"
echo "     sudo systemctl enable --now chronos-qdrant"
echo "     sudo systemctl enable --now chronos-ui"
echo "     sudo systemctl enable --now chronos-auto-sync"
echo "     sudo systemctl enable --now chronos-api"
echo "     sudo systemctl enable --now chronos-mcp"
echo "     sudo systemctl enable --now chronos-watchdog.timer"
echo ""
echo "  3. Check status:    sudo systemctl status 'chronos-*'"
echo "  4. View logs:       journalctl -u chronos-ui -f"
echo "  5. Update later:    ~/PlaudBlender/deploy/update-pi.sh"
echo ""
