#!/usr/bin/env bash
# Chronos System Update — automated daily updates for apt and global npm packages.
# Designed to be low-priority to avoid impacting Pi performance.

set -euo pipefail

REPO_DIR="${CHRONOS_ROOT:-$HOME/PlaudBlender}"
LOG_FILE="$REPO_DIR/logs/system-update.log"
DATE=$(date '+%Y-%m-%d %H:%M:%S')

mkdir -p "$(dirname "$LOG_FILE")"

exec >> "$LOG_FILE" 2>&1

echo "═══════════════════════════════════════════════"
echo "  System Update: $DATE"
echo "═══════════════════════════════════════════════"

# Check if we have sudo without password for these specific commands
if ! sudo -n true >/dev/null 2>&1; then
    echo "ERROR: Passwordless sudo required for automated system updates."
    exit 1
fi

echo "[1/2] Updating system packages (apt)..."
# Use nice and ionice to keep resource usage low
sudo nice -n 19 ionice -c 3 apt-get update -q
sudo DEBIAN_FRONTEND=noninteractive nice -n 19 ionice -c 3 apt-get upgrade -y -q

echo "[2/2] Updating global npm packages..."
if command -v npm >/dev/null 2>&1; then
    nice -n 19 ionice -c 3 npm update -g
else
    echo "npm not found, skipping."
fi

echo "✓ System update complete."
echo ""
