#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Tailscale Setup for Chronos Pi
# Run: ssh your-pi-username@your-pi-lan-ip 'bash -s' < deploy/setup-tailscale-pi.sh
# ─────────────────────────────────────────────────────────────
set -euo pipefail

echo "══════════════════════════════════════"
echo "  Chronos Pi — Tailscale Setup"
echo "══════════════════════════════════════"

# 1. Install Tailscale
if ! command -v tailscale &>/dev/null; then
    echo "── Installing Tailscale ──"
    curl -fsSL https://tailscale.com/install.sh | sh
else
    echo "── Tailscale already installed ──"
    tailscale version
fi

# 2. Start and authenticate
echo ""
echo "── Starting Tailscale ──"
sudo systemctl enable --now tailscaled

echo ""
echo "══════════════════════════════════════"
echo "  Run this command to authenticate:"
echo ""
echo "    sudo tailscale up --ssh"
echo ""
echo "  It will print a URL — open it in your"
echo "  browser to authorize this Pi."
echo ""
echo "  Once done, run:"
echo "    tailscale ip -4"
echo ""
echo "  That 100.x.x.x IP is your anywhere-"
echo "  accessible Chronos server. Set it in"
echo "  the iOS app Settings → Server URL:"
echo "    http://100.x.x.x:8000"
echo "══════════════════════════════════════"
