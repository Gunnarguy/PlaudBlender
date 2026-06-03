#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Pi Remote Access — Full Setup
# Run from Mac while SSH'd into Pi:
#   ssh your-pi-username@your-pi-lan-ip 'bash -s' < deploy/pi-remote-access.sh
#
# Sets up:
#   1. ngrok (HTTP API tunnel + HTTP UI tunnel + HTTP webhook tunnel)
#   2. RealVNC Server (for RaspController iOS)
#   3. Tailscale (mesh VPN — access from anywhere)
#   4. Restarts all Chronos services
# ─────────────────────────────────────────────────────────────
set -euo pipefail

# Load environment overrides from .env if present
ENV_FILE="$HOME/PlaudBlender/.env"
if [[ -f "$ENV_FILE" ]]; then
    set -a
    source "$ENV_FILE" 2>/dev/null || true
    set +a
fi

NGROK_AUTH="${NGROK_AUTHTOKEN:-your-ngrok-authtoken}"
NGROK_DOMAIN="${CHRONOS_NGROK_DOMAIN:-your-ngrok-domain.ngrok-free.dev}"

GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[0;36m'
RESET='\033[0m'

section() { echo -e "\n${CYAN}══ $1 ══${RESET}"; }
ok()      { echo -e "  ${GREEN}✔${RESET} $1"; }
fail()    { echo -e "  ${RED}✘${RESET} $1"; }

section "1. RESTART CHRONOS SERVICES"
for svc in chronos-qdrant chronos-api chronos-ui chronos-auto-sync; do
    if sudo systemctl restart "$svc" 2>/dev/null; then
        ok "$svc restarted"
    else
        fail "$svc failed to restart"
    fi
done

# Verify API is up with a short warmup window after restart.
API_READY=false
for _ in $(seq 1 20); do
    if curl -sf http://localhost:8000/api/v1/health > /dev/null 2>&1; then
        API_READY=true
        break
    fi
    sleep 1
done
if $API_READY; then
    ok "FastAPI is healthy"
else
    fail "FastAPI not responding — check: sudo journalctl -u chronos-api -n 20"
fi

section "2. NGROK SETUP (API + UI + WEBHOOK TUNNELS)"
# Install ngrok if missing
if ! command -v ngrok &>/dev/null; then
    echo "  Installing ngrok..."
    curl -fsSL https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-arm64.tgz | \
        sudo tar -xz -C /usr/local/bin
    ok "ngrok installed"
else
    ok "ngrok already installed ($(ngrok version 2>/dev/null || echo 'unknown'))"
fi

# Set auth token
ngrok config add-authtoken "$NGROK_AUTH" 2>/dev/null
ok "ngrok auth token configured"

# Write ngrok config with API, UI, and webhook tunnels.
NGROK_CONFIG="$HOME/.config/ngrok/ngrok.yml"
mkdir -p "$(dirname "$NGROK_CONFIG")"
cat > "$NGROK_CONFIG" << NGROK_CFG
version: "3"
agent:
    authtoken: $NGROK_AUTH
tunnels:
    chronos-api:
        proto: http
        addr: 8000
        domain: $NGROK_DOMAIN
    chronos-ui:
        proto: http
        addr: 8050
    chronos-webhook:
        proto: http
        addr: 8090
NGROK_CFG
ok "ngrok config written (API + UI + webhook tunnels)"

# Create systemd service for ngrok
sudo tee /etc/systemd/system/chronos-ngrok.service > /dev/null << SYSTEMD
[Unit]
Description=Chronos ngrok tunnels (API + UI + webhook)
After=network-online.target chronos-api.service
Wants=network-online.target

[Service]
Type=simple
User=$USER
ExecStart=/usr/local/bin/ngrok start --all --config $HOME/.config/ngrok/ngrok.yml
Restart=always
RestartSec=10
Environment=HOME=$HOME

[Install]
WantedBy=multi-user.target
SYSTEMD
sudo systemctl daemon-reload
sudo systemctl enable chronos-ngrok
sudo systemctl restart chronos-ngrok
ok "chronos-ngrok systemd service enabled"

# Wait for tunnels to come up
sleep 5
echo "  Fetching tunnel URLs..."
TUNNEL_INFO=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null || echo '{}')
HTTP_URL=$(echo "$TUNNEL_INFO" | python3 -c "
import json,sys
data=json.load(sys.stdin)
for t in data.get('tunnels',[]):
    if t.get('name')=='chronos-api':
        print(t['public_url']); break
" 2>/dev/null || echo "pending...")
UI_URL=$(echo "$TUNNEL_INFO" | python3 -c "
import json,sys
data=json.load(sys.stdin)
for t in data.get('tunnels',[]):
    if t.get('name')=='chronos-ui':
        print(t['public_url']); break
" 2>/dev/null || echo "pending...")
WEBHOOK_BASE_URL=$(echo "$TUNNEL_INFO" | python3 -c "
import json,sys
data=json.load(sys.stdin)
for t in data.get('tunnels',[]):
    if t.get('name')=='chronos-webhook':
        print(t['public_url']); break
" 2>/dev/null || echo "pending...")
ok "API tunnel:  $HTTP_URL"
ok "UI tunnel:   $UI_URL"
ok "Webhook tunnel: $WEBHOOK_BASE_URL"

# Update Pi environment with current public webhook/callback URLs.
ENV_FILE="$HOME/PlaudBlender/.env"
WEBHOOK_URL="$WEBHOOK_BASE_URL/webhook/plaud"
REDIRECT_URL="$UI_URL/auth/plaud/callback"

if [ -f "$ENV_FILE" ]; then
    if grep -q '^PLAUD_WEBHOOK_URL=' "$ENV_FILE"; then
        sed -i "s|^PLAUD_WEBHOOK_URL=.*|PLAUD_WEBHOOK_URL=$WEBHOOK_URL|" "$ENV_FILE"
    else
        printf '\nPLAUD_WEBHOOK_URL=%s\n' "$WEBHOOK_URL" >> "$ENV_FILE"
    fi

    if grep -q '^PLAUD_REDIRECT_URI=' "$ENV_FILE"; then
        sed -i "s|^PLAUD_REDIRECT_URI=.*|PLAUD_REDIRECT_URI=$REDIRECT_URL|" "$ENV_FILE"
    else
        printf 'PLAUD_REDIRECT_URI=%s\n' "$REDIRECT_URL" >> "$ENV_FILE"
    fi
    ok "Updated .env with current Plaud webhook and redirect URLs"
fi

# Reload services that read these values.
sudo systemctl restart chronos-ui chronos-auto-sync >/dev/null 2>&1 || true
ok "Reloaded Chronos services after URL updates"

section "3. REALVNC SERVER SETUP"
# Install RealVNC Server if not present
if ! command -v vncserver-x11 &>/dev/null && ! dpkg -l realvnc-vnc-server 2>/dev/null | grep -q '^ii'; then
    echo "  Installing RealVNC Server..."
    # RealVNC is in Raspberry Pi OS repos
    sudo apt-get update -qq
    sudo apt-get install -y -qq realvnc-vnc-server 2>/dev/null || {
        # Fallback: enable via raspi-config
        sudo raspi-config nonint do_vnc 0 2>/dev/null && ok "VNC enabled via raspi-config" || fail "Could not install VNC — install manually"
    }
fi

# Enable VNC
if command -v raspi-config &>/dev/null; then
    sudo raspi-config nonint do_vnc 0 2>/dev/null && ok "VNC enabled" || echo "  VNC may already be enabled"
fi
sudo systemctl enable --now vncserver-x11-serviced 2>/dev/null && ok "VNC service started" || \
    sudo systemctl enable --now vncserver-virtuald 2>/dev/null && ok "VNC virtual service started" || \
    ok "VNC service status unknown — check manually"

# Check VNC port
if ss -tln | grep -q ':5900'; then
    ok "VNC listening on port 5900"
else
    echo "  VNC port 5900 not detected — may need reboot or manual start"
fi

section "4. TAILSCALE SETUP"
if ! command -v tailscale &>/dev/null; then
    echo "  Installing Tailscale..."
    curl -fsSL https://tailscale.com/install.sh | sh
    ok "Tailscale installed"
else
    ok "Tailscale already installed ($(tailscale version 2>/dev/null | head -1))"
fi
sudo systemctl enable --now tailscaled
ok "tailscaled service enabled"

# Check if already authenticated
if tailscale status 2>/dev/null | grep -q "$(hostname)"; then
    TS_IP=$(tailscale ip -4 2>/dev/null || echo "unknown")
    ok "Tailscale already connected: $TS_IP"
else
    echo ""
    echo -e "  ${CYAN}Tailscale needs authentication.${RESET}"
    echo "  Run this manually after the script:"
    echo ""
    echo "    sudo tailscale up --ssh"
    echo ""
    echo "  Open the URL it prints in your browser."
    echo "  Then get your IP: tailscale ip -4"
fi

section "5. FIREWALL RULES"
# Ensure SSH, VNC, and Chronos ports are open
if command -v ufw &>/dev/null && sudo ufw status | grep -q "active"; then
    sudo ufw allow 22/tcp comment "SSH"
    sudo ufw allow 5900/tcp comment "VNC"
    sudo ufw allow 8000/tcp comment "Chronos API"
    sudo ufw allow 8050/tcp comment "Chronos UI"
    ok "UFW rules added (22, 5900, 8000, 8050)"
else
    ok "No active firewall — ports should be open"
fi

section "SUMMARY"
echo ""
echo -e "  ${GREEN}Services:${RESET}"
for svc in chronos-qdrant chronos-api chronos-ui chronos-auto-sync chronos-ngrok vncserver-x11-serviced tailscaled; do
    state=$(systemctl is-active "$svc" 2>/dev/null || echo "not-found")
    if [ "$state" = "active" ]; then
        echo -e "    ${GREEN}●${RESET} $svc"
    else
        echo -e "    ${RED}○${RESET} $svc ($state)"
    fi
done

LAN_IP=$(hostname -I | awk '{print $1}' 2>/dev/null || echo "your-pi-lan-ip")
[[ -z "$LAN_IP" ]] && LAN_IP="your-pi-lan-ip"
CURRENT_USER=$(whoami)

echo ""
echo -e "  ${GREEN}Remote Access:${RESET}"
echo "    ngrok API:   $HTTP_URL"
echo "    ngrok UI:    $UI_URL"
echo "    ngrok Webhook: $WEBHOOK_URL"
echo "    SSH (LAN):   ssh $CURRENT_USER@$LAN_IP"
echo "    VNC:         port 5900 (local) / via Tailscale"
TS_IP=$(tailscale ip -4 2>/dev/null || echo "not yet configured")
echo "    Tailscale:   $TS_IP"
echo ""
echo -e "  ${CYAN}From your Mac (SSH):${RESET}"
echo "    LAN:       ssh $CURRENT_USER@$LAN_IP"
echo "    Tailscale: ssh $CURRENT_USER@<tailscale-ip>   (after 'sudo tailscale up --ssh')"
echo ""
echo -e "  ${CYAN}From RaspController (VNC via Tailscale):${RESET}"
echo "    Connect to $TS_IP:5900"
echo ""
echo -e "  ${CYAN}iOS app server URL:${RESET}"
echo "    Primary: $HTTP_URL"
echo "    LAN:     http://$LAN_IP:8000"
echo "    Tailscale: http://$TS_IP:8000"
echo ""
echo -e "  ${CYAN}Dash UI URL:${RESET}"
echo "    Primary: $UI_URL"
echo "    LAN:     http://$LAN_IP:8050"
echo "    Tailscale: http://$TS_IP:8050"
echo ""
echo -e "  ${CYAN}Plaud Webhook URL:${RESET}"
echo "    Primary: $WEBHOOK_URL"
echo "    LAN:     http://$LAN_IP:8090/webhook/plaud"
echo "    Tailscale: http://$TS_IP:8090/webhook/plaud"
echo ""
echo "══════════════════════════════════════════════════"
echo "  DONE — Pi is now accessible from anywhere!"
echo "══════════════════════════════════════════════════"
