#!/bin/bash
# ╔══════════════════════════════════════════════════════════════╗
# ║  PlaudBlender Server                                        ║
# ║  Starts API + ngrok tunnel for iOS & web                    ║
# ║                                                             ║
# ║  Usage:  ./serve.sh          (start everything)             ║
# ║          ./serve.sh stop     (stop everything)              ║
# ║          ./serve.sh status   (check what's running)         ║
# ║          ./serve.sh logs     (tail API logs)                ║
# ║          ./serve.sh url      (print ngrok URL)              ║
# ╚══════════════════════════════════════════════════════════════╝

set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"

# Load environment overrides from .env if present
if [[ -f "$ROOT/.env" ]]; then
    set -a
    source "$ROOT/.env" 2>/dev/null || true
    set +a
fi

QDRANT_REMOTE_URL="${QDRANT_REMOTE_URL:-http://100.76.130.109:6333}"

API_PORT=8000
export CHRONOS_API_WORKERS=2
export UVLOOP_INSTALL="1"
API_LOG="$ROOT/.logs/api.log"
NGROK_LOG="$ROOT/.logs/ngrok.log"
PID_DIR="$ROOT/.logs"
NGROK_DOMAIN="${CHRONOS_NGROK_DOMAIN:-your-ngrok-domain.ngrok-free.dev}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
RESET='\033[0m'

mkdir -p "$PID_DIR"

# ── helpers ──────────────────────────────────────────────────

_api_pid() { cat "$PID_DIR/api.pid" 2>/dev/null; }
_ngrok_pid() { cat "$PID_DIR/ngrok.pid" 2>/dev/null; }

_is_running() {
    local pid="$1"
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

_ngrok_url() {
    python3 -c "
import urllib.request, json, sys
try:
    data = json.loads(urllib.request.urlopen('http://127.0.0.1:4040/api/tunnels', timeout=2).read())
    tunnels = data.get('tunnels', [])
    for t in tunnels:
        if t.get('proto') == 'https':
            print(t['public_url']); sys.exit(0)
    if tunnels:
        print(tunnels[0]['public_url']); sys.exit(0)
    sys.exit(1)
except: sys.exit(1)
" 2>/dev/null
}

_wait_for_ngrok() {
    for i in $(seq 1 20); do
        local url=""
        url=$(_ngrok_url) || true
        if [[ -n "$url" ]]; then
            echo "$url"
            return 0
        fi
        sleep 0.5
    done
    return 1
}

_update_env_redirect() {
    local url="$1"
    local callback="${url}/api/v1/auth/notion/callback"
    local env_file="$ROOT/.env"

    if grep -q "^NOTION_REDIRECT_URI=" "$env_file" 2>/dev/null; then
        sed -i '' "s|^NOTION_REDIRECT_URI=.*|NOTION_REDIRECT_URI=${callback}|" "$env_file"
    else
        echo "NOTION_REDIRECT_URI=${callback}" >> "$env_file"
    fi
}

_probe_qdrant() {
    # Check if remote Qdrant is online and reachable
    if curl -s -m 1.5 "${QDRANT_REMOTE_URL}/v1/health" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# ── commands ─────────────────────────────────────────────────

do_start() {
    echo ""
    echo -e "${BOLD}${CYAN}  ╔═══════════════════════════════════╗${RESET}"
    echo -e "${BOLD}${CYAN}  ║   PlaudBlender Server Starting…   ║${RESET}"
    echo -e "${BOLD}${CYAN}  ╚═══════════════════════════════════╝${RESET}"
    echo ""

    # Kill stale processes
    if _is_running "$(_api_pid)"; then
        echo -e "  ${YELLOW}⚠  Stopping old API server…${RESET}"
        kill "$(_api_pid)" 2>/dev/null; sleep 1
    fi
    if _is_running "$(_ngrok_pid)"; then
        echo -e "  ${YELLOW}⚠  Stopping old ngrok tunnel…${RESET}"
        kill "$(_ngrok_pid)" 2>/dev/null; sleep 1
    fi

    # Also kill anything lingering on the port
    local stale
    stale=$(lsof -ti:"$API_PORT" 2>/dev/null || true)
    if [[ -n "$stale" ]]; then
        echo -e "  ${YELLOW}⚠  Clearing port $API_PORT (PID $stale)…${RESET}"
        kill "$stale" 2>/dev/null; sleep 1
    fi

    # 1. Start ngrok first (so we get the URL before the API loads .env)
    echo -e "  ${DIM}Starting ngrok tunnel…${RESET}"
    nohup ngrok http "$API_PORT" --domain="$NGROK_DOMAIN" --log stdout --log-level info < /dev/null > "$NGROK_LOG" 2>&1 &
    echo $! > "$PID_DIR/ngrok.pid"

    local ngrok_url
    ngrok_url=$(_wait_for_ngrok)
    if [[ -z "$ngrok_url" ]]; then
        echo -e "  ${RED}✘  ngrok failed to start. Check $NGROK_LOG${RESET}"
        echo ""
        return 1
    fi
    echo -e "  ${GREEN}✔  ngrok${RESET}  →  ${BOLD}${ngrok_url}${RESET}"

    # 2. Update .env with the new ngrok URL
    _update_env_redirect "$ngrok_url"
    echo -e "  ${GREEN}✔  .env${RESET}  →  NOTION_REDIRECT_URI updated"

    # Dynamic Qdrant Database Routing & Fallback
    echo -e "  ${DIM}Checking connection to Remote Qdrant (${QDRANT_REMOTE_URL})…${RESET}"
    if _probe_qdrant; then
        echo -e "  ${GREEN}✔  Remote Qdrant is ONLINE! Connecting directly…${RESET}"
        export QDRANT_URL="${QDRANT_REMOTE_URL}"
        echo "0" > "$PID_DIR/docker_auto_started"
    else
        echo -e "  ${YELLOW}⚠  Raspberry Pi Qdrant is OFFLINE. Activating local Docker fallback…${RESET}"
        export QDRANT_URL="http://127.0.0.1:6333"
        
        # Ensure Docker Desktop is running
        if ! docker info >/dev/null 2>&1; then
            echo -e "  ${DIM}Starting Docker Desktop in background…${RESET}"
            open -a Docker
            echo "0" > "$PID_DIR/docker_auto_started"
            for i in {1..20}; do
                if docker info >/dev/null 2>&1; then
                    echo -e "  ${GREEN}✔  Docker Desktop is ready.${RESET}"
                    echo "1" > "$PID_DIR/docker_auto_started"
                    break
                fi
                sleep 1.5
            done
        else
            echo "0" > "$PID_DIR/docker_auto_started"
        fi
        
        # Run docker compose
        echo -e "  ${DIM}Starting local Qdrant container…${RESET}"
        docker compose up -d >/dev/null 2>&1 || true
    fi

    # 3. Start the API server
    echo -e "  ${DIM}Starting API server on port $API_PORT…${RESET}"
    cd "$ROOT"
    if [[ -d "$ROOT/.venv" ]]; then
        local local_python="$ROOT/.venv/bin/python"
    else
        local local_python="$ROOT/venv/bin/python"
    fi
    nohup "$local_python" scripts/launch_api.py --port "$API_PORT" > "$API_LOG" 2>&1 &
    echo $! > "$PID_DIR/api.pid"
    sleep 2

    if ! _is_running "$(_api_pid)"; then
        echo -e "  ${RED}✘  API server failed to start. Check:${RESET}"
        echo -e "     ${DIM}tail $API_LOG${RESET}"
        echo ""
        return 1
    fi
    echo -e "  ${GREEN}✔  API${RESET}   →  http://0.0.0.0:${API_PORT}"

    # Summary
    echo ""
    echo -e "  ${BOLD}────────────────────────────────────────${RESET}"
    echo -e "  ${GREEN}${BOLD}All services running.${RESET}"
    echo ""
    echo -e "  ${BOLD}Notion redirect URI (put this in notion.so/my-integrations):${RESET}"
    echo -e "  ${CYAN}${ngrok_url}/api/v1/auth/notion/callback${RESET}"
    echo ""
    echo -e "  ${DIM}API logs:    tail -f $API_LOG${RESET}"
    echo -e "  ${DIM}ngrok logs:  tail -f $NGROK_LOG${RESET}"
    echo -e "  ${DIM}ngrok UI:    http://127.0.0.1:4040${RESET}"
    echo -e "  ${DIM}Stop:        ./serve.sh stop${RESET}"
    echo ""
}

do_stop() {
    echo ""
    local stopped=0

    # Auto-stop local Docker container if it was started by this session
    local auto_started
    auto_started=$(cat "$PID_DIR/docker_auto_started" 2>/dev/null || echo "0")
    if [[ "$auto_started" == "1" ]]; then
        echo -e "  ${DIM}Stopping local Qdrant container…${RESET}"
        docker compose down >/dev/null 2>&1 || true
        echo -e "  ${GREEN}✔${RESET}  Local Qdrant container cleaned up and stopped."
        stopped=1
    fi
    rm -f "$PID_DIR/docker_auto_started"

    if _is_running "$(_api_pid)"; then
        kill "$(_api_pid)" 2>/dev/null
        echo -e "  ${GREEN}✔${RESET}  API server stopped"
        stopped=1
    fi
    rm -f "$PID_DIR/api.pid"

    if _is_running "$(_ngrok_pid)"; then
        kill "$(_ngrok_pid)" 2>/dev/null
        echo -e "  ${GREEN}✔${RESET}  ngrok tunnel stopped"
        stopped=1
    fi
    rm -f "$PID_DIR/ngrok.pid"

    # Cleanup any strays on the port
    local stale
    stale=$(lsof -ti:"$API_PORT" 2>/dev/null || true)
    if [[ -n "$stale" ]]; then
        kill "$stale" 2>/dev/null
        echo -e "  ${GREEN}✔${RESET}  Cleared stale process on port $API_PORT"
        stopped=1
    fi

    if [[ $stopped -eq 0 ]]; then
        echo -e "  ${DIM}Nothing was running.${RESET}"
    fi
    echo ""
}

do_status() {
    echo ""
    if _is_running "$(_api_pid)"; then
        echo -e "  ${GREEN}●${RESET}  API server    ${DIM}PID $(_api_pid)${RESET}  →  port $API_PORT"
    else
        echo -e "  ${RED}○${RESET}  API server    ${DIM}not running${RESET}"
    fi

    if _is_running "$(_ngrok_pid)"; then
        local url
        url=$(_ngrok_url) || url="(starting…)"
        echo -e "  ${GREEN}●${RESET}  ngrok tunnel  ${DIM}PID $(_ngrok_pid)${RESET}  →  $url"
    else
        echo -e "  ${RED}○${RESET}  ngrok tunnel  ${DIM}not running${RESET}"
    fi
    echo ""
}

do_logs() {
    if [[ ! -f "$API_LOG" ]]; then
        echo "No logs yet. Start the server first: ./serve.sh"
        return 1
    fi
    tail -f "$API_LOG"
}

do_url() {
    local url
    url=$(_ngrok_url)
    if [[ -n "$url" ]]; then
        echo "${url}/api/v1/auth/notion/callback"
    else
        echo "ngrok is not running. Start with: ./serve.sh"
        return 1
    fi
}

# ── dispatch ─────────────────────────────────────────────────

case "${1:-start}" in
    start)  do_start ;;
    stop)   do_stop ;;
    status) do_status ;;
    logs)   do_logs ;;
    url)    do_url ;;
    *)
        echo "Usage: ./serve.sh [start|stop|status|logs|url]"
        exit 1
        ;;
esac
