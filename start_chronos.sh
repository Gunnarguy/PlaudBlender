#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "$ROOT_DIR/.venv" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_BIN="$ROOT_DIR/venv/bin/python"
fi

print_usage() {
  cat <<'EOF'
Usage:
  ./start_chronos.sh                Run full pipeline, then start UI
  ./start_chronos.sh --ui-only      Start UI only
  ./start_chronos.sh --pipeline-only Run pipeline only
  ./start_chronos.sh --help         Show this help
EOF
}

run_pipeline=1
run_ui=1

case "${1:-}" in
  "")
    ;;
  --ui-only)
    run_pipeline=0
    ;;
  --pipeline-only)
    run_ui=0
    ;;
  --help|-h)
    print_usage
    exit 0
    ;;
  *)
    echo "Unknown option: ${1}"
    print_usage
    exit 1
    ;;
esac

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python environment at: $PYTHON_BIN"
  echo "Create it first, then re-run this script."
  exit 1
fi

# Load environment overrides from .env if present
if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  source "$ROOT_DIR/.env" 2>/dev/null || true
  set +a
fi

QDRANT_REMOTE_URL="${QDRANT_REMOTE_URL:-}"

cd "$ROOT_DIR"

_probe_qdrant() {
  [[ -n "$QDRANT_REMOTE_URL" ]] || return 1
  if curl -s -m 1.5 "${QDRANT_REMOTE_URL}/v1/health" >/dev/null 2>&1; then
    return 0
  else
    return 1
  fi
}

echo "Checking configured remote Qdrant..."
if _probe_qdrant; then
  echo "✔ Remote Qdrant is ONLINE! Connecting directly..."
  export QDRANT_URL="${QDRANT_REMOTE_URL}"
else
  echo "⚠ Raspberry Pi Qdrant is OFFLINE. Activating local Docker fallback..."
  export QDRANT_URL="http://127.0.0.1:6333"
  
  DOCKER_AUTO_STARTED=0
  if ! docker info >/dev/null 2>&1; then
    echo "Starting Docker Desktop in background..."
    open -a Docker
    for i in {1..20}; do
      if docker info >/dev/null 2>&1; then
        echo "✔ Docker Desktop is ready."
        DOCKER_AUTO_STARTED=1
        break
      fi
      sleep 1.5
    done
  fi
  
  echo "Starting local Qdrant container..."
  docker compose up -d >/dev/null 2>&1 || true
  
  cleanup_docker() {
    echo ""
    echo "Cleaning up local fallback resources..."
    docker compose down >/dev/null 2>&1 || true
  fi
  trap cleanup_docker EXIT
fi

if [[ "$run_pipeline" -eq 1 ]]; then
  echo "Running full Chronos pipeline..."
  "$PYTHON_BIN" scripts/chronos_pipeline.py --full
fi

if [[ "$run_ui" -eq 1 ]]; then
  if (command -v lsof &>/dev/null && lsof -nP -iTCP:8050 -sTCP:LISTEN >/dev/null 2>&1) || \
     (command -v ss &>/dev/null && ss -tlnp 2>/dev/null | grep -q ":8050 "); then
    echo "Chronos UI is already running at http://localhost:8050"
    exit 0
  fi

  echo "Starting Chronos UI at http://localhost:8050"
  "$PYTHON_BIN" scripts/launch_app.py
fi
