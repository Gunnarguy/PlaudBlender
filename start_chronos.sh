#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$ROOT_DIR/venv/bin/python"

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

cd "$ROOT_DIR"

if [[ "$run_pipeline" -eq 1 ]]; then
  echo "Running full Chronos pipeline..."
  "$PYTHON_BIN" scripts/chronos_pipeline.py --full
fi

if [[ "$run_ui" -eq 1 ]]; then
  if lsof -nP -iTCP:8050 -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Chronos UI is already running at http://127.0.0.1:8050"
    exit 0
  fi

  echo "Starting Chronos UI at http://127.0.0.1:8050"
  exec "$PYTHON_BIN" scripts/launch_app.py
fi
