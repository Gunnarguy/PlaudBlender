#!/usr/bin/env bash
# PostToolUse hook for Edit|Write in PlaudBlender.
# After an edit to a .py file, runs the matching pytest file only (targeted, not the
# 419-test suite). Emits only failures, as additionalContext for Claude. Never blocks.
# Always exits 0.
# Stdin: the hook JSON; the edited path is tool_input.file_path.
set +e

ROOT="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"

FILE="$(python3 -c 'import json,sys
try:
    print(json.load(sys.stdin).get("tool_input", {}).get("file_path", ""))
except Exception:
    print("")' 2>/dev/null)"
[ -n "$FILE" ] || exit 0

case "$FILE" in
  /*) case "$FILE" in "$ROOT"/*) REL="${FILE#"$ROOT"/}" ;; *) exit 0 ;; esac ;;
  *) REL="$FILE" ;;
esac

case "$REL" in
  *.py) ;;
  *) exit 0 ;;
esac
case "$REL" in
  .venv/*|scratch/*|lib/*|.agent/*) exit 0 ;;
esac

cd "$ROOT" || exit 0

# Pick the test file: an edited test runs itself; a module runs tests/test_<name>.py,
# where <name> is tried as the full path joined by underscores, the same without a
# leading src/, and the bare basename. No match means no run and no output.
TARGET=""
case "$REL" in
  tests/test_*.py) TARGET="$REL" ;;
  *)
    stem="${REL%.py}"
    joined="$(printf '%s' "$stem" | tr '/' '_')"
    nosrc="$(printf '%s' "${stem#src/}" | tr '/' '_')"
    base="$(basename "$stem")"
    for cand in "tests/test_${joined}.py" "tests/test_${nosrc}.py" "tests/test_${base}.py"; do
      if [ -f "$cand" ]; then TARGET="$cand"; break; fi
    done
    ;;
esac
[ -n "$TARGET" ] || exit 0

PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"
[ -n "$PY" ] || exit 0
# No pytest in the chosen interpreter is an environment gap, not a test failure: stay silent.
"$PY" -c 'import pytest' >/dev/null 2>&1 || exit 0

if command -v gtimeout >/dev/null 2>&1; then TO="gtimeout 100"; else TO=""; fi
OUT="$($TO "$PY" -m pytest "$TARGET" -q -x -p no:cacheprovider 2>&1)"
RC=$?
[ "$RC" -eq 0 ] && exit 0

TAIL="$(printf '%s\n' "$OUT" | tail -n 40 | cut -c1-400)"
python3 - "$REL" "$RC" "$TARGET" "$TAIL" <<'PY'
import json, sys
rel, rc, target, tail = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
msg = (f"post-edit check FAILED (exit {rc}) after editing {rel}: .venv/bin/python -m pytest {target} -q -x\n"
       f"{tail}\nFix this before reporting done, then run the full suite: .venv/bin/python -m pytest tests/")
print(json.dumps({"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": msg}}))
PY
exit 0
