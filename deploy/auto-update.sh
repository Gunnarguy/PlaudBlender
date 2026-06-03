#!/usr/bin/env bash
# Chronos Auto-Update — polls GitHub and runs the hardened Pi updater when needed.

set -euo pipefail

REPO_DIR="${CHRONOS_ROOT:-$HOME/PlaudBlender}"
UPDATE_SCRIPT="$REPO_DIR/deploy/update-pi.sh"
REMOTE="origin"
LOG_PREFIX="[auto-update $(date '+%Y-%m-%d %H:%M:%S')]"

mkdir -p "$REPO_DIR/logs" "$REPO_DIR/.run"

if [[ ! -x "$UPDATE_SCRIPT" ]]; then
    echo "$LOG_PREFIX WARNING: update script missing at $UPDATE_SCRIPT"
    exit 0
fi

if ! git -C "$REPO_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "$LOG_PREFIX WARNING: $REPO_DIR is not a git repository"
    exit 0
fi

CURRENT_BRANCH=$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || true)
if [[ -z "$CURRENT_BRANCH" || "$CURRENT_BRANCH" == "HEAD" ]]; then
    echo "$LOG_PREFIX WARNING: repo is not on a normal branch; skipping auto-update"
    exit 0
fi

if ! git -C "$REPO_DIR" remote get-url "$REMOTE" >/dev/null 2>&1; then
    echo "$LOG_PREFIX WARNING: git remote '$REMOTE' is not configured; skipping auto-update"
    exit 0
fi

if ! git -C "$REPO_DIR" diff --quiet --ignore-submodules --exit-code || \
   ! git -C "$REPO_DIR" diff --cached --quiet --ignore-submodules --exit-code; then
    echo "$LOG_PREFIX WARNING: local tracked changes exist; skipping auto-update to avoid overwriting them"
    exit 0
fi

if ! sudo -n true >/dev/null 2>&1; then
    echo "$LOG_PREFIX WARNING: passwordless sudo is required for auto-update; skipping"
    exit 0
fi

echo "$LOG_PREFIX checking $REMOTE/$CURRENT_BRANCH for updates"
git -C "$REPO_DIR" fetch --quiet "$REMOTE" "$CURRENT_BRANCH"

LOCAL_SHA=$(git -C "$REPO_DIR" rev-parse HEAD)
REMOTE_SHA=$(git -C "$REPO_DIR" rev-parse "$REMOTE/$CURRENT_BRANCH")
BASE_SHA=$(git -C "$REPO_DIR" merge-base HEAD "$REMOTE/$CURRENT_BRANCH")

if [[ "$LOCAL_SHA" == "$REMOTE_SHA" ]]; then
    echo "$LOG_PREFIX already up to date on $CURRENT_BRANCH"
    exit 0
fi

if [[ "$LOCAL_SHA" == "$BASE_SHA" ]]; then
    echo "$LOG_PREFIX new commits detected on $CURRENT_BRANCH; applying update"
    "$UPDATE_SCRIPT"
    exit $?
fi

if [[ "$REMOTE_SHA" == "$BASE_SHA" ]]; then
    echo "$LOG_PREFIX WARNING: local branch is ahead of GitHub; skipping auto-update"
    exit 0
fi

echo "$LOG_PREFIX WARNING: local branch has diverged from GitHub; skipping auto-update"
exit 0
