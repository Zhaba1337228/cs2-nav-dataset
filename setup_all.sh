#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

mkdir -p logs
LOG_FILE="logs/setup_$(date +%Y%m%d_%H%M%S).log"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ERROR] Python not found: $PYTHON_BIN"
  exit 1
fi

nohup "$PYTHON_BIN" setup_all.py --run-installer "$@" >"$LOG_FILE" 2>&1 &
PID=$!

echo "[INFO] Bootstrap started in background"
echo "[INFO] PID: $PID"
echo "[INFO] Log: $LOG_FILE"
echo "[INFO] Watch logs:"
echo "tail -f $LOG_FILE"
