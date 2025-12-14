#!/bin/bash

ROOT_DIR=$(dirname "$(realpath "$0")")
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

echo ""
echo "===================================================="
echo "  Running SBERT Diachronic Drift Test"
echo "===================================================="
echo ""

TEST_SCRIPT="$ROOT_DIR/tests/test_diachronic_word_embeddings_sbert.py"
LOG_FILE="$LOG_DIR/diachronic_sbert.log"

if [ ! -f "$TEST_SCRIPT" ]; then
  echo "[Error] Test script not found: $TEST_SCRIPT"
  exit 1
fi

echo "[*] Starting SBERT diachronic test ..."
nohup python3 "$TEST_SCRIPT" > "$LOG_FILE" 2>&1 &

OPT_PID=$!
echo "  → PID: $OPT_PID (log: logs/diachronic_sbert.log)"

echo ""
echo "===================================================="
echo "  SBERT diachronic test running in BACKGROUND."
echo ""
echo "  Monitor progress:"
echo "     tail -f logs/diachronic_sbert.log"
echo ""
echo "  Stop test:"
echo "     kill $OPT_PID"
echo ""
echo "  Log:"
echo "     $LOG_FILE"
echo "===================================================="
echo ""
