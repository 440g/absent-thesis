#!/bin/bash

ROOT_DIR=$(dirname "$(realpath "$0")")
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

echo ""
echo "===================================================="
echo "  Running Optimized Diachronic Embedding Test"
echo "===================================================="
echo ""

TEST_SCRIPT="$ROOT_DIR/tests/test_diachronic_word_embeddings_opt.py"
LOG_FILE="$LOG_DIR/diachronic_opt.log"

if [ ! -f "$TEST_SCRIPT" ]; then
  echo "[Error] Test script not found: $TEST_SCRIPT"
  exit 1
fi

echo "[*] Starting test_diachronic_word_embeddings_opt.py ..."
nohup python3 "$TEST_SCRIPT" > "$LOG_FILE" 2>&1 &

OPT_PID=$!
echo "  → PID: $OPT_PID (log: logs/diachronic_opt.log)"

echo ""
echo "===================================================="
echo "  Optimized test is now running in BACKGROUND."
echo ""
echo "  Monitor progress with:"
echo "     tail -f logs/diachronic_opt.log"
echo ""
echo "  Check running process:"
echo "     ps -ef | grep python"
echo ""
echo "  Stop the test manually if needed:"
echo "     kill $OPT_PID"
echo ""
echo "  Log file:"
echo "     $LOG_FILE"
echo "===================================================="
echo ""
