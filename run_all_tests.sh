#!/bin/bash

ROOT_DIR=$(dirname "$(realpath "$0")")

LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "[*] Running preprocessing test in background..."
nohup python3 "$ROOT_DIR/tests/test_data_preprocessing.py" \
    > "$LOG_DIR/preprocess.log" 2>&1 &

PRE_PID=$!
echo "  → PID: $PRE_PID (log: logs/preprocess.log)"


echo "[*] Running diachronic embedding test in background..."
nohup python3 "$ROOT_DIR/tests/test_diachronic_word_embeddings.py" \
    > "$LOG_DIR/diachronic.log" 2>&1 &

DIA_PID=$!
echo "  → PID: $DIA_PID (log: logs/diachronic.log)"


echo ""
echo "===================================================="
echo "  All tests are now running in BACKGROUND."
echo ""
echo "  Monitor with:"
echo "     tail -f logs/preprocess.log"
echo "     tail -f logs/diachronic.log"
echo ""
echo "  Or check running processes:"
echo "     ps -ef | grep python"
echo ""
echo "  Stop tests manually:"
echo "     kill $PRE_PID"
echo "     kill $DIA_PID"
echo "===================================================="
