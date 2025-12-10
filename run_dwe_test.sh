#!/bin/bash
# ===============================================
# Minimal Safe Test Runner (skip preprocessing)
# Purpose: Run only diachronic semantic analysis
# ===============================================

ROOT_DIR=$(dirname "$(realpath "$0")")
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "[*] Checking slot parquet files..."
if [ ! -d "$ROOT_DIR/dataset/slots" ] || [ -z "$(ls -A "$ROOT_DIR/dataset/slots" 2>/dev/null)" ]; then
  echo "No slot parquet files found. Please run preprocessing first."
  exit 1
fi

echo "[*] Found slot parquet files. Proceeding with diachronic analysis..."

nohup python3 "$ROOT_DIR/tests/test_diachronic_word_embeddings.py" \
  > "$LOG_DIR/diachronic_only.log" 2>&1 &

PID=$!
echo ""
echo "===================================================="
echo "  Diachronic embedding test is running in BACKGROUND"
echo "----------------------------------------------------"
echo "  PID: $PID"
echo "  Log: logs/diachronic_only.log"
echo ""
echo "  Monitor with:"
echo "     tail -f logs/diachronic_only.log"
echo ""
echo "  Stop test manually with:"
echo "     kill $PID"
echo "===================================================="
