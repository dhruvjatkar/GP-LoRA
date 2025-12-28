#!/bin/bash
# Run all NLG experiments: LoRA (baseline) and GP-LoRA
# This script runs both LoRA and GP-LoRA on all three datasets: E2E, WebNLG, DART

set -e  # Exit on error

echo "=============================================="
echo "Running all NLG experiments (LoRA vs GP-LoRA)"
echo "=============================================="

# E2E Dataset
echo ""
echo "[1/6] E2E - LoRA (Baseline)"
echo "=============================================="
bash run_e2e_lora.sh

echo ""
echo "[2/6] E2E - GP-LoRA"
echo "=============================================="
bash run_e2e_gplora.sh

# WebNLG Dataset
echo ""
echo "[3/6] WebNLG - LoRA (Baseline)"
echo "=============================================="
bash run_webnlg_lora.sh

echo ""
echo "[4/6] WebNLG - GP-LoRA"
echo "=============================================="
bash run_webnlg_gplora.sh

# DART Dataset
echo ""
echo "[5/6] DART - LoRA (Baseline)"
echo "=============================================="
bash run_dart_lora.sh

echo ""
echo "[6/6] DART - GP-LoRA"
echo "=============================================="
bash run_dart_gplora.sh

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "=============================================="
echo ""
echo "Results are saved in:"
echo "  - ./trained_models/GPT2_M/e2e_lora/"
echo "  - ./trained_models/GPT2_M/e2e_gplora/"
echo "  - ./trained_models/GPT2_M/webnlg_lora/"
echo "  - ./trained_models/GPT2_M/webnlg_gplora/"
echo "  - ./trained_models/GPT2_M/dart_lora/"
echo "  - ./trained_models/GPT2_M/dart_gplora/"

