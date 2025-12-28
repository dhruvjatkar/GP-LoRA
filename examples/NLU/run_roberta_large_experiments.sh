#!/bin/bash
# Run all NLU experiments: LoRA (baseline) and GP-LoRA
# This script runs both LoRA and GP-LoRA on all GLUE tasks with RoBERTa Large

set -e  # Exit on error

echo "=============================================="
echo "Running all NLU experiments (LoRA vs GP-LoRA)"
echo "RoBERTa Large on GLUE Benchmark"
echo "=============================================="

# Array of tasks
TASKS=("mnli" "sst2" "mrpc" "cola" "qnli" "qqp" "rte" "stsb")

# Run baseline LoRA experiments
echo ""
echo "=== PHASE 1: Running baseline LoRA experiments ==="
echo ""

for task in "${TASKS[@]}"; do
    echo "[LoRA] Running $task..."
    bash roberta_large_${task}.sh
    echo "[LoRA] $task completed!"
    echo ""
done

# Run GP-LoRA experiments
echo ""
echo "=== PHASE 2: Running GP-LoRA experiments ==="
echo ""

for task in "${TASKS[@]}"; do
    echo "[GP-LoRA] Running $task..."
    bash roberta_large_${task}_gplora.sh
    echo "[GP-LoRA] $task completed!"
    echo ""
done

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "=============================================="
echo ""
echo "Results are saved in the following directories:"
for task in "${TASKS[@]}"; do
    echo "  LoRA:    ./${task}_large/model/"
    echo "  GP-LoRA: ./${task}_large_gplora/model/"
done

