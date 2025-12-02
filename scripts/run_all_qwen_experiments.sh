#!/bin/bash
# =============================================================================
# Run All Qwen3-VL VWSD Experiments
# =============================================================================
# This script runs all Qwen3-VL inference methods with various configurations
# and saves results to results/rank_metrics.jsonl with descriptive model names.
#
# Naming convention for results:
#   {vlm_model}_{method}_{quantization}[_def-{def_model}]_batch{batch_size}
#
# Examples:
#   - qwen3-vl-8b_matching_bfloat16_batch1
#   - qwen3-vl-8b_matching_cot_bfloat16_batch1
#   - qwen3-vl-8b_description_bfloat16_def-qwen3-8b_batch1
#   - qwen3-vl-8b_embedding_bfloat16_batch1
# =============================================================================

# Note: Don't use 'set -e' as ((var++)) returns 1 when var=0, causing exit

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Clear previous results (optional - comment out to append)
# rm -f results/rank_metrics.jsonl

# Languages to evaluate
LANGUAGES=("en")  # Add "fa" "it" for other languages

# VLM models to test
VLM_MODELS=("qwen3-vl-8b")  # Can add "qwen3-vl-4b" "qwen3-vl-2b"

# Definition models (only for description method)
DEF_MODELS=("qwen3-8b")  # Can add "qwen3-4b" "qwen3-14b"

# Quantization options
QUANTIZATIONS=("bfloat16")  # Can add "4bit" for lower VRAM

# Batch size
BATCH_SIZE=1

# Methods to test (4 Qwen3-VL methods)
# - matching: Pure baseline (NO definitions)
# - matching_cot: Pure baseline + CoT (NO definitions)
# - description: Definition-based (REQUIRES definitions)
# - embedding: Direct cosine similarity (NO prompts/definitions)
METHODS=("matching" "matching_cot" "description" "embedding")

# =============================================================================
# Helper Functions
# =============================================================================

log_header() {
    echo ""
    echo "============================================================================="
    echo "$1"
    echo "============================================================================="
}

log_step() {
    echo ""
    echo ">>> $1"
    echo "-----------------------------------------------------------------------------"
}

run_experiment() {
    local vlm_model=$1
    local method=$2
    local quantization=$3
    local def_model=$4
    local language=$5
    local batch_size=$6

    # Build experiment name
    if [ "$method" == "description" ]; then
        exp_name="${vlm_model}_${method}_${quantization}_def-${def_model}_batch${batch_size}"
    else
        exp_name="${vlm_model}_${method}_${quantization}_batch${batch_size}"
    fi

    log_step "Running: $exp_name (lang: $language)"

    # Build command
    cmd="python main.py \
        --model-type vlm \
        --vlm-model $vlm_model \
        --method $method \
        --quantization $quantization \
        --vlm-batch-size $batch_size \
        --language $language"

    # Add definition model only for description method
    if [ "$method" == "description" ]; then
        cmd="$cmd --definition-model $def_model"
    fi

    echo "Command: $cmd"
    echo ""

    # Run experiment
    eval $cmd

    echo ""
    echo ">>> Completed: $exp_name"
}

# =============================================================================
# Main Execution
# =============================================================================

log_header "Qwen3-VL VWSD Experiment Suite"

echo "Configuration:"
echo "  - VLM Models: ${VLM_MODELS[*]}"
echo "  - Methods: ${METHODS[*]}"
echo "  - Quantizations: ${QUANTIZATIONS[*]}"
echo "  - Definition Models: ${DEF_MODELS[*]}"
echo "  - Languages: ${LANGUAGES[*]}"
echo "  - Batch Size: $BATCH_SIZE"

# Count total experiments
total_exp=0
for vlm in "${VLM_MODELS[@]}"; do
    for method in "${METHODS[@]}"; do
        for quant in "${QUANTIZATIONS[@]}"; do
            for lang in "${LANGUAGES[@]}"; do
                if [ "$method" == "description" ]; then
                    for def in "${DEF_MODELS[@]}"; do
                        ((total_exp++))
                    done
                else
                    ((total_exp++))
                fi
            done
        done
    done
done

echo "  - Total Experiments: $total_exp"
echo ""

# Auto-start (no confirmation needed)
echo "Starting experiments..."

# Track experiment count
exp_count=0
start_time=$(date +%s)

# Run all experiments
for vlm_model in "${VLM_MODELS[@]}"; do
    for quantization in "${QUANTIZATIONS[@]}"; do
        for language in "${LANGUAGES[@]}"; do
            for method in "${METHODS[@]}"; do
                if [ "$method" == "description" ]; then
                    # Description method requires definition model
                    for def_model in "${DEF_MODELS[@]}"; do
                        ((exp_count++))
                        log_header "Experiment $exp_count / $total_exp"
                        run_experiment "$vlm_model" "$method" "$quantization" "$def_model" "$language" "$BATCH_SIZE"
                    done
                else
                    # Other methods don't use definitions
                    ((exp_count++))
                    log_header "Experiment $exp_count / $total_exp"
                    run_experiment "$vlm_model" "$method" "$quantization" "" "$language" "$BATCH_SIZE"
                fi
            done
        done
    done
done

# Summary
end_time=$(date +%s)
elapsed=$((end_time - start_time))

log_header "All Experiments Complete!"
echo ""
echo "Total experiments: $exp_count"
echo "Total time: ${elapsed}s ($(($elapsed / 60))m $(($elapsed % 60))s)"
echo ""
echo "Results saved to: results/rank_metrics.jsonl"
echo ""

# Display results summary
echo "Results Summary:"
echo "-----------------------------------------------------------------------------"
if [ -f "results/rank_metrics.jsonl" ]; then
    python -c "
import json
import pandas as pd

with open('results/rank_metrics.jsonl', 'r') as f:
    data = [json.loads(line) for line in f if line.strip()]

# Filter to show key metrics
cols = ['model', 'hit_official/en', 'mrr_official/en']
available_cols = [c for c in cols if c in data[0]]

df = pd.DataFrame(data)[available_cols]
print(df.to_markdown(index=False))
"
fi

echo ""
echo "Done!"
