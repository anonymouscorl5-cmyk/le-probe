#!/bin/bash
# Residual CLT sweep for ONE experiment at a time.
# Point ACTIVATIONS_DIR / OUTPUT_DIR at the harvest output for that run, e.g.:
#
#   ACTIVATIONS_DIR=activations_granular_multiview \
#   OUTPUT_DIR=transcoder_weights_multiview \
#   bash interpretability/transcoders/batch_train.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ACTIVATIONS_DIR="${ACTIVATIONS_DIR:-activations_granular}"
OUTPUT_DIR="${OUTPUT_DIR:-transcoder_weights_residual}"
DICT_SIZE="${DICT_SIZE:-12288}"
L1_COEFF="${L1_COEFF:-3e-3}"
EPOCHS="${EPOCHS:-10}"
WINDOW_SIZE="${WINDOW_SIZE:-1}"

mkdir -p "$OUTPUT_DIR"

echo "🔥 Residual Crosscoder Sweep"
echo "   Activations: $ACTIVATIONS_DIR"
echo "   Output:      $OUTPUT_DIR"

LAYERS=(
    "encoder_L0" "encoder_L1" "encoder_L2" "encoder_L3" "encoder_L4" "encoder_L5"
    "encoder_L6" "encoder_L7" "encoder_L8" "encoder_L9" "encoder_L10" "encoder_L11"
    "predictor_L0" "predictor_L1" "predictor_L2" "predictor_L3" "predictor_L4" "predictor_L5"
)

NUM_LAYERS=${#LAYERS[@]}

for i in $(seq 0 $((NUM_LAYERS - 1))); do
    SRC=${LAYERS[$i]}

    TGT_LIST=""
    for j in $(seq $((i - WINDOW_SIZE)) $((i + WINDOW_SIZE))); do
        if [ $j -ge 0 ] && [ $j -lt $NUM_LAYERS ]; then
            if [ -z "$TGT_LIST" ]; then
                TGT_LIST="${LAYERS[$j]}"
            else
                TGT_LIST="$TGT_LIST,${LAYERS[$j]}"
            fi
        fi
    done

    echo "⚙️ Training Residual Crosscoder for $SRC ⮕ {$TGT_LIST}..."

    OUTPUT_FILE="$OUTPUT_DIR/${SRC}_residual_clt.pt"
    if [ -f "$OUTPUT_FILE" ]; then
        echo "⏭️  Weights already exist at $OUTPUT_FILE. Skipping..."
        continue
    fi

    BATCH_SIZE=4096
    if [[ "$SRC" == *"predictor"* ]]; then
        BATCH_SIZE=512
    fi

    python train_transcoder.py \
        --dir "$ACTIVATIONS_DIR" \
        --source_layer "$SRC" \
        --target_layer "$TGT_LIST" \
        --output "$OUTPUT_FILE" \
        --dict_size "$DICT_SIZE" \
        --batch_size "$BATCH_SIZE" \
        --l1 "$L1_COEFF" \
        --epochs "$EPOCHS"
done

echo "✨ Sweep complete. Weights stored in $OUTPUT_DIR"
