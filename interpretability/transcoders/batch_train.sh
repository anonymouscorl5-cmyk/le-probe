#!/bin/bash
# 🚀 SAE BATCH SWEEP ENGINE
# Role: Trains an SAE for every layer in the model automatically.

# 1. Configuration
ACTIVATIONS_DIR="activations_granular"
OUTPUT_DIR="transcoder_weights_residual"
DICT_SIZE=12288
L1_COEFF=3e-3
EPOCHS=10
WINDOW_SIZE=1 # Target current layer + 1 before + 1 after

mkdir -p $OUTPUT_DIR

echo "🔥 Starting Residual Crosscoder Sweep..."

# Define the full hierarchical sequence
LAYERS=(
    "encoder_L0" "encoder_L1" "encoder_L2" "encoder_L3" "encoder_L4" "encoder_L5"
    "encoder_L6" "encoder_L7" "encoder_L8" "encoder_L9" "encoder_L10" "encoder_L11"
    "predictor_L0" "predictor_L1" "predictor_L2" "predictor_L3" "predictor_L4" "predictor_L5"
)

NUM_LAYERS=${#LAYERS[@]}

for i in $(seq 0 $((NUM_LAYERS - 1))); do
    SRC=${LAYERS[$i]}
    
    # Construct Target Window (L-k to L+k)
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

    # Use smaller batch size for predictor-involved transcoders to ensure updates
    BATCH_SIZE=4096
    if [[ "$SRC" == *"predictor"* ]]; then
        BATCH_SIZE=512
    fi

    python train_transcoder.py \
        --dir $ACTIVATIONS_DIR \
        --source_layer "$SRC" \
        --target_layer "$TGT_LIST" \
        --output "$OUTPUT_DIR/${SRC}_residual_clt.pt" \
        --dict_size $DICT_SIZE \
        --batch_size $BATCH_SIZE \
        --l1 $L1_COEFF \
        --epochs $EPOCHS
done

echo "✨ Sweep Complete! Weights stored in $OUTPUT_DIR"
