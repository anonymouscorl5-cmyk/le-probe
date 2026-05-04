#!/bin/bash
# 🚀 SAE BATCH SWEEP ENGINE
# Role: Trains an SAE for every layer in the model automatically.

# 1. Configuration
ACTIVATIONS_DIR="activations_granular"
OUTPUT_DIR="transcoder_weights_granular"
DICT_SIZE=12288
EPOCHS=10

# 🚀 MODULE CONTROL (Set to false to skip sections)
TRAIN_ENCODER=true
TRAIN_BRIDGE=true
TRAIN_PREDICTOR=true

mkdir -p $OUTPUT_DIR

echo "🔥 Starting Layered SAE Sweep..."

# --- ENCODER SWEEP ---
if [ "$TRAIN_ENCODER" = true ]; then
    for i in {0..11}; do
        LAYER="encoder_L$i"
        echo "⚙️ Training Identity SAE for $LAYER..."
        python train_transcoder.py \
            --dir $ACTIVATIONS_DIR \
            --source_layer $LAYER \
            --target_layer $LAYER \
            --output "$OUTPUT_DIR/${LAYER}_sae.pt" \
            --dict_size $DICT_SIZE \
            --epochs $EPOCHS
    done
fi

# --- THE BRIDGE: Predictor Entry ---
# In Residual SAE mode, we just train predictor_L0 as its own SAE
if [ "$TRAIN_BRIDGE" = true ]; then
    echo "⚙️ Training Predictor Entry SAE: predictor_L0..."
    python train_transcoder.py \
        --dir $ACTIVATIONS_DIR \
        --source_layer predictor_L0 \
        --target_layer predictor_L0 \
        --output "$OUTPUT_DIR/predictor_L0_sae.pt" \
        --dict_size $DICT_SIZE \
        --epochs $EPOCHS
fi

# --- PREDICTOR SWEEP (High-Fidelity) ---
if [ "$TRAIN_PREDICTOR" = true ]; then
    # The Predictor has 257x fewer tokens than the Encoder.
    # We use a smaller batch size (512) to ensure enough gradient updates per epoch.
    for i in {1..5}; do
        LAYER="predictor_L$i"
        echo "⚙️ Training High-Fidelity Identity SAE for $LAYER..."
        python train_transcoder.py \
            --dir $ACTIVATIONS_DIR \
            --source_layer $LAYER \
            --target_layer $LAYER \
            --output "$OUTPUT_DIR/${LAYER}_sae.pt" \
            --dict_size $DICT_SIZE \
            --batch_size 512 \
            --epochs $EPOCHS
    done
fi

echo "✨ Sweep Complete! Weights stored in $OUTPUT_DIR"
