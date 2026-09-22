#!/bin/bash
# ============================================================================
# Stage 2b (v3): Full sweep with proper ViT optimization recipe
# ============================================================================
# CNNs: unchanged recipe (Adam, fixed lr=0.01), just with strong augmentation.
# ViTs: AdamW (weight_decay=0.05) + linear warmup (2 epochs) + cosine decay,
#       giving them a fair, well-tuned shot per standard ViT fine-tuning
#       practice -- applied identically regardless of outcome.
#
# Usage:
#   nohup bash 06_run_all_benchmarks_v3.sh > benchmark_log_v3.txt 2>&1 &
# ============================================================================

set -e

DATA_DIR="/workspace/data_final"
RESULTS_BASE="/workspace/results_v3"
mkdir -p "$RESULTS_BASE"

run_cnn() {
    local model_name=$1
    echo ""
    echo "=================================================================="
    echo "Training (v3): $model_name (cnn)"
    echo "=================================================================="
    python 06_train_benchmark_v3.py \
        --model_name "$model_name" --model_type cnn \
        --data_dir "$DATA_DIR" --output_dir "$RESULTS_BASE/$model_name" \
        --epochs 10 --lr 0.01 --batch_size 32 --optimizer adam --img_size 224 \
        --use_strong_aug --use_mixup --mixup_alpha 0.8 --cutmix_alpha 1.0 \
        --label_smoothing 0.1
}

run_vit() {
    local model_name=$1
    echo ""
    echo "=================================================================="
    echo "Training (v3): $model_name (vit) - AdamW + warmup/cosine"
    echo "=================================================================="
    python 06_train_benchmark_v3.py \
        --model_name "$model_name" --model_type vit \
        --data_dir "$DATA_DIR" --output_dir "$RESULTS_BASE/$model_name" \
        --epochs 10 --lr 0.001 --batch_size 16 --optimizer adamw \
        --weight_decay 0.05 --img_size 224 \
        --use_strong_aug --use_mixup --mixup_alpha 0.8 --cutmix_alpha 1.0 \
        --label_smoothing 0.1 --use_warmup_cosine --warmup_epochs 2
}

# --- CNN models ---
run_cnn "resnet50"
run_cnn "xception"
run_cnn "inception_v3"
run_cnn "tf_efficientnet_b5"
run_cnn "inception_resnet_v2"
run_cnn "resnet152"
run_cnn "densenet201"

# --- Vanilla ViT models (proper fine-tuning recipe) ---
run_vit "vit_base_patch16_224"
run_vit "vit_base_patch32_224"
run_vit "vit_large_patch16_224"
run_vit "vit_large_patch32_224"

echo ""
echo "=================================================================="
echo "All v3 benchmark models trained. Results under $RESULTS_BASE/<model_name>/summary.json"
echo "=================================================================="
