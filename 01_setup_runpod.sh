#!/bin/bash
# ============================================================================
# Stage 1a: Environment setup for RunPod
# ============================================================================
# Run this once on a fresh RunPod pod (recommend: a PyTorch template with
# CUDA 12.x preinstalled, at least 16GB VRAM e.g. RTX 4090 / A5000 tier is
# plenty for ViT-B/16 fine-tuning at this dataset scale).
#
# Usage:
#   bash 01_setup_runpod.sh
# ============================================================================

set -e  # stop on first error

echo "=== Updating system packages ==="
apt-get update -y
apt-get install -y wget unzip git

echo "=== Installing Python packages ==="
pip install --upgrade pip

# Core ML stack
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install timm==0.9.16
pip install transformers

# Data handling / augmentation
pip install pillow imagehash pandas numpy scikit-learn

# Metrics / efficiency measurement
pip install fvcore thop ptflops

# Experiment tracking (optional but recommended so results are reproducible
# and you have logs to cite in the paper -- e.g. "training curves logged via
# Weights & Biases")
pip install wandb

# Plotting for accuracy/loss curves and confusion matrices
pip install matplotlib seaborn

echo "=== Verifying GPU availability ==="
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

echo "=== Setup complete ==="
