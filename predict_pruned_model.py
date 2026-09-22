"""
Predictions for a PRUNED model variant (adaptive or uniform).

The pruned checkpoints have a different internal architecture (fewer
attention heads per layer) than the vanilla ViT-B/16, so we must rebuild
the same structurally-pruned architecture before loading the saved
weights -- loading directly into a vanilla timm model will fail with a
shape mismatch.

This reuses the same pruning reconstruction logic as the training
pipeline: re-run the importance learning + head selection to get the
identical kept_heads_per_layer list that was used when the checkpoint was
created, then rebuild that exact architecture.

Usage
-----
    python predict_pruned_model.py \
        --variant adaptive \
        --importance_scores /workspace/results_v3/pruning_full/importance_scores.pt \
        --summary_json /workspace/results_v3/pruning_full/adaptive/summary.json \
        --checkpoint /workspace/results_v3/pruning_full/adaptive/pruned_model.pt \
        --data_dir /workspace/data_final \
        --output_file /workspace/preds_pruned_adaptive.npz \
        --img_size 224 --batch_size 8
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

from importlib import import_module
pruning_core = import_module("07_pruning_core")


def build_eval_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["adaptive", "uniform"], required=True)
    parser.add_argument("--summary_json", required=True,
                         help="Path to the variant's summary.json, which contains "
                              "kept_heads_per_layer needed to rebuild the exact architecture.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load the kept_heads_per_layer structure that defines this variant's architecture
    with open(args.summary_json) as f:
        summary = json.load(f)
    kept_heads = summary["kept_heads_per_layer"]
    print(f"Loaded architecture spec for '{args.variant}': "
          f"{sum(len(k) for k in kept_heads)} heads kept across {len(kept_heads)} layers")

    test_ds = datasets.ImageFolder(
        os.path.join(args.data_dir, "test"),
        transform=build_eval_transform(args.img_size))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    class_names = test_ds.classes
    print(f"Test set: {len(test_ds)} images, classes: {class_names}")

    # Rebuild the exact pruned architecture (gated wrapper -> structural pruning),
    # matching how it was built during training, before loading the checkpoint.
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=len(class_names))
    model = pruning_core.wrap_model_with_gates(model)
    model = pruning_core.apply_structural_pruning(model, kept_heads)

    state_dict = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    del state_dict
    model = model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"  batch {i+1}/{len(test_loader)} done")

    preds = np.array(all_preds)
    labels = np.array(all_labels)

    np.savez(args.output_file, preds=preds, labels=labels, class_names=class_names)
    acc = (preds == labels).mean()
    print(f"\nSaved predictions to {args.output_file}")
    print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
