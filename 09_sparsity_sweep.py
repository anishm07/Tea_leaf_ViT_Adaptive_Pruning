"""
Sparsity sweep: adaptive vs. uniform pruning at multiple overall sparsity
levels, reusing the already-computed importance scores (no need to
re-run the 5-epoch search phase).

For each target sparsity, builds both an adaptive-ratio and a uniform-ratio
pruned model from the SAME importance scores, fine-tunes each for 10
epochs, evaluates, and saves per-image predictions for McNemar's test.

Usage
-----
    python 09_sparsity_sweep.py \
        --vit_checkpoint /workspace/results_v3/vit_base_patch16_224/best_model.pt \
        --importance_scores /workspace/results_v3/pruning_full/importance_scores.pt \
        --data_dir /workspace/data_final \
        --output_dir /workspace/results_v3/sparsity_sweep \
        --sparsity_levels 0.15 0.45 \
        --finetune_epochs 10
"""

import argparse
import json
import os

import numpy as np
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              confusion_matrix, classification_report)

from importlib import import_module
pruning_core = import_module("07_pruning_core")


def build_transforms(img_size, is_train):
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_dataloaders(data_dir, img_size, batch_size):
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"),
                                     transform=build_transforms(img_size, True))
    valid_ds = datasets.ImageFolder(os.path.join(data_dir, "valid"),
                                     transform=build_transforms(img_size, False))
    test_ds = datasets.ImageFolder(os.path.join(data_dir, "test"),
                                    transform=build_transforms(img_size, False))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=0),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0),
        train_ds.classes,
    )


def finetune(model, train_loader, valid_loader, device, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_acc, best_state = 0.0, None

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += images.size(0)
        val_acc = val_correct / val_total
        print(f"    epoch {epoch+1}/{epochs} - val_acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_acc


def evaluate_and_predict(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vit_checkpoint", required=True)
    parser.add_argument("--importance_scores", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sparsity_levels", type=float, nargs="+", required=True,
                         help="e.g. --sparsity_levels 0.15 0.45")
    parser.add_argument("--finetune_epochs", type=int, default=10)
    parser.add_argument("--finetune_lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, valid_loader, test_loader, class_names = load_dataloaders(
        args.data_dir, args.img_size, args.batch_size)
    print(f"Classes: {class_names}")

    importance_scores = torch.load(args.importance_scores, map_location="cpu")
    print(f"Loaded importance scores: shape {importance_scores.shape}")

    all_results = {}

    for sparsity in args.sparsity_levels:
        for scheme_name, ratio_fn in [
            ("adaptive", lambda: pruning_core.adaptive_ratios_at_sparsity(sparsity)),
            ("uniform", lambda: [sparsity] * importance_scores.shape[0]),
        ]:
            tag = f"{scheme_name}_{int(sparsity*100)}pct"
            print(f"\n{'='*60}\n{tag}\n{'='*60}")

            ratios = ratio_fn()
            kept_heads = pruning_core.select_heads_to_keep(importance_scores, ratios)
            n_pruned, n_total = pruning_core.count_total_and_pruned_heads(kept_heads)
            print(f"Ratios: {ratios}")
            print(f"Heads pruned: {n_pruned}/{n_total}")

            model = timm.create_model("vit_base_patch16_224", pretrained=False,
                                       num_classes=len(class_names))
            model.load_state_dict(torch.load(args.vit_checkpoint, map_location=device))
            model = pruning_core.wrap_model_with_gates(model)
            model = pruning_core.apply_structural_pruning(model, kept_heads)
            model = model.to(device)
            for p in model.parameters():
                p.requires_grad = True

            print(f"Fine-tuning for {args.finetune_epochs} epochs...")
            model, best_val_acc = finetune(model, train_loader, valid_loader, device,
                                            args.finetune_epochs, args.finetune_lr)

            preds, labels = evaluate_and_predict(model, test_loader, device)
            acc = accuracy_score(labels, preds)
            _, _, f1_macro, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)

            np.savez(os.path.join(args.output_dir, f"preds_{tag}.npz"),
                     preds=preds, labels=labels, class_names=class_names)

            result = {
                "sparsity_target": sparsity, "scheme": scheme_name,
                "ratios": ratios, "heads_pruned": n_pruned, "heads_total": n_total,
                "test_accuracy": float(acc), "test_f1_macro": float(f1_macro),
                "best_val_acc": best_val_acc,
            }
            all_results[tag] = result
            print(f"[{tag}] test_acc: {acc:.4f} | macro_f1: {f1_macro:.4f}")

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    with open(os.path.join(args.output_dir, "sweep_summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}\nSPARSITY SWEEP SUMMARY\n{'='*60}")
    for tag, r in sorted(all_results.items()):
        print(f"{tag:20s} | sparsity={r['sparsity_target']:.0%} | "
              f"heads_pruned={r['heads_pruned']}/{r['heads_total']} | "
              f"acc={r['test_accuracy']:.4f} | macro_f1={r['test_f1_macro']:.4f}")


if __name__ == "__main__":
    main()
