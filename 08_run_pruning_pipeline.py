"""
Stage 3b: Full pruning pipeline
==================================

Runs the complete Algorithm 1 pipeline on the fine-tuned ViT-B/16 model
(from the benchmarking stage), then produces TWO pruned models from the
SAME learned importance scores:

  1. "adaptive" - your proposed layer-wise ratios (25%/35%/30%)
  2. "uniform"  - a flat ratio per layer, matched to the SAME overall
                  sparsity as the adaptive scheme

Comparing these two head-to-head, evaluated identically, is the ablation
that determines whether the adaptive layer-wise scheme is actually
contributing anything over a naive uniform baseline -- this is the
experiment your original draft was missing.

Prerequisite: run 06_run_all_benchmarks.sh first so that
/workspace/results/vit_base_patch16_224/best_model.pt exists (this is
used as both the pruning starting point AND the frozen KD teacher).

Usage
-----
    python 08_run_pruning_pipeline.py \
        --vit_checkpoint /workspace/results/vit_base_patch16_224/best_model.pt \
        --data_dir /workspace/data_final \
        --output_dir /workspace/results/pruning \
        --search_epochs 5 \
        --finetune_epochs 10
"""

import argparse
import copy
import json
import os
import time

import numpy as np
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

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
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4),
        DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=4),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4),
        train_ds.classes,
    )


def evaluate(model, loader, device, class_names):
    model.eval()
    all_preds, all_labels, latencies = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            start = time.perf_counter()
            outputs = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) / images.size(0))
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    return {
        "accuracy": acc, "precision_macro": p_macro, "recall_macro": r_macro,
        "f1_macro": f1_macro, "confusion_matrix": cm.tolist(),
        "classification_report": report, "avg_latency_ms": float(np.mean(latencies)) * 1000,
    }


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def estimate_flops(model, img_size, device):
    try:
        from fvcore.nn import FlopCountAnalysis
        dummy = torch.randn(1, 3, img_size, img_size).to(device)
        return FlopCountAnalysis(model, dummy).total()
    except Exception as e:
        print(f"FLOPs estimation failed: {e}")
        return None


def finetune(model, train_loader, valid_loader, device, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = {"train_acc": [], "train_loss": [], "val_acc": [], "val_loss": []}
    best_val_acc, best_state = 0.0, None

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)
        train_loss, train_acc = running_loss / total, correct / total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += images.size(0)
        val_loss, val_acc = val_loss / val_total, val_correct / val_total

        history["train_acc"].append(train_acc)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)
        print(f"  Epoch {epoch+1}/{epochs} - train_acc: {train_acc:.4f} - val_acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_val_acc


def save_variant_results(name, model, history, test_results, n_params, flops,
                          kept_heads, output_dir, class_names):
    variant_dir = os.path.join(output_dir, name)
    os.makedirs(variant_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(variant_dir, "pruned_model.pt"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_acc"], marker="o", label="Train Acc")
    axes[0].plot(history["val_acc"], marker="o", label="Val Acc")
    axes[0].set_title(f"{name}: Accuracy")
    axes[0].legend()
    axes[1].plot(history["train_loss"], marker="o", label="Train Loss")
    axes[1].plot(history["val_loss"], marker="o", label="Val Loss")
    axes[1].set_title(f"{name}: Loss")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(variant_dir, "finetune_curves.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(np.array(test_results["confusion_matrix"]), annot=True, fmt="d",
                cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f"{name}: Confusion Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(variant_dir, "confusion_matrix.png"))
    plt.close(fig)

    n_pruned, n_total = pruning_core.count_total_and_pruned_heads(kept_heads)

    summary = {
        "variant": name,
        "heads_pruned": n_pruned,
        "heads_total": n_total,
        "heads_pruned_pct": round(100 * n_pruned / n_total, 2),
        "kept_heads_per_layer": kept_heads,
        "test_accuracy": test_results["accuracy"],
        "test_precision_macro": test_results["precision_macro"],
        "test_recall_macro": test_results["recall_macro"],
        "test_f1_macro": test_results["f1_macro"],
        "num_params": n_params,
        "flops": flops,
        "avg_latency_ms": test_results["avg_latency_ms"],
        "classification_report": test_results["classification_report"],
    }
    with open(os.path.join(variant_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[{name}] heads pruned: {n_pruned}/{n_total} "
          f"({summary['heads_pruned_pct']}%) | test_acc: {test_results['accuracy']:.4f} "
          f"| params: {n_params:,} | flops: {flops}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vit_checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--search_epochs", type=int, default=5)
    parser.add_argument("--search_lr", type=float, default=0.01)
    parser.add_argument("--reg_weight", type=float, default=1e-4)
    parser.add_argument("--kd_temperature", type=float, default=4.0)
    parser.add_argument("--finetune_epochs", type=int, default=10)
    parser.add_argument("--finetune_lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, valid_loader, test_loader, class_names = load_dataloaders(
        args.data_dir, args.img_size, args.batch_size)
    print(f"Classes: {class_names}")

    # --- Load the fine-tuned teacher (frozen) ---
    print("Loading frozen teacher model...")
    teacher = timm.create_model("vit_base_patch16_224", pretrained=False,
                                 num_classes=len(class_names))
    teacher.load_state_dict(torch.load(args.vit_checkpoint, map_location=device))
    teacher = teacher.to(device)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    # --- Build gated student from a fresh copy of the same weights ---
    print("Building gated student model...")
    student = timm.create_model("vit_base_patch16_224", pretrained=False,
                                 num_classes=len(class_names))
    student.load_state_dict(torch.load(args.vit_checkpoint, map_location=device))
    student = pruning_core.wrap_model_with_gates(student).to(device)
    pruning_core.freeze_backbone_except_zeta(student)

    # --- Importance learning (search phase) ---
    print(f"\n=== Importance learning: {args.search_epochs} epochs ===")
    zeta_params = pruning_core.get_all_zeta_params(student)
    optimizer = torch.optim.Adam(zeta_params, lr=args.search_lr)

    for epoch in range(args.search_epochs):
        epoch_losses = []
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            stats = pruning_core.importance_learning_step(
                student, teacher, images, labels, optimizer,
                reg_weight=args.reg_weight, temperature=args.kd_temperature)
            epoch_losses.append(stats["loss"])
        print(f"  Search epoch {epoch+1}/{args.search_epochs} - "
              f"avg_loss: {np.mean(epoch_losses):.4f}")

    importance_scores = pruning_core.extract_importance_scores(student)
    torch.save(importance_scores, os.path.join(args.output_dir, "importance_scores.pt"))
    print(f"\nImportance scores extracted. Shape: {importance_scores.shape}")
    print(f"Per-layer mean importance: {importance_scores.mean(dim=1).tolist()}")

    # --- Produce both pruning variants from the SAME importance scores ---
    variants = {
        "adaptive": pruning_core.adaptive_layerwise_ratios(num_layers=len(student.blocks)),
        "uniform": pruning_core.uniform_ratios(num_layers=len(student.blocks)),
    }

    all_summaries = {}

    for variant_name, ratios in variants.items():
        print(f"\n{'='*70}\nVariant: {variant_name} (ratios: {ratios})\n{'='*70}")

        kept_heads = pruning_core.select_heads_to_keep(importance_scores, ratios)

        # Rebuild a fresh gated model, apply structural pruning, then finetune
        pruned_model = timm.create_model("vit_base_patch16_224", pretrained=False,
                                          num_classes=len(class_names))
        pruned_model.load_state_dict(torch.load(args.vit_checkpoint, map_location=device))
        pruned_model = pruning_core.wrap_model_with_gates(pruned_model)
        pruned_model = pruning_core.apply_structural_pruning(pruned_model, kept_heads)
        pruned_model = pruned_model.to(device)

        # Unfreeze everything for fine-tuning
        for p in pruned_model.parameters():
            p.requires_grad = True

        print(f"Fine-tuning {variant_name} pruned model for {args.finetune_epochs} epochs...")
        pruned_model, history, best_val_acc = finetune(
            pruned_model, train_loader, valid_loader, device,
            args.finetune_epochs, args.finetune_lr)

        print(f"Evaluating {variant_name} on held-out test set...")
        test_results = evaluate(pruned_model, test_loader, device, class_names)

        n_params = count_params(pruned_model)
        flops = estimate_flops(pruned_model, args.img_size, device)

        summary = save_variant_results(
            variant_name, pruned_model, history, test_results,
            n_params, flops, kept_heads, args.output_dir, class_names)
        all_summaries[variant_name] = summary

    with open(os.path.join(args.output_dir, "all_variants_summary.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)

    print("\n" + "=" * 70)
    print("FINAL COMPARISON (adaptive vs. uniform, same overall sparsity)")
    print("=" * 70)
    for name, s in all_summaries.items():
        print(f"{name:10s} | heads pruned: {s['heads_pruned']}/{s['heads_total']} "
              f"({s['heads_pruned_pct']}%) | test_acc: {s['test_accuracy']:.4f} "
              f"| macro_f1: {s['test_f1_macro']:.4f} | params: {s['num_params']:,} "
              f"| flops: {s['flops']} | latency_ms: {s['avg_latency_ms']:.3f}")


if __name__ == "__main__":
    main()
