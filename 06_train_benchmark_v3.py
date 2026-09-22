"""
Stage 2a (v3): Benchmarking with proper ViT optimization recipe
====================================================================

Adds AdamW + linear-warmup-then-cosine-decay LR schedule, applied to ViT
models specifically (CNNs keep their original Adam/fixed-lr recipe from
Table 2, since that's not where the methodological gap was). This is the
standard, well-documented recipe for training/fine-tuning ViTs well (see
the DeiT training recipe) -- not a change aimed at producing any specific
winner, just fixing an actual under-specification in how ViTs were being
optimized.

Everything else (RandAugment, color jitter, random erasing, MixUp/CutMix,
leakage-safe data, val-based checkpoint selection, test touched once) is
unchanged from v2.

Usage
-----
    # ViT model with the new schedule:
    python 06_train_benchmark_v3.py --model_name vit_base_patch16_224 \
        --model_type vit --data_dir /workspace/data_final \
        --output_dir /workspace/results_v3/vit_base_patch16_224 \
        --epochs 10 --lr 0.001 --batch_size 16 --img_size 224 \
        --use_strong_aug --use_mixup \
        --optimizer adamw --weight_decay 0.05 \
        --use_warmup_cosine --warmup_epochs 2

    # CNN model, unchanged recipe:
    python 06_train_benchmark_v3.py --model_name resnet50 \
        --model_type cnn --data_dir /workspace/data_final \
        --output_dir /workspace/results_v3/resnet50 \
        --epochs 10 --lr 0.01 --batch_size 32 --img_size 224 \
        --use_strong_aug --use_mixup --optimizer adam
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import timm
from timm.data import create_transform, Mixup
from timm.loss import SoftTargetCrossEntropy

from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns


def build_train_transform(img_size, use_strong_aug):
    if use_strong_aug:
        return create_transform(
            input_size=img_size, is_training=True,
            auto_augment="rand-m9-mstd0.5-inc1", color_jitter=0.3,
            re_prob=0.25, re_mode="pixel", re_count=1, interpolation="bicubic")
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_eval_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_optimizer(name, params, lr, weight_decay):
    if name.lower() == "adam":
        return torch.optim.Adam(params, lr=lr)
    elif name.lower() == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif name.lower() == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {name}")


class WarmupCosineScheduler:
    """Linear warmup for `warmup_epochs`, then cosine decay to ~0 over the
    remaining epochs. Stepped once per epoch (simple, matches this study's
    per-epoch granularity)."""

    def __init__(self, optimizer, base_lr, warmup_epochs, total_epochs):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / max(1, self.warmup_epochs)
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


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
    p_per, r_per, f1_per, support_per = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)

    return {
        "accuracy": acc, "precision_macro": p_macro, "recall_macro": r_macro, "f1_macro": f1_macro,
        "per_class": {class_names[i]: {"precision": p_per[i], "recall": r_per[i],
                                        "f1": f1_per[i], "support": int(support_per[i])}
                      for i in range(len(class_names))},
        "confusion_matrix": cm.tolist(), "classification_report": report,
        "avg_latency_ms": float(np.mean(latencies)) * 1000,
    }


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def estimate_flops(model, img_size, device):
    try:
        from fvcore.nn import FlopCountAnalysis
        dummy = torch.randn(1, 3, img_size, img_size).to(device)
        return FlopCountAnalysis(model, dummy).total()
    except Exception as e:
        print(f"FLOPs estimation failed ({e}); skipping.")
        return None


def plot_curves(history, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_acc"], marker="o", label="Train Accuracy")
    axes[0].plot(history["val_acc"], marker="o", label="Val Accuracy")
    axes[0].set_title("Model Accuracy")
    axes[0].legend()
    axes[1].plot(history["train_loss"], marker="o", label="Train Loss")
    axes[1].plot(history["val_loss"], marker="o", label="Val Loss")
    axes[1].set_title("Model Loss")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_curves.png"))
    plt.close(fig)


def plot_confusion(cm, class_names, output_dir):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_type", choices=["cnn", "vit"], required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--optimizer", default="adam")
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_strong_aug", action="store_true")
    parser.add_argument("--use_mixup", action="store_true")
    parser.add_argument("--mixup_alpha", type=float, default=0.8)
    parser.add_argument("--cutmix_alpha", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--use_warmup_cosine", action="store_true",
                         help="Use linear warmup + cosine decay LR schedule (recommended for ViT)")
    parser.add_argument("--warmup_epochs", type=int, default=2)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = datasets.ImageFolder(
        os.path.join(args.data_dir, "train"),
        transform=build_train_transform(args.img_size, args.use_strong_aug))
    valid_ds = datasets.ImageFolder(
        os.path.join(args.data_dir, "valid"), transform=build_eval_transform(args.img_size))
    test_ds = datasets.ImageFolder(
        os.path.join(args.data_dir, "test"), transform=build_eval_transform(args.img_size))

    class_names = train_ds.classes
    num_classes = len(class_names)
    print(f"Classes: {class_names}")
    print(f"Train: {len(train_ds)}, Valid: {len(valid_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=4, drop_last=args.use_mixup)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = timm.create_model(args.model_name, pretrained=True, num_classes=num_classes).to(device)
    optimizer = get_optimizer(args.optimizer, model.parameters(), args.lr, args.weight_decay)

    scheduler = None
    if args.use_warmup_cosine:
        scheduler = WarmupCosineScheduler(optimizer, args.lr, args.warmup_epochs, args.epochs)

    mixup_fn = None
    train_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    if args.use_mixup:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha,
            prob=1.0, switch_prob=0.5, mode="batch",
            label_smoothing=args.label_smoothing, num_classes=num_classes)
        train_criterion = SoftTargetCrossEntropy()

    val_criterion = nn.CrossEntropyLoss()

    history = {"train_acc": [], "train_loss": [], "val_acc": [], "val_loss": [], "lr": []}
    best_val_acc, best_state = 0.0, None

    for epoch in range(args.epochs):
        current_lr = scheduler.step(epoch) if scheduler is not None else args.lr
        history["lr"].append(current_lr)

        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            hard_labels = labels.clone()

            if mixup_fn is not None:
                images, labels = mixup_fn(images, labels)

            optimizer.zero_grad()
            outputs = model(images)
            loss = train_criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == hard_labels).sum().item()
            total += images.size(0)

        train_loss, train_acc = running_loss / total, correct / total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = val_criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += images.size(0)
        val_loss, val_acc = val_loss / val_total, val_correct / val_total

        history["train_acc"].append(train_acc)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch+1}/{args.epochs} - lr: {current_lr:.6f} - "
              f"train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} - "
              f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))

    print("\n=== Final test set evaluation ===")
    test_results = evaluate(model, test_loader, device, class_names)
    print(f"Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"Test Macro F1: {test_results['f1_macro']:.4f}")
    print(test_results["classification_report"])

    n_params = count_params(model)
    flops = estimate_flops(model, args.img_size, device)

    plot_curves(history, args.output_dir)
    plot_confusion(np.array(test_results["confusion_matrix"]), class_names, args.output_dir)

    summary = {
        "model_name": args.model_name, "model_type": args.model_type,
        "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
        "optimizer": args.optimizer, "weight_decay": args.weight_decay,
        "use_strong_aug": args.use_strong_aug, "use_mixup": args.use_mixup,
        "use_warmup_cosine": args.use_warmup_cosine, "warmup_epochs": args.warmup_epochs,
        "best_val_acc": best_val_acc,
        "test_accuracy": test_results["accuracy"],
        "test_precision_macro": test_results["precision_macro"],
        "test_recall_macro": test_results["recall_macro"],
        "test_f1_macro": test_results["f1_macro"],
        "per_class": test_results["per_class"], "num_params": n_params,
        "flops": flops, "avg_latency_ms": test_results["avg_latency_ms"],
        "confusion_matrix": test_results["confusion_matrix"],
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved all results to {args.output_dir}")


if __name__ == "__main__":
    main()
