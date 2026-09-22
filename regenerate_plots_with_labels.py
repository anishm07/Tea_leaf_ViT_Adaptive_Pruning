"""
Regenerate training curve plots WITH axis labels, parsed directly from the
saved log files -- no retraining needed.

Usage
-----
    python regenerate_plots_with_labels.py \
        --benchmark_log /workspace/benchmark_log_v3.txt \
        --pruning_log /workspace/pruning_log.txt \
        --output_dir /workspace/figures_labeled
"""

import argparse
import os
import re

import matplotlib.pyplot as plt


def parse_benchmark_log(log_path):
    """
    Returns {model_name: {"train_acc": [...], "val_acc": [...],
                           "train_loss": [...], "val_loss": [...]}}
    by scanning for 'Training (v3): <model_name> (...)' headers and the
    epoch lines that follow, up until the next such header.
    """
    with open(log_path, "r") as f:
        text = f.read()

    # Split the log into per-model chunks using the training header line
    header_pattern = re.compile(r"Training \(v3\):\s*(\S+)\s*\(")
    headers = list(header_pattern.finditer(text))

    epoch_pattern = re.compile(
        r"Epoch\s+\d+/\d+.*?train_loss:\s*([\d.]+)\s*train_acc:\s*([\d.]+)\s*-\s*"
        r"val_loss:\s*([\d.]+)\s*val_acc:\s*([\d.]+)"
    )

    results = {}
    for i, h in enumerate(headers):
        model_name = h.group(1)
        start = h.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        chunk = text[start:end]

        epochs = epoch_pattern.findall(chunk)
        if not epochs:
            continue

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        for train_loss, train_acc, val_loss, val_acc in epochs:
            history["train_loss"].append(float(train_loss))
            history["train_acc"].append(float(train_acc))
            history["val_loss"].append(float(val_loss))
            history["val_acc"].append(float(val_acc))

        # Keep the LAST occurrence of each model name (in case of re-runs)
        results[model_name] = history

    return results


def parse_pruning_log(log_path):
    """
    Returns {"adaptive": {...}, "uniform": {...}} in the same history format,
    by scanning for 'Variant: <name>' headers.
    """
    with open(log_path, "r") as f:
        text = f.read()

    header_pattern = re.compile(r"Variant:\s*(adaptive|uniform)")
    headers = list(header_pattern.finditer(text))

    epoch_pattern = re.compile(
        r"Epoch\s+\d+/\d+\s*-\s*train_acc:\s*([\d.]+)\s*-\s*val_acc:\s*([\d.]+)"
    )

    results = {}
    for i, h in enumerate(headers):
        variant = h.group(1)
        start = h.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        chunk = text[start:end]

        epochs = epoch_pattern.findall(chunk)
        if not epochs:
            continue

        history = {"train_acc": [], "val_acc": []}
        for train_acc, val_acc in epochs:
            history["train_acc"].append(float(train_acc))
            history["val_acc"].append(float(val_acc))

        results[variant] = history

    return results


def plot_benchmark_curves(model_name, history, output_dir):
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
                          "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11})
    fig, axes = plt.subplots(2, 1, figsize=(6, 9))

    epochs = range(1, len(history["train_acc"]) + 1)

    axes[0].plot(epochs, history["train_acc"], marker="o", linewidth=2, markersize=6, label="Train Accuracy")
    axes[0].plot(epochs, history["val_acc"], marker="o", linewidth=2, markersize=6, label="Val Accuracy")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_loss"], marker="o", linewidth=2, markersize=6, label="Train Loss")
    axes[1].plot(epochs, history["val_loss"], marker="o", linewidth=2, markersize=6, label="Val Loss")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(output_dir, f"{model_name}_curves_labeled.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_pruning_curves(variant_name, history, output_dir):
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
                          "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11})
    fig, ax = plt.subplots(figsize=(6, 4.5))
    epochs = range(1, len(history["train_acc"]) + 1)
    ax.plot(epochs, history["train_acc"], marker="o", linewidth=2, markersize=6, label="Train Accuracy")
    ax.plot(epochs, history["val_acc"], marker="o", linewidth=2, markersize=6, label="Val Accuracy")
    ax.set_title(f"Pruned ViT-B/16 ({variant_name.capitalize()})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(output_dir, f"pruned_{variant_name}_curves_labeled.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_log", default=None)
    parser.add_argument("--pruning_log", default=None)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.benchmark_log and os.path.exists(args.benchmark_log):
        print(f"Parsing {args.benchmark_log} ...")
        benchmark_histories = parse_benchmark_log(args.benchmark_log)
        print(f"Found {len(benchmark_histories)} models: {list(benchmark_histories.keys())}")
        for model_name, history in benchmark_histories.items():
            plot_benchmark_curves(model_name, history, args.output_dir)

    if args.pruning_log and os.path.exists(args.pruning_log):
        print(f"Parsing {args.pruning_log} ...")
        pruning_histories = parse_pruning_log(args.pruning_log)
        print(f"Found variants: {list(pruning_histories.keys())}")
        for variant_name, history in pruning_histories.items():
            plot_pruning_curves(variant_name, history, args.output_dir)

    print(f"\nAll labeled plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
