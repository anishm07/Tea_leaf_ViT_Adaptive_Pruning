"""
Compute FLOPs for adaptive and uniform pruned architectures at multiple
sparsity levels. Since FLOPs depend only on model STRUCTURE (which heads
survive), not trained weights, this needs only the importance scores
(to determine which heads get pruned) -- no checkpoints, no dataset, no
GPU strictly required.

Usage
-----
    python compute_sweep_flops.py \
        --importance_scores /workspace/results_v3/pruning_full/importance_scores.pt \
        --sparsity_levels 0.15 0.30 0.45 \
        --img_size 224
"""

import argparse
import json

import timm
import torch

from importlib import import_module
pruning_core = import_module("07_pruning_core")


def estimate_flops(model, img_size, device):
    try:
        from fvcore.nn import FlopCountAnalysis
        dummy = torch.randn(1, 3, img_size, img_size).to(device)
        return FlopCountAnalysis(model, dummy).total()
    except Exception as e:
        print(f"FLOPs estimation failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--importance_scores", required=True)
    parser.add_argument("--sparsity_levels", type=float, nargs="+", required=True)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--output_json", default="/workspace/sweep_flops.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    importance_scores = torch.load(args.importance_scores, map_location="cpu")
    print(f"Loaded importance scores: shape {importance_scores.shape}\n")

    results = {}

    # Unpruned baseline for reference
    base_model = timm.create_model("vit_base_patch16_224", pretrained=False,
                                    num_classes=args.num_classes).to(device)
    base_flops = estimate_flops(base_model, args.img_size, device)
    results["unpruned"] = {"flops": base_flops}
    print(f"Unpruned: {base_flops / 1e9:.2f} GFLOPs\n")
    del base_model

    for sparsity in args.sparsity_levels:
        for scheme_name, ratio_fn in [
            ("adaptive", lambda: pruning_core.adaptive_ratios_at_sparsity(sparsity)),
            ("uniform", lambda: [sparsity] * importance_scores.shape[0]),
        ]:
            tag = f"{scheme_name}_{int(sparsity*100)}pct"
            ratios = ratio_fn()
            kept_heads = pruning_core.select_heads_to_keep(importance_scores, ratios)
            n_pruned, n_total = pruning_core.count_total_and_pruned_heads(kept_heads)

            model = timm.create_model("vit_base_patch16_224", pretrained=False,
                                       num_classes=args.num_classes)
            model = pruning_core.wrap_model_with_gates(model)
            model = pruning_core.apply_structural_pruning(model, kept_heads)
            model = model.to(device)

            flops = estimate_flops(model, args.img_size, device)
            results[tag] = {
                "sparsity_target": sparsity, "scheme": scheme_name,
                "heads_pruned": n_pruned, "heads_total": n_total,
                "flops": flops, "gflops": flops / 1e9 if flops else None,
            }
            print(f"{tag:20s} | heads_pruned={n_pruned}/{n_total} | "
                  f"FLOPs={flops/1e9:.2f} GFLOPs")

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output_json}")


if __name__ == "__main__":
    main()
