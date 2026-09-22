# Optimizing Vision Transformers for Tea Leaf Disease Detection: A Comparative Study of Adaptive and Uniform Attention-Head Pruning

This repository contains the full experimental pipeline supporting the paper
*"Optimizing Vision Transformers for Tea Leaf Disease Detection: A
Comparative Study of Adaptive and Uniform Attention-Head Pruning"*

## What this repository contains

A reproducible, end-to-end pipeline covering:
1. Dataset acquisition, leakage verification, and finalization
2. Benchmarking of 7 CNN and 4 vanilla ViT architectures
3. Adaptive and uniform attention-head pruning of ViT-B/16, across three
   sparsity levels (15%, 30%, 45%)
4. Statistical significance testing (McNemar's test) of all key comparisons
5. Figure regeneration utilities

**Dataset used:** K-Kotagiri Tea Leaf Dataset, publicly available at
Mendeley Data: https://data.mendeley.com/datasets/68yy284sh9/1

## Repository structure

```
.
├── data_preparation/
│   ├── 01_setup_runpod.sh              # Environment setup (PyTorch, timm, etc.)
│   ├── 02_prepare_dataset.py           # Extract and organize the raw dataset
│   ├── 03_grouped_split.py             # Perceptual-hash-based grouped split (superseded, see note below)
│   ├── 04_verify_near_duplicate_leakage.py  # Exact + perceptual hash leakage verification
│   ├── 05_finalize_dataset.py          # Deduplicate and build final train/valid/test folders
│   └── archive_exploratory/
│       └── grouped_split_by_similarity.py   # Early exploratory approach, not used in final pipeline
│
├── benchmarking/
│   ├── 06_train_benchmark_v3.py        # FINAL benchmark training script (used for all reported results)
│   ├── 06_run_all_benchmarks_v3.sh     # Sweep runner for all 11 CNN/ViT models
│   └── archive_earlier_iterations/     # v1, v2, v4 -- kept for transparency of the full
│                                        # experimental process; NOT used for reported numbers
│
├── pruning/
│   ├── 07_pruning_core.py              # Zeta-gated attention head pruning core (importance
│   │                                    # learning, structural pruning, ratio schemes)
│   ├── 08_run_pruning_pipeline.py      # Full pipeline: importance learning + adaptive/uniform
│   │                                    # pruning at 30% sparsity + fine-tuning
│   ├── 09_sparsity_sweep.py            # Adaptive vs. uniform pruning at 15%/30%/45% sparsity,
│   │                                    # reusing pre-computed importance scores
│   └── compute_sweep_flops.py          # FLOPs computation for each pruned architecture
│
├── evaluation/
│   ├── predict_single_model.py         # Extract per-image predictions from a benchmark checkpoint
│   ├── predict_pruned_model.py         # Extract per-image predictions from a pruned checkpoint
│   │                                    # (rebuilds the correct structurally-pruned architecture)
│   ├── compare_predictions.py          # McNemar's test from two saved prediction files
│   └── mcnemar_test.py                 # Combined single-script version (loads two models directly)
│
└── figures/
    └── regenerate_plots_with_labels.py # Regenerate labeled, legible training curve figures
                                         # from saved training logs (no retraining needed)
```

## Pipeline order (for full reproduction)

```bash
# 1. Environment setup (run on a GPU instance, e.g. RunPod with an RTX 4090)
bash data_preparation/01_setup_runpod.sh

# 2. Download the K-Kotagiri dataset from Mendeley Data (manual step, see link above),
#    then organize it
python data_preparation/02_prepare_dataset.py --zip_path <path_to_zip> --output_dir /workspace/data

# 3. Verify leakage-free split (exact + perceptual hash)
python data_preparation/04_verify_near_duplicate_leakage.py --base_dir <dataset_path>

# 4. Finalize deduplicated train/valid/test folders
python data_preparation/05_finalize_dataset.py --base_dir <dataset_path> --output_dir /workspace/data_final

# 5. Benchmark all 11 CNN/ViT models
bash benchmarking/06_run_all_benchmarks_v3.sh

# 6. Run the primary pruning pipeline (30% sparsity, adaptive + uniform)
python pruning/08_run_pruning_pipeline.py \
    --vit_checkpoint /workspace/results/vit_base_patch16_224/best_model.pt \
    --data_dir /workspace/data_final \
    --output_dir /workspace/results/pruning \
    --search_epochs 5 --finetune_epochs 10

# 7. Run the full sparsity sweep (15%, 45%; reuses importance scores from step 6)
python pruning/09_sparsity_sweep.py \
    --vit_checkpoint /workspace/results/vit_base_patch16_224/best_model.pt \
    --importance_scores /workspace/results/pruning/importance_scores.pt \
    --data_dir /workspace/data_final \
    --output_dir /workspace/results/sparsity_sweep \
    --sparsity_levels 0.15 0.45 --finetune_epochs 10

# 8. Compute FLOPs at each sparsity level
python pruning/compute_sweep_flops.py \
    --importance_scores /workspace/results/pruning/importance_scores.pt \
    --sparsity_levels 0.15 0.30 0.45

# 9. Statistical significance testing (example: ResNet-50 vs. ViT-B/16)
python evaluation/predict_single_model.py --model_name resnet50 ...
python evaluation/predict_single_model.py --model_name vit_base_patch16_224 ...
python evaluation/compare_predictions.py --preds_a <path> --name_a resnet50 --preds_b <path> --name_b vit_base_patch16_224
```

See each script's docstring/`--help` for full argument details.

## Key methodological notes

- **Data leakage verification**: the dataset's augmented images carry no
  metadata linking them to their source original image. We verified the
  official Train/Test/Valid split is leakage-safe using both exact-hash
  (MD5) and perceptual-hash (pHash) comparison — see
  `04_verify_near_duplicate_leakage.py`.
- **Structural (not masked) pruning**: pruned attention heads are
  physically removed from the `W_qkv`/`W_proj` weight matrices, so
  reported FLOPs reflect genuine computational savings, not a nominal
  reduction.
- **Statistical testing**: all headline accuracy comparisons in the paper
  are backed by McNemar's test on paired per-image predictions, not raw
  accuracy differences alone.

## Environment

- Python 3.12, PyTorch, `timm==0.9.16` (pinned — a newer `timm` changes
  the `Block.forward()` signature and will break the custom pruning
  modules in `07_pruning_core.py`)
- `fvcore` (FLOPs counting), `scipy`, `scikit-learn`, `matplotlib`
- Trained on a single NVIDIA RTX 4090 (24GB VRAM)

## Citation

If you use this code, please cite:

```bibtex
@misc{george2026tealeafcode,
  title={Code for: Optimizing Vision Transformers for Tea Leaf Disease Detection: A Comparative Study of Adaptive and Uniform Attention-Head Pruning},
  author={George, Anish M and John, Shajimon K},
  year={2026},
  note={Software, archived at Zenodo. DOI: <https://doi.org/10.5281/zenodo.22900492>}
}

@inproceedings{george2026efficient,
  title={Efficient Pretrained Vision Transformers for Agricultural Disease Detection via Structured Attention-Head Pruning},
  author={George, Anish M and John, Shajimon K},
  booktitle={International Conference on Smart Communication and Sustainable Technologies (ICSCST)},
  year={2026},
  note={in press}
}
```

## License

[Choose and add a license here, e.g. MIT — see https://choosealicense.com/]

## Data Availability

The dataset supporting this study is available at Mendeley Data:
https://data.mendeley.com/datasets/68yy284sh9/1
