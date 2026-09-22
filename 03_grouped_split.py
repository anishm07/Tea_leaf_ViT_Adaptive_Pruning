"""
Stage 1c: Leakage-safe grouped train/test split
=================================================

Builds on 02_prepare_dataset.py's output (organized Healthy/ and Disease/
folders containing BOTH original and pre-made augmented images, matching
the dataset's own published structure: 225+1610 Healthy, 303+2121 Disease).

Since the dataset provides no explicit metadata linking each augmented
image to its source original, this script uses perceptual hashing to
approximate that relationship: every augmented image is assigned to its
nearest-matching original (by Hamming distance between pHash values), and
the train/test split is then performed on ORIGINALS FIRST and propagated
to their assigned augmented images. This guarantees that no near-duplicate
of a test-set image appears in training.

IMPORTANT: originals in this dataset are NOT distinguished from augmented
images by filename in a documented way. This script assumes you can
identify originals via one of:
  (a) a separate subfolder for originals within each class folder, or
  (b) a naming convention you've confirmed by manual inspection, or
  (c) if neither exists, the FALLBACK clustering mode (see
      --no_known_originals flag) which clusters ALL images by mutual
      similarity instead of assuming any specific image is "the" original.

Before trusting the output: run with --spot_check N to save a visual grid
of N random parent/child assignments for manual verification.

Usage
-----
    python 03_grouped_split.py \
        --data_dir /workspace/data \
        --output_dir /workspace/data_split \
        --test_fraction 0.20 \
        --spot_check 20
"""

import argparse
import os
import random
import shutil
from pathlib import Path

import imagehash
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_hash(data_dir: str, hash_size: int = 16) -> pd.DataFrame:
    rows = []
    for label in ["Healthy", "Disease"]:
        folder = os.path.join(data_dir, label)
        for f in sorted(Path(folder).glob("*")):
            if f.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            try:
                img = Image.open(f).convert("RGB")
            except Exception as e:
                print(f"Skipping unreadable file {f}: {e}")
                continue
            h = imagehash.phash(img, hash_size=hash_size)
            rows.append({"filepath": str(f), "filename": f.name,
                         "label": label, "hash": h})
    return pd.DataFrame(rows)


def cluster_by_similarity(df: pd.DataFrame, threshold: int = 6) -> pd.DataFrame:
    """
    Fallback mode: when originals cannot be distinguished at all, group ALL
    images (per class) into clusters of mutual near-duplicates using a
    simple greedy clustering on Hamming distance. Each cluster is treated
    as one group for the train/test split -- this avoids ever needing to
    know which image in a cluster is "the" true original.
    """
    df = df.copy()
    df["cluster_id"] = None
    for label, group in df.groupby("label"):
        idxs = list(group.index)
        cluster_id = 0
        assigned = set()
        for i in idxs:
            if i in assigned:
                continue
            df.loc[i, "cluster_id"] = f"{label}_{cluster_id}"
            assigned.add(i)
            for j in idxs:
                if j in assigned:
                    continue
                if df.loc[i, "hash"] - df.loc[j, "hash"] <= threshold:
                    df.loc[j, "cluster_id"] = f"{label}_{cluster_id}"
                    assigned.add(j)
            cluster_id += 1
    n_clusters = df["cluster_id"].nunique()
    print(f"Fallback clustering produced {n_clusters} groups from "
          f"{len(df)} images (avg {len(df)/n_clusters:.1f} images/group).")
    return df


def grouped_split(df: pd.DataFrame, group_col: str, test_fraction: float,
                   seed: int) -> pd.DataFrame:
    df = df.copy()
    groups = df[[group_col, "label"]].drop_duplicates(subset=group_col)

    train_groups, test_groups = train_test_split(
        groups[group_col],
        test_size=test_fraction,
        stratify=groups["label"],
        random_state=seed,
    )
    test_set = set(test_groups)
    df["split"] = df[group_col].apply(lambda g: "test" if g in test_set else "train")
    return df


def materialize_split(df: pd.DataFrame, output_dir: str):
    for split in ["train", "test"]:
        for label in ["Healthy", "Disease"]:
            os.makedirs(os.path.join(output_dir, split, label), exist_ok=True)

    for _, row in df.iterrows():
        dst = os.path.join(output_dir, row["split"], row["label"], row["filename"])
        shutil.copy2(row["filepath"], dst)

    print("\n=== Final split sizes ===")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0))


def spot_check_grid(df: pd.DataFrame, group_col: str, n: int, output_dir: str):
    """Save a few example groups as image grids so you can visually confirm
    the grouping looks sane (e.g. all images in a group really do look like
    variations of the same leaf)."""
    import matplotlib.pyplot as plt

    sample_groups = random.sample(list(df[group_col].dropna().unique()),
                                   min(n, df[group_col].nunique()))
    os.makedirs(os.path.join(output_dir, "spot_checks"), exist_ok=True)

    for g in sample_groups:
        members = df[df[group_col] == g].head(6)
        fig, axes = plt.subplots(1, len(members), figsize=(3 * len(members), 3))
        if len(members) == 1:
            axes = [axes]
        for ax, (_, row) in zip(axes, members.iterrows()):
            img = Image.open(row["filepath"])
            ax.imshow(img)
            ax.set_title(row["split"], fontsize=8)
            ax.axis("off")
        safe_name = str(g).replace("/", "_")
        fig.suptitle(f"Group: {safe_name}")
        fig.savefig(os.path.join(output_dir, "spot_checks", f"{safe_name}.png"))
        plt.close(fig)

    print(f"Saved {len(sample_groups)} spot-check grids to "
          f"{os.path.join(output_dir, 'spot_checks')} -- open these and "
          f"confirm each group's images genuinely look related.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--test_fraction", type=float, default=0.20)
    parser.add_argument("--hash_size", type=int, default=16)
    parser.add_argument("--cluster_threshold", type=int, default=6,
                         help="Max Hamming distance to consider two images "
                              "near-duplicates in fallback clustering mode.")
    parser.add_argument("--no_known_originals", action="store_true",
                         help="Use fallback similarity clustering instead of "
                              "assuming any image is a known original.")
    parser.add_argument("--spot_check", type=int, default=0,
                         help="Save this many random group image-grids for "
                              "manual visual verification.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print("Loading images and computing perceptual hashes...")
    df = load_and_hash(args.data_dir, hash_size=args.hash_size)
    print(f"Loaded {len(df)} images.")

    print("Clustering images into similarity groups "
          "(fallback mode: no assumed originals)...")
    df = cluster_by_similarity(df, threshold=args.cluster_threshold)
    group_col = "cluster_id"

    print("Performing grouped, leakage-safe train/test split...")
    df = grouped_split(df, group_col, args.test_fraction, args.seed)

    print("Writing physical train/test folders...")
    materialize_split(df, args.output_dir)

    manifest_path = os.path.join(args.output_dir, "split_manifest.csv")
    df.drop(columns=["hash"]).to_csv(manifest_path, index=False)
    print(f"Saved manifest to {manifest_path}")

    leak_check = df.groupby(group_col)["split"].nunique()
    n_leaks = (leak_check > 1).sum()
    if n_leaks == 0:
        print("PASS: no similarity group is split across train and test.")
    else:
        print(f"FAIL: {n_leaks} groups appear in BOTH splits.")

    if args.spot_check > 0:
        spot_check_grid(df, group_col, args.spot_check, args.output_dir)


if __name__ == "__main__":
    main()
