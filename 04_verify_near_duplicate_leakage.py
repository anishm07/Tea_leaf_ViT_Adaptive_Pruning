"""
Stage 1d: Near-duplicate leakage verification
================================================

Exact MD5 hashing (already run) cannot detect leakage where an augmented
training image was generated FROM a test/valid original -- the augmented
version is visually similar but not byte-identical, so md5 correctly
reports it as "different" even though it's functionally a leaked copy.

This script uses perceptual hashing (pHash) to catch that case: for every
image in Augmented Data/Train, it finds the closest match (by Hamming
distance) among Original Data/Test and Original Data/Valid images. If any
match is suspiciously close (below a threshold), that augmented training
image is likely derived from a held-out original -- true leakage.

Usage
-----
    python 04_verify_near_duplicate_leakage.py --base_dir "/workspace/_extracted/K-Kotagiri Tea Leaf Dataset/New folder/Kotagiri Tea Leaf Dataset"

Interpretation
--------------
- If NO close matches are found (all distances well above threshold),
  the dataset's official split is genuinely leakage-safe and can be used
  directly for training with confidence.
- If close matches ARE found, note how many and inspect the flagged pairs
  (saved to suspicious_pairs.csv) before deciding whether to exclude those
  specific augmented images or fall back to rebuilding the split entirely.
"""

import argparse
import os
from pathlib import Path

import imagehash
from PIL import Image
import pandas as pd


def hash_folder(folder: str, hash_size: int = 16) -> dict:
    hashes = {}
    for f in Path(folder).rglob("*"):
        if f.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        try:
            img = Image.open(f).convert("RGB")
            hashes[str(f)] = imagehash.phash(img, hash_size=hash_size)
        except Exception as e:
            print(f"Skipping unreadable file {f}: {e}")
    return hashes


def find_closest_matches(source_hashes: dict, reference_hashes: dict,
                          threshold: int) -> pd.DataFrame:
    """For each image in source_hashes, find its closest match in
    reference_hashes. Flag pairs at or below the leakage threshold."""
    rows = []
    ref_items = list(reference_hashes.items())

    for src_path, src_hash in source_hashes.items():
        best_dist = None
        best_ref = None
        for ref_path, ref_hash in ref_items:
            dist = src_hash - ref_hash
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_ref = ref_path
        rows.append({
            "source_image": src_path,
            "closest_reference_image": best_ref,
            "hamming_distance": best_dist,
            "flagged_as_leak": best_dist <= threshold,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True,
                         help="Path to 'Kotagiri Tea Leaf Dataset' folder")
    parser.add_argument("--hash_size", type=int, default=16)
    parser.add_argument("--leak_threshold", type=int, default=8,
                         help="Hamming distance at/below which a match is "
                              "flagged as likely leakage. Lower = stricter. "
                              "8 is a reasonable starting point for a 16x16 "
                              "pHash; tune based on results.")
    parser.add_argument("--output_csv", default="/workspace/leak_check_results.csv")
    args = parser.parse_args()

    base = args.base_dir

    checks = [
        ("Augmented Data/Train/Disease", "Original Data/Test/Disease", "Original Data/valid/Disease"),
        ("Augmented Data/Train/Healthy", "Original Data/Test/Healthy", "Original Data/valid/Healthy"),
    ]

    all_results = []

    for aug_train_rel, orig_test_rel, orig_valid_rel in checks:
        print(f"\n=== Checking {aug_train_rel} against held-out originals ===")

        aug_train_path = os.path.join(base, aug_train_rel)
        orig_test_path = os.path.join(base, orig_test_rel)
        orig_valid_path = os.path.join(base, orig_valid_rel)

        print("Hashing augmented training images...")
        aug_hashes = hash_folder(aug_train_path, args.hash_size)
        print(f"  {len(aug_hashes)} images hashed.")

        print("Hashing held-out original Test + Valid images...")
        ref_hashes = hash_folder(orig_test_path, args.hash_size)
        ref_hashes.update(hash_folder(orig_valid_path, args.hash_size))
        print(f"  {len(ref_hashes)} reference images hashed.")

        df = find_closest_matches(aug_hashes, ref_hashes, args.leak_threshold)
        df["class_check"] = aug_train_rel
        all_results.append(df)

        n_flagged = df["flagged_as_leak"].sum()
        print(f"  Minimum distance found: {df['hamming_distance'].min()}")
        print(f"  Median distance: {df['hamming_distance'].median()}")
        print(f"  Images flagged as likely leakage (distance <= {args.leak_threshold}): {n_flagged} / {len(df)}")

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(args.output_csv, index=False)
    print(f"\nFull results saved to {args.output_csv}")

    total_flagged = combined["flagged_as_leak"].sum()
    print(f"\n=== SUMMARY ===")
    print(f"Total augmented training images checked: {len(combined)}")
    print(f"Total flagged as likely leaked from held-out originals: {total_flagged}")
    if total_flagged == 0:
        print("PASS: no near-duplicate leakage detected between Augmented "
              "Train and held-out Original Test/Valid images. The "
              "official dataset split appears safe to use directly.")
    else:
        print("ATTENTION: some augmented training images closely resemble "
              "held-out originals. Inspect suspicious pairs in the CSV "
              "before proceeding -- consider removing flagged images from "
              "training or rebuilding the split.")


if __name__ == "__main__":
    main()
