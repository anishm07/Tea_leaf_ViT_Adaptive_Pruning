"""
Stage 1e: Finalize dataset - deduplicate and consolidate
===========================================================

Combines Original Data + Augmented Data for each split (Train/Valid/Test)
into clean, deduplicated folders ready for training. The dataset's own
official split has already been verified leakage-safe (exact + perceptual
hash checks, Stage 1d), so this step just cleans it up for direct use.

Usage
-----
    python 05_finalize_dataset.py \
        --base_dir "/workspace/_extracted/K-Kotagiri Tea Leaf Dataset/New folder/Kotagiri Tea Leaf Dataset" \
        --output_dir /workspace/data_final
"""

import argparse
import hashlib
import os
import shutil
from pathlib import Path


def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_dedup(folders: list) -> list:
    """Given a list of source folders, return a deduplicated list of
    (filepath, unique_name) pairs -- exact duplicate files (by content
    hash) are kept only once."""
    seen_hashes = set()
    kept = []
    for folder in folders:
        if not os.path.isdir(folder):
            print(f"  (folder not found, skipping: {folder})")
            continue
        for f in sorted(Path(folder).glob("*")):
            if f.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            h = file_md5(str(f))
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            kept.append(str(f))
    return kept


def build_split(base: str, output_dir: str, split_name: str,
                 orig_subfolder: str, aug_subfolder: str):
    for label in ["Healthy", "Disease"]:
        orig_folder = os.path.join(base, "Original Data", orig_subfolder, label)
        aug_folder = os.path.join(base, "Augmented Data", aug_subfolder, label)

        print(f"\n[{split_name}/{label}] Collecting and deduplicating...")
        files = collect_dedup([orig_folder, aug_folder])

        dst_dir = os.path.join(output_dir, split_name, label)
        os.makedirs(dst_dir, exist_ok=True)

        for i, src in enumerate(files):
            ext = Path(src).suffix
            dst = os.path.join(dst_dir, f"{split_name}_{label}_{i:05d}{ext}")
            shutil.copy2(src, dst)

        print(f"  -> {len(files)} unique images written to {dst_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    # Note: folder casing is inconsistent in the source archive
    # ("valid" under Original Data, "Valid" under Augmented Data) --
    # handled explicitly below.
    build_split(args.base_dir, args.output_dir, "train", "Train", "Train")
    build_split(args.base_dir, args.output_dir, "valid", "valid", "Valid")
    build_split(args.base_dir, args.output_dir, "test", "Test", "Test")

    print("\n=== Final dataset summary ===")
    total = 0
    for split in ["train", "valid", "test"]:
        for label in ["Healthy", "Disease"]:
            folder = os.path.join(args.output_dir, split, label)
            n = len(list(Path(folder).glob("*")))
            total += n
            print(f"{split}/{label}: {n}")
    print(f"\nTotal unique images across all splits: {total}")


if __name__ == "__main__":
    main()
