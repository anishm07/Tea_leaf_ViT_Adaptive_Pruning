"""
Stage 1b: Dataset acquisition and organization
================================================

Mendeley Data often requires a browser session to download (no stable
direct-download API endpoint), so the reliable approach is:

  1. On your own machine (or the RunPod pod's browser-accessible Jupyter/
     VNC if available), go to:
         https://data.mendeley.com/datasets/68yy284sh9/1
     and download the dataset zip.

  2. Upload the zip to your RunPod pod, e.g.:
         scp K-Kotagiri.zip root@<pod-ip>:/workspace/raw/

  3. Run this script to extract and organize it, and to VERIFY the file
     counts match the dataset's own published description:
         225 original Healthy   + 1,610 augmented Healthy  = 1,835
         303 original Disease   + 2,121 augmented Disease  = 2,424
         Total = 4,259 images

     If your extracted counts don't match this, STOP and re-check the
     download/extraction before proceeding to training -- mismatched counts
     mean something is missing or duplicated, which would silently corrupt
     every downstream result.

Usage
-----
    python 02_prepare_dataset.py --zip_path /workspace/raw/K-Kotagiri.zip \
                                  --output_dir /workspace/data
"""

import argparse
import os
import shutil
import zipfile
from pathlib import Path


EXPECTED_COUNTS = {
    "Healthy": {"original": 225, "augmented": 1610, "total": 1835},
    "Disease": {"original": 303, "augmented": 2121, "total": 2424},
}
EXPECTED_TOTAL = 4259


def extract_zip(zip_path: str, extract_to: str):
    print(f"Extracting {zip_path} -> {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)


def find_class_folders(root: str) -> dict:
    """
    Walk the extracted directory tree and try to auto-detect the Healthy
    and Disease folders. Mendeley archives vary in structure, so this
    looks for folder names containing 'healthy' or 'disease' (case
    insensitive) anywhere in the tree.
    """
    found = {}
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            low = d.lower()
            if "healthy" in low and "Healthy" not in found:
                found["Healthy"] = os.path.join(dirpath, d)
            elif "disease" in low and "Disease" not in found:
                found["Disease"] = os.path.join(dirpath, d)
    return found


def count_images(folder: str) -> int:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sum(
        1 for f in Path(folder).rglob("*")
        if f.suffix.lower() in exts
    )


def organize(class_folders: dict, output_dir: str):
    for label, src in class_folders.items():
        dst = os.path.join(output_dir, label)
        os.makedirs(dst, exist_ok=True)
        n_copied = 0
        for f in Path(src).rglob("*"):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                shutil.copy2(f, os.path.join(dst, f.name))
                n_copied += 1
        print(f"Copied {n_copied} images -> {dst}")


def verify_counts(output_dir: str):
    print("\n=== Verifying counts against published dataset description ===")
    total = 0
    all_ok = True
    for label, expected in EXPECTED_COUNTS.items():
        folder = os.path.join(output_dir, label)
        actual = count_images(folder)
        total += actual
        status = "OK" if actual == expected["total"] else "MISMATCH"
        if status == "MISMATCH":
            all_ok = False
        print(f"{label}: expected {expected['total']}, found {actual} -> {status}")

    print(f"\nTotal: expected {EXPECTED_TOTAL}, found {total}")
    if all_ok and total == EXPECTED_TOTAL:
        print("PASS: counts match the dataset's published description.")
    else:
        print("WARNING: counts do NOT match. Do not proceed to training "
              "until this is resolved -- check for missing/duplicated "
              "files, nested subfolders not walked, or a corrupted "
              "extraction.")
    return all_ok and total == EXPECTED_TOTAL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_path", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    extract_dir = os.path.join(os.path.dirname(args.output_dir), "_extracted")
    os.makedirs(extract_dir, exist_ok=True)

    extract_zip(args.zip_path, extract_dir)

    class_folders = find_class_folders(extract_dir)
    if "Healthy" not in class_folders or "Disease" not in class_folders:
        print("ERROR: could not auto-detect Healthy/Disease folders in the "
              f"extracted archive. Found: {class_folders}. "
              "Inspect the extracted structure manually and edit "
              "find_class_folders() or pass explicit paths.")
        return

    print(f"Detected folders: {class_folders}")
    organize(class_folders, args.output_dir)
    verify_counts(args.output_dir)


if __name__ == "__main__":
    main()
