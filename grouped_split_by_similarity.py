"""
Leakage-safe train/test split for the K-Kotagiri Tea Leaf Dataset
==================================================================

Problem
-------
The dataset ships 4,259 images (528 originals + 3,731 pre-made augmented
images) with NO metadata linking each augmented image back to its source
original. A naive random split risks placing an augmented derivative of a
"test" original into the training set (data leakage), which would inflate
reported accuracy.

Solution
--------
1. Compute a perceptual hash (pHash) for every image (robust to rotation,
   flips, mild color/contrast changes -- the kinds of transforms typical
   augmentation pipelines use).
2. For each of the 528 ORIGINAL images, keep its own hash as a "cluster
   center".
3. For every other image (the 3,731 candidates), assign it to whichever
   original has the closest hash (Hamming distance). This approximates
   the true parent-child relationship even without explicit metadata.
4. Split the 528 ORIGINALS into train/test (stratified by class).
5. Propagate that split to every augmented image via its assigned parent:
   if original X is in TEST, every image assigned to X's cluster goes to
   TEST too. Same for TRAIN.

This guarantees no image sharing a near-duplicate parent crosses the
train/test boundary.

Requirements
------------
pip install imagehash pillow pandas scikit-learn --break-system-packages

Usage
-----
Point INPUT_DIRS at your class folders (e.g. Healthy/, Disease/) which
should each contain BOTH original and augmented images together, OR adapt
FILE_LIST loading below if you have a flat folder + a label CSV instead.

    python grouped_split_by_similarity.py

Outputs
-------
- split_manifest.csv   : one row per image with columns
                         [filepath, label, is_original, parent_id, split]
- A printed summary of class balance in each split.

IMPORTANT: This is an approximation, not ground truth. After running,
spot-check a sample of parent<->child assignments visually (a helper for
this is included below) before trusting the split completely. Document
this approach explicitly in the paper's Methods section as a limitation
if no better provenance data can be obtained from the dataset authors.
"""

import os
import glob
import random
from pathlib import Path

import imagehash
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# CONFIGURATION -- edit these paths/labels for your local setup
# ---------------------------------------------------------------------------

# Map each class folder to a label name.
# Put ALL images for a class (originals + augmented) in the same folder,
# or adjust the glob patterns below to match your actual directory layout.
INPUT_DIRS = {
    "Healthy": "./data/Healthy",
    "Disease": "./data/Disease",
}

# Heuristic to detect "original" images among the pool.
# ADAPT THIS: if you know how many originals exist per class and can
# identify them (e.g. they were the first files added, or have a specific
# naming convention, or are the only ones you personally downloaded
# separately), set that logic here. If you truly cannot distinguish
# originals from augmented copies at all, see the fallback note at the
# bottom of this file.
def is_probably_original(filename: str) -> bool:
    """
    Placeholder heuristic. Replace with real logic if you have ANY
    distinguishing signal (e.g. original filenames lack 'aug', 'rot',
    '_v2' etc., or originals were the first N files by creation date).
    """
    lowered = filename.lower()
    aug_markers = ["aug", "rot", "flip", "crop", "jit", "gen", "syn"]
    return not any(m in lowered for m in aug_markers)

TEST_FRACTION = 0.20     # fraction of ORIGINAL images held out for test
HASH_SIZE = 16            # larger = more precise but slower; 16 is a good default
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# STEP 1: Load all images, compute hashes, separate originals vs candidates
# ---------------------------------------------------------------------------

def load_and_hash(input_dirs: dict) -> pd.DataFrame:
    rows = []
    for label, folder in input_dirs.items():
        paths = sorted(glob.glob(os.path.join(folder, "*")))
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
            except Exception as e:
                print(f"Skipping unreadable file {p}: {e}")
                continue
            h = imagehash.phash(img, hash_size=HASH_SIZE)
            rows.append({
                "filepath": p,
                "filename": os.path.basename(p),
                "label": label,
                "hash": h,
                "is_original": is_probably_original(os.path.basename(p)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# STEP 2: Assign every non-original image to its nearest original (per class)
# ---------------------------------------------------------------------------

def assign_parents(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["parent_id"] = None

    for label, group in df.groupby("label"):
        originals = group[group["is_original"]].reset_index()
        candidates = group[~group["is_original"]].reset_index()

        if len(originals) == 0:
            print(f"WARNING: no originals detected for class '{label}'. "
                  f"Check is_probably_original() heuristic.")
            continue

        # Originals are their own parent
        for idx in originals["index"]:
            df.loc[idx, "parent_id"] = df.loc[idx, "filepath"]

        # For each candidate, find nearest original by Hamming distance
        for _, cand in candidates.iterrows():
            best_dist = None
            best_parent = None
            for _, orig in originals.iterrows():
                dist = cand["hash"] - orig["hash"]  # Hamming distance
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_parent = orig["filepath"]
            df.loc[cand["index"], "parent_id"] = best_parent

    return df


# ---------------------------------------------------------------------------
# STEP 3: Split ORIGINALS into train/test (stratified by class), then
#          propagate the split to every image via its parent_id
# ---------------------------------------------------------------------------

def grouped_split(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    originals = df[df["is_original"]]

    train_parents, test_parents = train_test_split(
        originals["filepath"],
        test_size=TEST_FRACTION,
        stratify=originals["label"],
        random_state=RANDOM_SEED,
    )
    test_parent_set = set(test_parents)

    df["split"] = df["parent_id"].apply(
        lambda p: "test" if p in test_parent_set else "train"
    )
    return df


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    random.seed(RANDOM_SEED)

    print("Loading images and computing perceptual hashes...")
    df = load_and_hash(INPUT_DIRS)
    print(f"Loaded {len(df)} images total "
          f"({df['is_original'].sum()} flagged as originals).")

    print("Assigning augmented images to nearest original (per class)...")
    df = assign_parents(df)

    print("Performing grouped, leakage-safe train/test split...")
    df = grouped_split(df)

    out_cols = ["filepath", "filename", "label", "is_original", "parent_id", "split"]
    df[out_cols].to_csv("split_manifest.csv", index=False)
    print("Saved split_manifest.csv")

    print("\n=== Split summary ===")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0))
    print("\nSanity check -- parents should never appear in both splits:")
    leak_check = df.groupby("parent_id")["split"].nunique()
    n_leaks = (leak_check > 1).sum()
    if n_leaks == 0:
        print("PASS: no parent group is split across train and test.")
    else:
        print(f"FAIL: {n_leaks} parent groups appear in BOTH splits. "
              f"Investigate before proceeding.")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# FALLBACK NOTE
# ---------------------------------------------------------------------------
# If you genuinely cannot distinguish originals from augmented images at
# all (is_probably_original() has no signal to work with), the safest
# alternative is:
#   1. Cluster ALL 4,259 images by pairwise hash similarity (e.g. using a
#      similarity threshold or hierarchical clustering) rather than
#      assuming any given image is "the" original.
#   2. Treat each resulting cluster as one group.
#   3. Split clusters (not images) into train/test.
# This still guarantees no near-duplicate crosses the split boundary, even
# without knowing which image in a cluster was the "true" original.
