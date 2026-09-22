"""
Stage 2 of McNemar's test (memory-safe split version): loads two saved
prediction files (from predict_single_model.py) and runs McNemar's test.
This step is extremely lightweight -- no model loading, just numpy arrays.

Usage
-----
    python compare_predictions.py \
        --preds_a /workspace/preds_resnet50.npz --name_a resnet50 \
        --preds_b /workspace/preds_vit_base_patch16_224.npz --name_b vit_base_patch16_224
"""

import argparse
import numpy as np
from scipy.stats import chi2


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray):
    both_correct = np.sum(correct_a & correct_b)
    a_only = np.sum(correct_a & ~correct_b)
    b_only = np.sum(~correct_a & correct_b)
    both_wrong = np.sum(~correct_a & ~correct_b)

    n_discordant = a_only + b_only
    if n_discordant == 0:
        return {"both_correct": int(both_correct), "a_only": int(a_only),
                "b_only": int(b_only), "both_wrong": int(both_wrong),
                "statistic": 0.0, "p_value": 1.0,
                "note": "No discordant pairs -- models agree on every prediction."}

    statistic = (abs(a_only - b_only) - 1) ** 2 / n_discordant
    p_value = 1 - chi2.cdf(statistic, df=1)

    return {"both_correct": int(both_correct), "a_only": int(a_only),
            "b_only": int(b_only), "both_wrong": int(both_wrong),
            "statistic": float(statistic), "p_value": float(p_value)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds_a", required=True)
    parser.add_argument("--name_a", required=True)
    parser.add_argument("--preds_b", required=True)
    parser.add_argument("--name_b", required=True)
    args = parser.parse_args()

    data_a = np.load(args.preds_a)
    data_b = np.load(args.preds_b)

    preds_a, labels_a = data_a["preds"], data_a["labels"]
    preds_b, labels_b = data_b["preds"], data_b["labels"]

    assert np.array_equal(labels_a, labels_b), \
        "Label order mismatch -- both prediction files must come from the same test set order."

    correct_a = (preds_a == labels_a)
    correct_b = (preds_b == labels_b)

    acc_a = correct_a.mean()
    acc_b = correct_b.mean()

    result = mcnemar_test(correct_a, correct_b)

    print(f"{'='*60}")
    print(f"McNemar's Test: {args.name_a} vs {args.name_b}")
    print(f"{'='*60}")
    print(f"{args.name_a} accuracy: {acc_a:.4f}")
    print(f"{args.name_b} accuracy: {acc_b:.4f}")
    print(f"\nContingency table:")
    print(f"  Both correct:            {result['both_correct']}")
    print(f"  {args.name_a} only right:     {result['a_only']}")
    print(f"  {args.name_b} only right:     {result['b_only']}")
    print(f"  Both wrong:              {result['both_wrong']}")
    print(f"\nMcNemar's statistic: {result['statistic']:.4f}")
    print(f"p-value: {result['p_value']:.4f}")
    if result['p_value'] < 0.05:
        print("=> Statistically significant difference (p < 0.05)")
    else:
        print("=> NOT statistically significant (p >= 0.05) -- the observed "
              "accuracy difference could plausibly be due to chance on this test set.")


if __name__ == "__main__":
    main()
