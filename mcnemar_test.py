"""
McNemar's test for paired model comparison
=============================================

Loads saved checkpoints, runs inference on the test set to get per-image
predictions (correct/incorrect for each image, for each model), and runs
McNemar's test to determine whether the accuracy difference between two
models is statistically significant -- not just numerically different.

McNemar's test is the right tool here because it's a PAIRED test: both
models were evaluated on the exact same 260 test images, so what matters
is how often they disagree with each other (not just their overall
accuracy), specifically the asymmetry in "model A right / model B wrong"
vs "model A wrong / model B right" cases.

Usage
-----
    python mcnemar_test.py \
        --data_dir /workspace/data_final \
        --model_a_name resnet50 --model_a_type cnn \
        --model_a_checkpoint /workspace/results_v3/resnet50/best_model.pt \
        --model_b_name vit_base_patch16_224 --model_b_type vit \
        --model_b_checkpoint /workspace/results_v3/vit_base_patch16_224/best_model.pt \
        --img_size 224
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from scipy.stats import chi2


def build_eval_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_predictions(model_name, checkpoint_path, test_loader, device, num_classes):
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # Explicitly free the model before returning, to reduce peak memory
    # when this function is called back-to-back for two different models.
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    import gc
    gc.collect()

    return np.array(all_preds), np.array(all_labels)


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray):
    """
    correct_a, correct_b: boolean arrays, True where that model got the
    image right. Returns the contingency table and McNemar's test
    statistic (with continuity correction) and p-value.
    """
    both_correct = np.sum(correct_a & correct_b)
    a_only = np.sum(correct_a & ~correct_b)   # A right, B wrong
    b_only = np.sum(~correct_a & correct_b)   # A wrong, B right
    both_wrong = np.sum(~correct_a & ~correct_b)

    n_discordant = a_only + b_only
    if n_discordant == 0:
        return {"both_correct": both_correct, "a_only": a_only, "b_only": b_only,
                "both_wrong": both_wrong, "statistic": 0.0, "p_value": 1.0,
                "note": "No discordant pairs -- models agree on every prediction."}

    # McNemar's test with continuity correction
    statistic = (abs(a_only - b_only) - 1) ** 2 / n_discordant
    p_value = 1 - chi2.cdf(statistic, df=1)

    return {"both_correct": int(both_correct), "a_only": int(a_only),
            "b_only": int(b_only), "both_wrong": int(both_wrong),
            "statistic": float(statistic), "p_value": float(p_value)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model_a_name", required=True)
    parser.add_argument("--model_a_checkpoint", required=True)
    parser.add_argument("--model_b_name", required=True)
    parser.add_argument("--model_b_checkpoint", required=True)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = datasets.ImageFolder(
        os.path.join(args.data_dir, "test"),
        transform=build_eval_transform(args.img_size))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    class_names = test_ds.classes
    num_classes = len(class_names)

    print(f"Test set: {len(test_ds)} images, classes: {class_names}\n")

    print(f"Running inference: {args.model_a_name}...")
    preds_a, labels = get_predictions(args.model_a_name, args.model_a_checkpoint,
                                       test_loader, device, num_classes)
    print(f"Running inference: {args.model_b_name}...")
    preds_b, labels_b = get_predictions(args.model_b_name, args.model_b_checkpoint,
                                         test_loader, device, num_classes)

    assert np.array_equal(labels, labels_b), "Label order mismatch between runs -- check DataLoader shuffle is False."

    correct_a = (preds_a == labels)
    correct_b = (preds_b == labels)

    acc_a = correct_a.mean()
    acc_b = correct_b.mean()

    result = mcnemar_test(correct_a, correct_b)

    print(f"\n{'='*60}")
    print(f"McNemar's Test: {args.model_a_name} vs {args.model_b_name}")
    print(f"{'='*60}")
    print(f"{args.model_a_name} accuracy: {acc_a:.4f}")
    print(f"{args.model_b_name} accuracy: {acc_b:.4f}")
    print(f"\nContingency table:")
    print(f"  Both correct:              {result['both_correct']}")
    print(f"  {args.model_a_name} only right:  {result['a_only']}")
    print(f"  {args.model_b_name} only right:  {result['b_only']}")
    print(f"  Both wrong:                {result['both_wrong']}")
    print(f"\nMcNemar's statistic: {result['statistic']:.4f}")
    print(f"p-value: {result['p_value']:.4f}")
    if result['p_value'] < 0.05:
        print("=> Statistically significant difference (p < 0.05)")
    else:
        print("=> NOT statistically significant (p >= 0.05) -- the observed "
              "accuracy difference could plausibly be due to chance on this test set.")


if __name__ == "__main__":
    main()
