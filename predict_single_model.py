"""
Stage 1 of McNemar's test (memory-safe split version): run inference for
ONE model, save its predictions to a .npz file. Run this twice (once per
model) as SEPARATE process invocations -- this guarantees full memory
release between models, avoiding the OOM seen when both models were
loaded in a single Python process.

Usage
-----
    python predict_single_model.py \
        --data_dir /workspace/data_final \
        --model_name resnet50 \
        --checkpoint /workspace/results_v3/resnet50/best_model.pt \
        --output_file /workspace/preds_resnet50.npz \
        --img_size 224 --batch_size 8
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm


def build_eval_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_ds = datasets.ImageFolder(
        os.path.join(args.data_dir, "test"),
        transform=build_eval_transform(args.img_size))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    class_names = test_ds.classes
    print(f"Test set: {len(test_ds)} images, classes: {class_names}")

    model = timm.create_model(args.model_name, pretrained=False, num_classes=len(class_names))
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    del state_dict
    model = model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"  batch {i+1}/{len(test_loader)} done")

    preds = np.array(all_preds)
    labels = np.array(all_labels)

    np.savez(args.output_file, preds=preds, labels=labels, class_names=class_names)
    acc = (preds == labels).mean()
    print(f"\nSaved predictions to {args.output_file}")
    print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
