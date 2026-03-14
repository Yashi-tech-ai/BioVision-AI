#!/usr/bin/env python3
"""
Evaluate skin lesion classifier: metrics, confusion matrix, ROC, PR curves.
Usage:
  python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/classification/best.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from utils.config import load_config
from data.datasets.skin_lesion import get_dataloaders
from models.classification.efficientnet_classifier import SkinLesionClassifier
from evaluation.metrics import compute_classification_metrics, classification_report_dict
from evaluation.plots import plot_confusion_matrix, plot_roc_curves, plot_pr_curves


CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="evaluation_outputs")
    parser.add_argument("--data_root", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_root:
        config["data"]["root"] = args.data_root

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = config["data"]
    clf_cfg = config["classification"]

    _, _, test_loader, _, _, _ = get_dataloaders(
        data_root=data_cfg["root"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg.get("num_workers", 4),
        train_ratio=data_cfg["train_ratio"],
        val_ratio=data_cfg["val_ratio"],
        test_ratio=data_cfg["test_ratio"],
        image_size=(data_cfg["image_size"], data_cfg["image_size"]),
        seed=config.get("seed", 42),
    )

    model = SkinLesionClassifier(
        num_classes=clf_cfg["num_classes"],
        backbone=clf_cfg.get("backbone", "efficientnet_b0"),
        pretrained=False,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["image"].to(device)
            y = batch["label"]
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
            all_probs.extend(probs)

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    metrics = compute_classification_metrics(y_true, y_pred, y_prob, class_names=CLASS_NAMES)
    print("Metrics:", metrics)
    report = classification_report_dict(y_true, y_pred, CLASS_NAMES)
    print("Classification report:", report)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(np.array(metrics["confusion_matrix"]), CLASS_NAMES, save_path=str(out_dir / "confusion_matrix.png"))
    plot_roc_curves(y_true, y_prob, CLASS_NAMES, save_path=str(out_dir / "roc_curves.png"))
    plot_pr_curves(y_true, y_prob, CLASS_NAMES, save_path=str(out_dir / "pr_curves.png"))
    print(f"Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
