#!/usr/bin/env python3
"""
Train skin lesion classifier (and optionally segmentation).
Usage:
  python scripts/train.py --config configs/default.yaml
  python scripts/train.py --config configs/classification.yaml --data_root ./data
"""

import argparse
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.config import load_config
from utils.logger import setup_logger
from data.datasets.skin_lesion import get_dataloaders
from models.classification.efficientnet_classifier import SkinLesionClassifier
from models.segmentation.unet import UNet
from training.losses import DiceBCELoss, FocalLoss


CLASS_NAMES = [
    "akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"
]


def train_classification(config: dict):
    logger = setup_logger("train")
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    data_cfg = config["data"]
    clf_cfg = config["classification"]

    train_loader, val_loader, test_loader, _, _, _ = get_dataloaders(
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
        pretrained=clf_cfg.get("pretrained", True),
    ).to(device)

    if clf_cfg.get("use_class_weights"):
        from collections import Counter
        labels = []
        for b in train_loader:
            labels.extend(b["label"].tolist())
        counts = Counter(labels)
        total = sum(counts.values())
        weights = torch.tensor([total / (clf_cfg["num_classes"] * counts.get(i, 1)) for i in range(clf_cfg["num_classes"])], dtype=torch.float32).to(device)
    else:
        weights = None

    if clf_cfg.get("loss") == "focal":
        criterion = FocalLoss(gamma=clf_cfg.get("focal_gamma", 2.0), weight=weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = AdamW(model.parameters(), lr=clf_cfg["lr"], weight_decay=clf_cfg.get("weight_decay", 0.01))
    scheduler = CosineAnnealingLR(optimizer, T_max=clf_cfg["epochs"])
    checkpoint_dir = Path(clf_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    patience = clf_cfg.get("early_stopping_patience", 10)
    patience_counter = 0

    for epoch in range(clf_cfg["epochs"]):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        train_loss /= len(train_loader)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(device)
                y = batch["label"].to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        val_acc = correct / total if total else 0
        logger.info(f"Epoch {epoch+1}/{clf_cfg['epochs']} train_loss={train_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "config": config,
            }, checkpoint_dir / "best.pt")
            logger.info(f"Saved best model (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    logger.info("Training complete.")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data_root", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_root:
        config["data"]["root"] = args.data_root

    train_classification(config)


if __name__ == "__main__":
    main()
