#!/usr/bin/env python3
"""
Train U-Net for lesion segmentation (optional; requires masks in data/masks/).
Usage: python scripts/train_segmentation.py --config configs/default.yaml --data_root ./data
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from utils.config import load_config
from utils.logger import setup_logger
from data.datasets.skin_lesion import get_dataloaders
from models.segmentation.unet import UNet
from training.losses import DiceBCELoss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data_root", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_root:
        config["data"]["root"] = args.data_root

    seg_cfg = config.get("segmentation", {})
    if not seg_cfg.get("enabled", True):
        print("Segmentation disabled in config. Enable segmentation.enabled and provide masks.")
        return

    logger = setup_logger("seg_train")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = config["data"]

    try:
        train_loader, val_loader, _, _, _, _ = get_dataloaders(
            data_root=data_cfg["root"],
            batch_size=data_cfg["batch_size"],
            num_workers=data_cfg.get("num_workers", 4),
            train_ratio=data_cfg["train_ratio"],
            val_ratio=data_cfg["val_ratio"],
            test_ratio=data_cfg["test_ratio"],
            image_size=(data_cfg["image_size"], data_cfg["image_size"]),
            mask_dir="masks",
            seed=config.get("seed", 42),
        )
    except Exception as e:
        logger.warning("Dataloaders with masks failed (no masks?): %s. Skipping segmentation training.", e)
        return

    # Filter batches that have at least one mask (optional: require all)
    model = UNet(
        in_channels=seg_cfg.get("in_channels", 3),
        out_channels=seg_cfg.get("out_channels", 1),
    ).to(device)
    criterion = DiceBCELoss(dice_weight=0.5, bce_weight=0.5)
    optimizer = Adam(model.parameters(), lr=seg_cfg.get("lr", 1e-4))
    ckpt_dir = Path(seg_cfg.get("checkpoint_dir", "checkpoints/segmentation"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(seg_cfg.get("epochs", 50)):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            x = batch["image"].to(device)
            mask = batch.get("mask")
            if mask is None:
                continue
            mask = mask.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        if n_batches == 0:
            logger.info("No batches with masks; skipping segmentation training.")
            break
        avg = total_loss / n_batches
        logger.info("Epoch %d seg_loss=%.4f", epoch + 1, avg)
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, ckpt_dir / "latest.pt")

    logger.info("Segmentation training done.")


if __name__ == "__main__":
    main()
