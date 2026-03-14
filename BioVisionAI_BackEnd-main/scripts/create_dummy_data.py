#!/usr/bin/env python3
"""
Create a small dummy dataset (random images + metadata) for testing the pipeline
when HAM10000/ISIC are not available.
Usage: python scripts/create_dummy_data.py --data_root ./data --num_samples 200
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()

    root = Path(args.data_root)
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = [["image_id", "dx"]]
    for i in range(args.num_samples):
        img_id = f"dummy_{i:04d}"
        label = CLASS_NAMES[i % len(CLASS_NAMES)]
        path = images_dir / f"{img_id}.jpg"
        # Random RGB image (simulate skin-like colors)
        arr = np.clip(
            np.random.randn(args.image_size, args.image_size, 3) * 30 + 180,
            0, 255
        ).astype(np.uint8)
        Image.fromarray(arr).save(path)
        rows.append([img_id + ".jpg", label])

    meta_path = root / "metadata.csv"
    with open(meta_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"Created {args.num_samples} dummy images in {images_dir}")
    print(f"Metadata: {meta_path}")
    print("You can now run: python scripts/train.py --config configs/default.yaml --data_root", args.data_root)


if __name__ == "__main__":
    main()
