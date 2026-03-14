#!/usr/bin/env python3
"""
Dataset download and preparation instructions for HAM10000 / ISIC.
Run this to create a minimal metadata.csv and folder structure.
For full HAM10000 you must download from the official source.
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Prepare data directory for BIOVISION-AI")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--create_dummy", action="store_true", help="Create dummy metadata and placeholder for testing")
    args = parser.parse_args()

    root = Path(args.data_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "images").mkdir(exist_ok=True)

    if args.create_dummy:
        # Create a minimal metadata.csv so that get_dataloaders can be tested with real images later
        meta = root / "metadata.csv"
        if not meta.exists():
            meta.write_text("image_id,dx\n")
            print(f"Created empty {meta}. Add rows: image_id,dx (dx one of: akiec,bcc,bkl,df,mel,nv,vasc)")
        return

    readme = root / "README_DATASET.md"
    readme.write_text("""
# Dataset setup for BIOVISION-AI

## HAM10000

1. Request access at https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
2. Download HAM10000_images.zip and HAM10000_metadata.csv
3. Unzip images into: data/ham10000_images/ (or data/images/)
4. Place HAM10000_metadata.csv in data/
5. Ensure CSV has columns: image_id, dx (diagnosis: akiec, bcc, bkl, df, mel, nv, vasc)

## ISIC

1. Visit https://www.isic-archive.com/
2. Download ISIC 2019 or 2020 dataset (metadata + images)
3. Place metadata CSV in data/ (e.g. metadata.csv) with columns: image_id (or filename), dx
4. Place images in data/images/ with filenames matching image_id in CSV

## Folder structure (expected)

    data/
      metadata.csv          # or HAM10000_metadata.csv
      images/               # *.jpg / *.png
      [optional] masks/     # segmentation masks (same base name as image)

## metadata.csv format

    image_id,dx
    ISIC_001,nv
    ISIC_002,mel
    ...
""")
    print(f"Created {readme}. Read it for download instructions.")
    print("Then create data/metadata.csv with columns: image_id, dx (dx in: akiec, bcc, bkl, df, mel, nv, vasc)")


if __name__ == "__main__":
    main()
