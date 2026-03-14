"""
Skin lesion dataset for HAM10000 / ISIC-style data.

Expected structure:
  data/
    images/           # or ham10000_images/ etc.
      *.jpg
    metadata.csv       # columns: image_id, dx (diagnosis), optional: lesion_id, path
    or
    HAM10000_metadata.csv  # image_id, dx, lesion_id, ...

If segmentation masks are available:
  data/
    masks/
      *.png  # same base name as image
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Callable

import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Default HAM10000 dx (diagnosis) to class index
DX_TO_IDX = {
    "akiec": 0,  # actinic keratosis
    "bcc": 1,    # basal cell carcinoma
    "bkl": 2,    # benign keratosis
    "df": 3,     # dermatofibroma
    "mel": 4,    # melanoma
    "nv": 5,     # nevus
    "vasc": 6,   # vascular
}


class SkinLesionDataset(Dataset):
    """
    Dataset for skin lesion classification (and optional segmentation).
    Supports CSV with image_id, dx (diagnosis). Image paths can be in a folder or in CSV.
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[Callable] = None,
        mask_paths: Optional[List[Optional[str]]] = None,
        class_names: Optional[List[str]] = None,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.mask_paths = mask_paths  # same length as image_paths, None if no mask
        self.class_names = class_names or list(DX_TO_IDX.keys())

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]

        out = {"image": image, "label": label, "path": path}
        if self.mask_paths and self.mask_paths[idx]:
            mask_path = self.mask_paths[idx]
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask = (mask > 127).astype(np.float32)
                out["mask"] = mask
            else:
                out["mask"] = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        else:
            out["mask"] = None

        if self.transform:
            if out["mask"] is not None and out["mask"].size > 0:
                transformed = self.transform(image=image, mask=out["mask"])
                out["image"] = transformed["image"]
                out["mask"] = transformed["mask"]
            else:
                transformed = self.transform(image=image)
                out["image"] = transformed["image"]
                if out["mask"] is not None:
                    out["mask"] = None  # drop mask if not transformed

        if not isinstance(out["image"], torch.Tensor):
            out["image"] = torch.from_numpy(np.asarray(out["image"]).transpose(2, 0, 1)).float()
        out["label"] = torch.tensor(out["label"], dtype=torch.long)
        if out.get("mask") is not None:
            m = out["mask"]
            if isinstance(m, np.ndarray):
                m = torch.from_numpy(m).float()
            if m.dim() == 2:
                m = m.unsqueeze(0)
            out["mask"] = m
        return out


def _resolve_image_path(root: Path, row: pd.Series, image_col: str, images_dir: str) -> str:
    """Get full path to image from metadata row."""
    if image_col in row and pd.notna(row[image_col]):
        p = root / row[image_col]
        if p.exists():
            return str(p)
        p2 = root / "images" / row[image_col]
        if p2.exists():
            return str(p2)
        p3 = root / images_dir / row[image_col]
        if p3.exists():
            return str(p3)
    # Try common patterns
    for folder in ["images", "ham10000_images", "ISIC", ""]:
        for ext in [".jpg", ".jpeg", ".png"]:
            cand = root / folder / (str(row.get("image_id", row.name)) + ext)
            if cand.exists():
                return str(cand)
    return ""


def load_metadata_and_paths(
    data_root: str | Path,
    metadata_file: Optional[str] = None,
    image_column: str = "image_id",
    label_column: str = "dx",
    images_dir: str = "images",
) -> Tuple[pd.DataFrame, List[str], List[int]]:
    """
    Load CSV and build list of image paths and labels.
    CSV must have columns for image id and diagnosis (dx).
    """
    root = Path(data_root)
    if metadata_file is None:
        for name in ["metadata.csv", "HAM10000_metadata.csv", "ISIC_metadata.csv", "labels.csv"]:
            if (root / name).exists():
                metadata_file = name
                break
        if metadata_file is None:
            raise FileNotFoundError(f"No metadata CSV found in {root}. Place metadata.csv with columns: {image_column}, {label_column}")

    df = pd.read_csv(root / metadata_file)
    if label_column not in df.columns:
        raise ValueError(f"CSV must have column '{label_column}'. Found: {list(df.columns)}")
    if image_column not in df.columns:
        # Try image_id or first column
        image_column = "image_id" if "image_id" in df.columns else df.columns[0]

    all_paths = []
    all_labels = []
    for _, row in df.iterrows():
        path = _resolve_image_path(root, row, image_column, images_dir)
        if not path or not os.path.exists(path):
            continue
        dx = str(row[label_column]).strip().lower()
        idx = DX_TO_IDX.get(dx)
        if idx is None:
            continue
        all_paths.append(path)
        all_labels.append(idx)

    return df, all_paths, all_labels


def get_dataloaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    image_size: Tuple[int, int] = (224, 224),
    metadata_file: Optional[str] = None,
    mask_dir: Optional[str] = None,
    seed: int = 42,
    image_column: str = "image_id",
    label_column: str = "dx",
):
    """
    Build train/val/test dataloaders for skin lesion classification.
    data_root: path to folder containing metadata.csv and images/ (or ham10000_images).
    """
    from data.augmentation import get_train_augmentation, get_val_augmentation

    _, paths, labels = load_metadata_and_paths(
        Path(data_root),
        metadata_file=metadata_file,
        image_column=image_column,
        label_column=label_column,
    )
    if not paths:
        raise ValueError(
            f"No valid images found in {data_root}. "
            "Ensure metadata.csv has 'image_id' and 'dx' columns and images exist in data/images/."
        )

    # Split
    n = len(paths)
    train_n = int(n * train_ratio)
    val_n = int(n * val_ratio)
    test_n = n - train_n - val_n
    indices = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(indices)
    train_idx, rest = indices[:train_n], indices[train_n:]
    val_n_actual = min(val_n, len(rest))
    val_idx, test_idx = rest[:val_n_actual], rest[val_n_actual:]

    train_paths = [paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_paths = [paths[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    test_paths = [paths[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    def _mask_paths(img_paths: List[str], root: Path, mask_d: Optional[str]) -> List[Optional[str]]:
        if not mask_d:
            return [None] * len(img_paths)
        out = []
        for p in img_paths:
            base = Path(p).stem
            for ext in [".png", ".jpg"]:
                m = root / mask_d / (base + ext)
                if m.exists():
                    out.append(str(m))
                    break
            else:
                out.append(None)
        return out

    root = Path(data_root)
    train_masks = _mask_paths(train_paths, root, mask_dir)
    val_masks = _mask_paths(val_paths, root, mask_dir)
    test_masks = _mask_paths(test_paths, root, mask_dir)

    train_tf = get_train_augmentation(image_size=image_size)
    val_tf = get_val_augmentation(image_size=image_size)

    train_ds = SkinLesionDataset(train_paths, train_labels, transform=train_tf, mask_paths=train_masks)
    val_ds = SkinLesionDataset(val_paths, val_labels, transform=val_tf, mask_paths=val_masks)
    test_ds = SkinLesionDataset(test_paths, test_labels, transform=val_tf, mask_paths=test_masks)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds
