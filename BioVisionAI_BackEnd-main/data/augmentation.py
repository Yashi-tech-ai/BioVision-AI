"""
Data augmentation using Albumentations.
- Random rotation, horizontal/vertical flip
- Brightness, contrast
- Random resize crop
- Normalize
"""

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, List


def get_train_augmentation(
    image_size: Tuple[int, int] = (224, 224),
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
    horizontal_flip: float = 0.5,
    vertical_flip: float = 0.5,
    rotate_limit: int = 30,
    brightness_limit: float = 0.2,
    contrast_limit: float = 0.2,
    random_crop_scale: Tuple[float, float] = (0.8, 1.0),
) -> A.Compose:
    """Training augmentations for skin lesion images."""
    return A.Compose([
        A.HorizontalFlip(p=horizontal_flip),
        A.VerticalFlip(p=vertical_flip),
        A.Rotate(limit=rotate_limit, border_mode=cv2.BORDER_REFLECT),
        A.RandomBrightnessContrast(
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            p=0.5,
        ),
        A.RandomResizedCrop(
            height=image_size[0],
            width=image_size[1],
            scale=random_crop_scale,
            ratio=(0.9, 1.1),
            p=1.0,
        ),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_val_augmentation(
    image_size: Tuple[int, int] = (224, 224),
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
) -> A.Compose:
    """Validation/test: resize + normalize only."""
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


