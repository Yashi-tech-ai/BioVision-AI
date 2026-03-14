"""
Image preprocessing for skin lesion images.
- Resize to 224x224
- Normalize (ImageNet stats)
- Optional hair removal (DullRazor-style)
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def resize_and_normalize(
    image: np.ndarray,
    size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Resize image and normalize to [0,1] with ImageNet stats."""
    image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0
    image = (image - np.array(mean)) / np.array(std)
    return image.astype(np.float32)


def optional_hair_removal(image: np.ndarray) -> np.ndarray:
    """
    Simple morphological hair reduction (DullRazor-style).
    Uses closing + inpainting to reduce dark hair artifacts.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Threshold to get dark structures (hair)
    _, mask = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel)
    result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    return result


def preprocess_image(
    image: np.ndarray,
    size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    use_hair_removal: bool = False,
) -> np.ndarray:
    """
    Full preprocessing: optional hair removal, resize, normalize.
    Input: BGR or RGB uint8 (H, W, 3). Output: float32 (H, W, 3).
    """
    if image.shape[-1] == 3 and image.dtype == np.uint8:
        if use_hair_removal:
            image = optional_hair_removal(image)
        return resize_and_normalize(image, size, mean, std)
    return resize_and_normalize(image, size, mean, std)


def get_preprocess_transforms(
    size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
):
    """Return a callable that preprocesses a numpy image (for non-albumentations use)."""
    def transform(image: np.ndarray) -> np.ndarray:
        return preprocess_image(image, size=size, mean=mean, std=std, use_hair_removal=False)
    return transform
