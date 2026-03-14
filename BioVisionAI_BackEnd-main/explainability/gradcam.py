"""
Grad-CAM for skin lesion classifier.
Produces heatmap and overlay for explainability.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import cv2

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRAD_CAM_AVAILABLE = True
except ImportError:
    GRAD_CAM_AVAILABLE = False


def _get_target_layer(model: torch.nn.Module):
    """Find last conv layer in EfficientNet for Grad-CAM."""
    for m in reversed(list(model.modules())):
        if isinstance(m, torch.nn.Conv2d):
            return m
    return None


def run_grad_cam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    use_cuda: bool = True,
) -> np.ndarray:
    """
    Run Grad-CAM on model for given input.
    Returns heatmap (H, W) in [0, 1].
    """
    if not GRAD_CAM_AVAILABLE:
        return np.zeros((224, 224), dtype=np.float32)  # dummy

    model.eval()
    target_layer = _get_target_layer(model)
    if target_layer is None:
        return np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)

    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = None
    if target_class is not None:
        targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    return grayscale_cam[0]


def get_gradcam_heatmap_overlay(
    rgb_image: np.ndarray,
    heatmap: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlay heatmap on image. rgb_image: (H,W,3) 0-255. Returns (H,W,3) RGB."""
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    if rgb_image.shape[:2] != heatmap.shape[:2]:
        heatmap_colored = cv2.resize(heatmap_colored, (rgb_image.shape[1], rgb_image.shape[0]))
        heatmap = cv2.resize(heatmap, (rgb_image.shape[1], rgb_image.shape[0]))
    overlay = show_cam_on_image(rgb_image.astype(np.float32) / 255.0, heatmap, use_rgb=True)
    return (overlay * 255).astype(np.uint8)


def save_heatmap_and_overlay(
    rgb_image: np.ndarray,
    heatmap: np.ndarray,
    save_dir: str | Path,
    prefix: str = "cam",
) -> Tuple[str, str]:
    """Save heatmap and overlay images; return paths."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    overlay = get_gradcam_heatmap_overlay(rgb_image, heatmap)
    heat_path = save_dir / f"{prefix}_heatmap.png"
    overlay_path = save_dir / f"{prefix}_overlay.png"
    cv2.imwrite(str(heat_path), cv2.cvtColor((heatmap * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return str(heat_path), str(overlay_path)
