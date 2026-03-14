#!/usr/bin/env python3
"""
Single-image prediction with optional Grad-CAM.
Usage:
  python scripts/predict.py --checkpoint checkpoints/classification/best.pt --image path/to/image.jpg
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch

from utils.config import load_config
from data.augmentation import get_val_augmentation
from models.classification.efficientnet_classifier import SkinLesionClassifier
from explainability.gradcam import run_grad_cam, save_heatmap_and_overlay


CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
RISK_HIGH = ["mel", "bcc"]  # consider melanoma and BCC as higher risk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output_dir", type=str, default="predict_outputs")
    parser.add_argument("--no_gradcam", action="store_true", help="Skip Grad-CAM")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf_cfg = config["classification"]

    model = SkinLesionClassifier(
        num_classes=clf_cfg["num_classes"],
        backbone=clf_cfg.get("backbone", "efficientnet_b0"),
        pretrained=False,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = get_val_augmentation(image_size=(224, 224))
    transformed = transform(image=image_rgb)
    x = transformed["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])

    risk = "high" if pred_class in RISK_HIGH else ("intermediate" if pred_class in ["akiec", "bkl"] else "low")

    print(f"Predicted class: {pred_class}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Risk category: {risk}")
    print("Probabilities:", dict(zip(CLASS_NAMES, [f"{p:.4f}" for p in probs])))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_gradcam:
        heatmap = run_grad_cam(model, x, target_class=pred_idx, use_cuda=(device.type == "cuda"))
        # Resize image_rgb to 224 if needed for overlay
        img_224 = cv2.resize(image_rgb, (224, 224))
        h_path, o_path = save_heatmap_and_overlay(img_224, heatmap, out_dir, prefix=Path(args.image).stem)
        print(f"Heatmap: {h_path}, Overlay: {o_path}")


if __name__ == "__main__":
    main()
