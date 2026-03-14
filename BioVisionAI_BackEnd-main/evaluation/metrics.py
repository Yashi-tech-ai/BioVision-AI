"""Classification and segmentation metrics."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from typing import List, Dict, Any, Optional


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Accuracy, precision, recall, F1, ROC-AUC, confusion matrix."""
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    metrics = {
        "accuracy": float(acc),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "confusion_matrix": cm.tolist(),
    }
    if y_prob is not None and y_prob.shape[1] > 1:
        try:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
            metrics["roc_auc_weighted"] = float(auc)
        except Exception:
            metrics["roc_auc_weighted"] = 0.0
    if class_names:
        metrics["classification_report"] = classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True
        )
    return metrics


def classification_report_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> Dict[str, Any]:
    """Sklearn classification_report as dict."""
    return classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True
    )


def compute_segmentation_metrics(
    pred_masks: np.ndarray,
    true_masks: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Dice score and IoU for binary segmentation."""
    pred_bin = (pred_masks > threshold).astype(np.float32).flatten()
    true_bin = true_masks.astype(np.float32).flatten()
    intersection = (pred_bin * true_bin).sum()
    dice = (2.0 * intersection + 1e-6) / (pred_bin.sum() + true_bin.sum() + 1e-6)
    union = pred_bin.sum() + true_bin.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return {"dice": float(dice), "iou": float(iou)}
