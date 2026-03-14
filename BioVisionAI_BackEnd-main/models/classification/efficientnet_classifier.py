"""
Skin lesion classifier using EfficientNet-B0 (timm).
Output: logits for num_classes.
"""

import torch
import torch.nn as nn
import timm


class SkinLesionClassifier(nn.Module):
    """EfficientNet-B0 backbone + classifier head for skin lesion classification."""

    def __init__(
        self,
        num_classes: int = 7,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        feat_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, num_classes),
        )
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
