"""
Multi-head navigation model for CS2 imitation learning.
Predicts movement, turning, and auxiliary actions from visual input.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class NavigationModel(nn.Module):
    """
    Multi-head model for CS2 navigation.

    Architecture:
    - CNN backbone (ResNet/EfficientNet) for feature extraction
    - Temporal aggregation (LSTM/Transformer) for sequence history
    - Multiple prediction heads for different action types

    Args:
        backbone: CNN backbone name ('resnet18', 'resnet34', 'efficientnet_b0', etc.)
        n_move_classes: Number of movement classes (default 9)
        n_turn_classes: Number of turning classes (default 9)
        history_len: Number of frames in sequence (1 = single frame)
        use_temporal: Whether to use LSTM for temporal modeling
        pretrained: Use ImageNet pretrained weights
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        n_move_classes: int = 9,
        n_turn_classes: int = 9,
        history_len: int = 1,
        use_temporal: bool = False,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.backbone_name = backbone
        self.n_move_classes = n_move_classes
        self.n_turn_classes = n_turn_classes
        self.history_len = history_len
        self.use_temporal = use_temporal and history_len > 1

        # Build CNN backbone
        self.backbone, feature_dim = self._build_backbone(backbone, pretrained)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Temporal modeling (optional)
        if self.use_temporal:
            self.temporal = nn.LSTM(
                input_size=feature_dim,
                hidden_size=512,
                num_layers=2,
                batch_first=True,
                dropout=0.2,
            )
            feature_dim = 512

        # Shared feature layer
        self.shared_fc = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Action heads
        self.move_head = nn.Linear(512, n_move_classes)
        self.turn_head = nn.Linear(512, n_turn_classes)
        self.jump_head = nn.Linear(512, 1)  # binary
        self.crouch_head = nn.Linear(512, 1)  # binary
        self.fire_head = nn.Linear(512, 1)  # binary

        # Optional: mouse delta regression heads
        self.mouse_dx_head = nn.Linear(512, 1)
        self.mouse_dy_head = nn.Linear(512, 1)

    def _build_backbone(self, name: str, pretrained: bool) -> tuple[nn.Module, int]:
        """Build CNN backbone and return (model, feature_dim)."""
        if name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
            feature_dim = model.fc.in_features
            model.fc = nn.Identity()
        elif name == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            model = models.resnet34(weights=weights)
            feature_dim = model.fc.in_features
            model.fc = nn.Identity()
        elif name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            model = models.resnet50(weights=weights)
            feature_dim = model.fc.in_features
            model.fc = nn.Identity()
        elif name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b0(weights=weights)
            feature_dim = model.classifier[1].in_features
            model.classifier = nn.Identity()
        elif name == "efficientnet_b1":
            weights = models.EfficientNet_B1_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b1(weights=weights)
            feature_dim = model.classifier[1].in_features
            model.classifier = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {name}")

        return model, feature_dim

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, history_len, C, H, W)

        Returns:
            Dict with keys: 'move', 'turn', 'jump', 'crouch', 'fire', 'mouse_dx', 'mouse_dy'
        """
        B, T, C, H, W = x.shape

        # Extract features from each frame
        # Reshape to (B*T, C, H, W) for batch processing
        x = x.view(B * T, C, H, W)
        features = self.backbone(x)  # (B*T, feature_dim)

        # Reshape back to (B, T, feature_dim)
        feature_dim = features.shape[-1]
        features = features.view(B, T, feature_dim)

        # Temporal modeling
        if self.use_temporal:
            features, _ = self.temporal(features)  # (B, T, 512)
            features = features[:, -1, :]  # Take last timestep (B, 512)
        else:
            # Just take the last frame if no temporal model
            features = features[:, -1, :]  # (B, feature_dim)

        # Shared feature layer
        shared = self.shared_fc(features)  # (B, 512)

        # Prediction heads
        move_logits = self.move_head(shared)
        turn_logits = self.turn_head(shared)
        jump_logits = self.jump_head(shared).squeeze(-1)
        crouch_logits = self.crouch_head(shared).squeeze(-1)
        fire_logits = self.fire_head(shared).squeeze(-1)
        mouse_dx = self.mouse_dx_head(shared).squeeze(-1)
        mouse_dy = self.mouse_dy_head(shared).squeeze(-1)

        return {
            "move": move_logits,
            "turn": turn_logits,
            "jump": jump_logits,
            "crouch": crouch_logits,
            "fire": fire_logits,
            "mouse_dx": mouse_dx,
            "mouse_dy": mouse_dy,
        }


def create_model(
    backbone: str = "resnet18",
    n_move_classes: int = 9,
    n_turn_classes: int = 9,
    history_len: int = 1,
    use_temporal: bool = False,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float = 0.3,
) -> NavigationModel:
    """Factory function to create a navigation model."""
    return NavigationModel(
        backbone=backbone,
        n_move_classes=n_move_classes,
        n_turn_classes=n_turn_classes,
        history_len=history_len,
        use_temporal=use_temporal,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
    )
