"""Model factory for supervised and self-supervised learning."""

from __future__ import annotations

from typing import Any

from microVis.log_utils import get_logger

_log = get_logger("microVis.model_arch")

# Supported backbone names → timm model IDs
BACKBONES = {
    "ResNet-18": "resnet18",
    "ResNet-34": "resnet34",
    "EfficientNet-B0": "efficientnet_b0",
    "ConvNeXt-Tiny": "convnext_tiny",
    "ViT-Small": "vit_small_patch16_224",
}

BACKBONE_NAMES = list(BACKBONES.keys())

SSL_METHODS = ["SimCLR", "Barlow Twins", "BYOL"]


def create_sl_model(
    backbone: str,
    num_classes: int,
    in_channels: int = 1,
    pretrained: bool = True,
) -> Any:
    """Create a supervised classification model using timm.

    Args:
        backbone: Backbone name from BACKBONES dict.
        num_classes: Number of output classes.
        in_channels: Number of input channels.
        pretrained: Whether to load ImageNet pretrained weights.

    Returns:
        A timm model ready for training.
    """
    import timm

    model_id = BACKBONES.get(backbone, "resnet18")
    model = timm.create_model(
        model_id,
        pretrained=pretrained,
        num_classes=num_classes,
        in_chans=in_channels,
    )
    _log.info("Created SL model: %s (classes=%d, in_ch=%d, pretrained=%s)",
              model_id, num_classes, in_channels, pretrained)
    return model


def create_ssl_model(
    backbone: str,
    in_channels: int = 1,
    pretrained: bool = True,
    method: str = "SimCLR",
    proj_dim: int = 128,
) -> tuple[Any, Any]:
    """Create a self-supervised learning model.

    Args:
        backbone: Backbone name from BACKBONES dict.
        in_channels: Number of input channels.
        pretrained: Whether to load ImageNet pretrained weights.
        method: SSL method (SimCLR, Barlow Twins, BYOL).
        proj_dim: Projection head output dimension.

    Returns:
        (backbone_model, projection_head) tuple.
    """
    import timm
    import torch.nn as nn

    model_id = BACKBONES.get(backbone, "resnet18")

    # Create backbone without classification head
    backbone_model = timm.create_model(
        model_id,
        pretrained=pretrained,
        num_classes=0,  # removes classifier, returns features
        in_chans=in_channels,
    )

    # Get feature dimension
    feat_dim = backbone_model.num_features

    if method == "SimCLR":
        projection_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, proj_dim),
        )
    elif method == "Barlow Twins":
        projection_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, proj_dim),
        )
    elif method == "BYOL":
        projection_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, proj_dim),
        )
    else:
        raise ValueError(f"Unknown SSL method: {method}")

    _log.info("Created SSL model: %s (method=%s, feat_dim=%d, proj_dim=%d)",
              model_id, method, feat_dim, proj_dim)
    return backbone_model, projection_head


def create_embedding_model(
    backbone: str,
    in_channels: int = 1,
    pretrained: bool = True,
) -> Any:
    """Create a feature embedding model (backbone without classifier).

    Args:
        backbone: Backbone name from BACKBONES dict.
        in_channels: Number of input channels.
        pretrained: Whether to load ImageNet pretrained weights.

    Returns:
        A timm model that outputs feature vectors.
    """
    import timm

    model_id = BACKBONES.get(backbone, "resnet18")
    model = timm.create_model(
        model_id,
        pretrained=pretrained,
        num_classes=0,
        in_chans=in_channels,
    )
    _log.info("Created embedding model: %s (in_ch=%d)", model_id, in_channels)
    return model


def get_feature_dim(backbone: str) -> int:
    """Get the feature dimension of a backbone."""
    import timm

    model_id = BACKBONES.get(backbone, "resnet18")
    model = timm.create_model(model_id, pretrained=False, num_classes=0)
    return model.num_features
