from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

import timm


class UnimodalClassifier(nn.Module):
    """Unimodal classifier using a timm backbone.

    The model accepts inputs of shape (B, N, C, H, W) and an image mask of shape
    (B, N). It produces logits of shape (B, num_classes).
    """

    def __init__(self, backbone: nn.Module, embed_dim: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = int(embed_dim)
        self.num_classes = int(num_classes)
        self.dropout: Optional[nn.Dropout] = nn.Dropout(dropout) if dropout and dropout > 0.0 else None
        self.classifier = nn.Linear(self.embed_dim, self.num_classes)

    def forward(self, images: torch.Tensor, image_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass returning only logits.

        images: (B, N, C, H, W)
        image_mask: (B, N) with 1 for valid images
        """
        B, N, C, H, W = images.shape

        # Collapse to (B*N, C, H, W)
        imgs = images.view(B * N, C, H, W)

        feats = self.backbone(imgs)
        # If backbone returns feature maps, global average pool
        if feats.dim() == 4:
            feats = feats.mean(dim=[2, 3])

        D = feats.shape[1]
        if D != self.embed_dim:
            # adapt embed_dim and classifier if needed
            self.embed_dim = D
            if self.classifier.in_features != D:
                self.classifier = nn.Linear(D, self.num_classes).to(feats.device)

        feats = feats.view(B, N, -1)  # (B, N, D)

        mask = image_mask.to(dtype=feats.dtype).unsqueeze(-1)  # (B, N, 1)
        masked = feats * mask
        sums = masked.sum(dim=1)  # (B, D)
        counts = mask.sum(dim=1).clamp(min=1.0)
        pooled = sums / counts

        if self.dropout is not None:
            pooled = self.dropout(pooled)

        logits = self.classifier(pooled)
        return logits


def build_unimodal(backbone_name: str, num_classes: int, pretrained: bool = True, dropout: float = 0.0) -> UnimodalClassifier:
    """Build a unimodal classifier with a timm backbone.

    The backbone is created with `num_classes=0` and `global_pool='avg'` so it
    returns a vector per image when called.
    """
    backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg")
    embed_dim = getattr(backbone, "num_features", None)
    if embed_dim is None:
        # try to infer by a dummy forward
        backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = backbone(dummy)
        if out.dim() == 2:
            embed_dim = out.shape[1]
        elif out.dim() == 4:
            embed_dim = out.shape[1]
        else:
            raise RuntimeError("Could not infer backbone embedding dimension")

    model = UnimodalClassifier(backbone=backbone, embed_dim=embed_dim, num_classes=num_classes, dropout=dropout)
    return model


__all__ = ["UnimodalClassifier", "build_unimodal"]
