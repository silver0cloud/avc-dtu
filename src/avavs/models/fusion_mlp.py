"""Method 1 — Latent Space Regression.

Predicts the visual (VGG16) embedding from audio alone, forcing genuine
cross-modal learning with no information leakage from the visual stream.

    audio (128,) -> Linear(128,512) -> LN -> GELU -> Dropout(0.2)
                  -> Linear(512,512) -> LN -> GELU -> Dropout(0.2)
                  -> Linear(512,512)
                  -> L2 normalise
                  -> predicted visual embedding (512,)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionMLP(nn.Module):
    def __init__(self, audio_dim: int = 128, latent_dim: int = 512, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(audio_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        out = self.net(audio)
        return F.normalize(out, p=2, dim=-1)


def regression_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L = 0.5 * MSE(pred, gt) + 0.5 * (1 - CosSim(pred, gt))."""
    mse = F.mse_loss(pred, target)
    cos = 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
    return 0.5 * mse + 0.5 * cos
