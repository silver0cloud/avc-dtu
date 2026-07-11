"""Baseline 1 (Naive Late Fusion) and Baseline 2 (ImageBind-style zero-shot ceiling).

Naive Late Fusion is the minimum viable product: a single-hidden-layer MLP
with no normalisation or residual connections. Any advanced method should
outperform it.

The ImageBind projection is used purely as a zero-shot upper-bound
benchmark; full ImageBind weights are not retrained here — this module
implements the lightweight linear-projection approximation described in
the project spec so the comparison is reproducible without pulling in the
full foundation-model checkpoint.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NaiveLateFusionMLP(nn.Module):
    def __init__(self, audio_dim: int = 128, visual_dim: int = 512, latent_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(audio_dim + visual_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, audio: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
        av = torch.cat([audio, visual], dim=-1)
        return F.normalize(self.net(av), p=2, dim=-1)


class ImageBindProjection(nn.Module):
    """Zero-shot-style linear projection ceiling (not a trained foundation model).

        audio  (128,) -> Linear(128,512)  ─┐
                                            ├─ concat(1024) -> Linear(1024,512) -> L2-norm
        visual (512,) -> Linear(512,512)  ─┘
    """

    def __init__(self, audio_dim: int = 128, visual_dim: int = 512, latent_dim: int = 512):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, latent_dim)
        self.visual_proj = nn.Linear(visual_dim, latent_dim)
        self.fuse = nn.Linear(latent_dim * 2, latent_dim)

    def forward(self, audio: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
        a = self.audio_proj(audio)
        v = self.visual_proj(visual)
        fused = self.fuse(torch.cat([a, v], dim=-1))
        return F.normalize(fused, p=2, dim=-1)
