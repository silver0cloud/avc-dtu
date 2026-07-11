"""Method 2 — Cross-Modal Distillation.

Teacher: a linear probe trained on frozen VGG16 visual embeddings that
predicts the class label (frozen after pretraining).

Student: receives the full audio+visual concatenation and learns to match
both the teacher's soft class-logits and its embedding, via a
temperature-scaled KD loss.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TeacherProbe(nn.Module):
    """Linear probe: frozen VGG16 visual embedding -> class logits."""

    def __init__(self, visual_dim: int = 512, num_classes: int = 100):
        super().__init__()
        self.probe = nn.Linear(visual_dim, num_classes)

    def forward(self, visual: torch.Tensor) -> torch.Tensor:
        return self.probe(visual)


class StudentMLP(nn.Module):
    """AV concat (audio+visual, 640) -> backbone embedding (512) -> class logits."""

    def __init__(
        self,
        audio_dim: int = 128,
        visual_dim: int = 512,
        latent_dim: int = 512,
        num_classes: int = 100,
        dropout: float = 0.2,
    ):
        super().__init__()
        in_dim = audio_dim + visual_dim
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
        )
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, audio: torch.Tensor, visual: torch.Tensor):
        av = torch.cat([audio, visual], dim=-1)
        emb = self.backbone(av)
        logits = self.classifier(emb)
        return F.normalize(emb, p=2, dim=-1), logits


def distillation_loss(
    student_emb: torch.Tensor,
    student_logits: torch.Tensor,
    teacher_emb: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.7,
    beta: float = 0.3,
) -> torch.Tensor:
    """L = alpha*T^2*KL(soft_s || soft_t) + beta*MSE(s_emb,t_emb) + (1-a-b)*(1-CosSim)."""
    gamma = max(1.0 - alpha - beta, 0.0)

    soft_t = F.softmax(teacher_logits / temperature, dim=-1)
    log_soft_s = F.log_softmax(student_logits / temperature, dim=-1)
    kd = F.kl_div(log_soft_s, soft_t, reduction="batchmean") * (temperature**2)

    mse = F.mse_loss(student_emb, teacher_emb)
    cos = 1.0 - F.cosine_similarity(student_emb, teacher_emb, dim=-1).mean()

    return alpha * kd + beta * mse + gamma * cos
