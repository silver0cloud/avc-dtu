"""Trainer for Baseline 1 — Naive Late Fusion MLP (minimum-viable baseline)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from avavs.config import Config
from avavs.models.baselines import NaiveLateFusionMLP
from avavs.training.common import run_epochs


def _step(model, batch, device):
    audio, visual, _labels = batch
    audio, visual = audio.to(device), visual.to(device)
    pred = model(audio, visual)
    target = F.normalize(visual, p=2, dim=-1)
    loss = F.mse_loss(pred, target)
    cos_sim = F.cosine_similarity(pred, target, dim=-1).mean()
    return {"loss": loss, "cos_sim": cos_sim, "mse": loss}


def train_naive_baseline(cfg: Config, train_ds, val_ds, device: str) -> NaiveLateFusionMLP:
    model = NaiveLateFusionMLP(
        audio_dim=cfg.audio_dim, visual_dim=cfg.visual_dim, latent_dim=cfg.latent_dim
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    run_epochs(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_step_fn=_step,
        eval_step_fn=_step,
        optimizer=optimizer,
        num_epochs=cfg.naive_epochs,
        device=device,
        use_amp=cfg.use_amp,
        grad_clip=cfg.grad_clip,
        early_stop_patience=cfg.early_stop_patience,
        checkpoint_path=f"{cfg.output_dir}/naive_baseline_model.pt",
        history_path=f"{cfg.output_dir}/naive_history.json",
    )
    return model
