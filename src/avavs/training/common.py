"""Shared training-loop scaffolding used by all three trainers.

Centralising this avoids re-implementing AMP handling, gradient clipping,
early stopping and history logging three separate times (as separate
notebook cells tend to encourage).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List

import torch
from torch.utils.data import DataLoader

from avavs.utils.logging import get_logger

log = get_logger(__name__)


class EarlyStopper:
    def __init__(self, patience: int = 10, mode: str = "min"):
        self.patience = patience
        self.mode = mode
        self.best = float("inf") if mode == "min" else -float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        improved = value < self.best if self.mode == "min" else value > self.best
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved


def run_epochs(
    *,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_step_fn: Callable,
    eval_step_fn: Callable,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: str,
    use_amp: bool,
    grad_clip: float,
    early_stop_patience: int,
    checkpoint_path: str,
    history_path: str,
) -> Dict[str, List[float]]:
    """Generic train/val loop. `train_step_fn` and `eval_step_fn` each take
    (model, batch, device, scaler_or_None) and return a dict of scalar metrics
    for that batch (already reduced, e.g. {'loss':..., 'cos_sim':...}).
    """
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device == "cuda"))
    stopper = EarlyStopper(patience=early_stop_patience, mode="min")
    history: Dict[str, List[float]] = {}

    best_val_loss = float("inf")
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_metrics_sum: Dict[str, float] = {}
        n_batches = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(use_amp and device == "cuda")):
                metrics = train_step_fn(model, batch, device)
            loss = metrics["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            for k, v in metrics.items():
                train_metrics_sum[k] = train_metrics_sum.get(k, 0.0) + float(v)
            n_batches += 1

        train_avg = {f"train_{k}": v / max(n_batches, 1) for k, v in train_metrics_sum.items()}

        model.eval()
        val_metrics_sum: Dict[str, float] = {}
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                metrics = eval_step_fn(model, batch, device)
                for k, v in metrics.items():
                    val_metrics_sum[k] = val_metrics_sum.get(k, 0.0) + float(v)
                n_val += 1
        val_avg = {f"val_{k}": v / max(n_val, 1) for k, v in val_metrics_sum.items()}

        epoch_log = {**train_avg, **val_avg}
        for k, v in epoch_log.items():
            history.setdefault(k, []).append(v)

        log.info(
            "epoch %3d/%d | " + " | ".join(f"{k}={v:.4f}" for k, v in epoch_log.items()),
            epoch,
            num_epochs,
        )

        val_loss = val_avg.get("val_loss", float("inf"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)

        stopper.step(val_loss)
        if stopper.should_stop:
            log.info("Early stopping at epoch %d (no improvement for %d epochs)", epoch, stopper.patience)
            break

    Path(history_path).parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Always save a final-weights checkpoint too (distinct from best-val one)
    final_path = checkpoint_path.replace(".pt", "_final.pt")
    torch.save(model.state_dict(), final_path)

    return history
