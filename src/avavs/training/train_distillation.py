"""Trainer for Method 2 — Cross-Modal Distillation.

Two phases:
  1. Pretrain the TeacherProbe (linear classifier on frozen VGG16 visual
     embeddings) — then freeze it.
  2. Train the StudentMLP (on the audio+visual concat) to match the
     teacher's soft logits + embedding via KD loss.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from avavs.config import Config
from avavs.models.student_teacher import StudentMLP, TeacherProbe, distillation_loss
from avavs.training.common import run_epochs
from avavs.utils.logging import get_logger

log = get_logger(__name__)


def _pretrain_teacher(
    teacher: TeacherProbe, train_loader: DataLoader, val_loader: DataLoader, device: str, epochs: int = 15
) -> TeacherProbe:
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=1e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        teacher.train()
        total_loss, n = 0.0, 0
        for audio, visual, labels in train_loader:
            visual, labels = visual.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = teacher(visual)
            loss = ce(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss)
            n += 1

        teacher.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for _audio, visual, labels in val_loader:
                visual, labels = visual.to(device), labels.to(device)
                preds = teacher(visual).argmax(dim=-1)
                correct += int((preds == labels).sum())
                total += len(labels)
        acc = correct / max(total, 1)
        log.info("teacher pretrain epoch %d/%d | loss=%.4f | val_acc=%.4f", epoch, epochs, total_loss / max(n, 1), acc)

    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    return teacher


def _make_student_step(teacher: TeacherProbe, cfg: Config):
    def step(model: StudentMLP, batch, device):
        audio, visual, _labels = batch
        audio, visual = audio.to(device), visual.to(device)

        with torch.no_grad():
            teacher_logits = teacher(visual)
            teacher_emb = F.normalize(visual, p=2, dim=-1)  # frozen visual embedding as target

        student_emb, student_logits = model(audio, visual)
        loss = distillation_loss(
            student_emb, student_logits, teacher_emb, teacher_logits,
            temperature=cfg.temperature, alpha=cfg.alpha, beta=cfg.beta,
        )
        cos_sim = F.cosine_similarity(student_emb, teacher_emb, dim=-1).mean()
        mse = F.mse_loss(student_emb, teacher_emb)
        return {"loss": loss, "cos_sim": cos_sim, "mse": mse}

    return step


def train_distillation(cfg: Config, train_ds, val_ds, num_classes: int, device: str) -> StudentMLP:
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    teacher = TeacherProbe(visual_dim=cfg.visual_dim, num_classes=num_classes).to(device)
    teacher = _pretrain_teacher(teacher, train_loader, val_loader, device)

    student = StudentMLP(
        audio_dim=cfg.audio_dim, visual_dim=cfg.visual_dim, latent_dim=cfg.latent_dim, num_classes=num_classes
    ).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    step_fn = _make_student_step(teacher, cfg)
    run_epochs(
        model=student,
        train_loader=train_loader,
        val_loader=val_loader,
        train_step_fn=step_fn,
        eval_step_fn=step_fn,
        optimizer=optimizer,
        num_epochs=cfg.dist_epochs,
        device=device,
        use_amp=cfg.use_amp,
        grad_clip=cfg.grad_clip,
        early_stop_patience=cfg.early_stop_patience,
        checkpoint_path=f"{cfg.output_dir}/distill_model.pt",
        history_path=f"{cfg.output_dir}/distillation_history.json",
    )
    return student
