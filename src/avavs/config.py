"""Central configuration for the AVAVS pipeline.

All hyperparameters live here (mirrors the original notebook's `Config`
dataclass) but can now be overridden from a YAML file or the CLI instead
of being hand-edited inside a notebook cell.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml


@dataclass
class Config:
    # ---- Paths --------------------------------------------------------
    tar_path: str = "./data/AVMIT_VGGish_VGG16.tar"
    cache_dir: str = "./avmit_cache"
    output_dir: str = "./outputs"

    # ---- Dataset --------------------------------------------------------
    max_samples: int = -1          # -1 = use all samples; e.g. 2000 for a quick test
    val_fraction: float = 0.15
    test_fraction: float = 0.10

    # ---- Embedding dims (fixed by AVMIT pre-extraction) -----------------
    audio_dim: int = 128            # VGGish output
    visual_dim: int = 512           # VGG16 fc output
    proj_dim: int = 256             # audio projection dim
    latent_dim: int = 512           # shared latent space

    # ---- Training ---------------------------------------------------------
    batch_size: int = 256
    reg_epochs: int = 50            # Latent Space Regression
    dist_epochs: int = 60           # Cross-Modal Distillation
    naive_epochs: int = 40          # Naive late-fusion baseline
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    use_amp: bool = True            # Mixed precision (GPU only, auto-disabled on CPU)
    num_workers: int = 2
    early_stop_patience: int = 10

    # ---- Distillation -----------------------------------------------------
    temperature: float = 4.0        # KD soft-target temperature
    alpha: float = 0.7              # KL divergence weight
    beta: float = 0.3               # MSE weight
    # (1 - alpha - beta) is implicitly the cosine-similarity weight

    # ---- Evaluation -------------------------------------------------------
    retrieval_k: List[int] = field(default_factory=lambda: [1, 5, 10])
    latency_runs: int = 200
    latency_warmup: int = 20

    # ---- Misc ---------------------------------------------------------
    seed: int = 42
    device: str = "auto"            # "auto" | "cuda" | "cpu"

    # ---------------------------------------------------------------
    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def ensure_dirs(self) -> None:
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls(**raw)

    @classmethod
    def from_yaml_with_overrides(cls, path: str | None, overrides: dict) -> "Config":
        cfg = cls.load(path) if path else cls()
        clean = {k: v for k, v in overrides.items() if v is not None}
        return dataclasses.replace(cfg, **clean)
