"""Deterministic seeding across random, numpy, torch (CPU + CUDA)."""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic-ish; full determinism on GPU trades away speed, so we
        # only opt into it lightly here.
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
