"""Latent-proximity and retrieval metrics used across all methods.

A. Latent space proximity: cosine similarity (higher better), MSE (lower better)
B. Downstream retrieval: R@1, R@5, R@10 (higher better), MedR (lower better)
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def cosine_similarity(pred: np.ndarray, target: np.ndarray) -> float:
    pred_n = pred / (np.linalg.norm(pred, axis=-1, keepdims=True) + 1e-8)
    target_n = target / (np.linalg.norm(target, axis=-1, keepdims=True) + 1e-8)
    return float(np.mean(np.sum(pred_n * target_n, axis=-1)))


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def retrieval_metrics(pred: np.ndarray, gt_gallery: np.ndarray, ks: List[int]) -> Dict[str, float]:
    """For each predicted embedding, rank the full GT gallery by cosine
    similarity and report where the matching (same-index) item lands.

    pred:       (N, D) predicted embeddings (queries)
    gt_gallery: (N, D) ground-truth visual embeddings (gallery, index-aligned
                with `pred` — i.e. row i of pred should best-match row i of
                gt_gallery)
    """
    pred_n = pred / (np.linalg.norm(pred, axis=-1, keepdims=True) + 1e-8)
    gal_n = gt_gallery / (np.linalg.norm(gt_gallery, axis=-1, keepdims=True) + 1e-8)

    sims = pred_n @ gal_n.T  # (N, N)
    ranks = np.zeros(len(pred), dtype=np.int64)
    order = np.argsort(-sims, axis=1)  # descending similarity
    correct_rank_pos = (order == np.arange(len(pred))[:, None]).argmax(axis=1)
    ranks = correct_rank_pos + 1  # 1-indexed rank

    results: Dict[str, float] = {}
    for k in ks:
        results[f"R@{k}"] = float(np.mean(ranks <= k) * 100.0)
    results["MedR"] = float(np.median(ranks))
    return results
