"""Embedding cache + torch Dataset + stratified train/val/test split.

On first run, parsed & mean-pooled embeddings are written to
`{cache_dir}/avmit_embeddings.npz`. Every subsequent run loads directly
from that cache — no TFRecord re-parsing needed (mirrors the original
notebook's caching behaviour, but as reusable, testable code).
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from avavs.data.tfrecord_parser import find_tfrecords, iter_parsed_clips
from avavs.utils.logging import get_logger

log = get_logger(__name__)

CACHE_FILENAME = "avmit_embeddings.npz"


def load_or_build_embeddings(
    tfrecord_dir: str, cache_dir: str, max_samples: int = -1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (audio_embs, visual_embs, label_ids, label_names), building+caching if needed."""
    cache_path = Path(cache_dir) / CACHE_FILENAME

    if cache_path.exists():
        log.info("Loading cached embeddings from %s", cache_path)
        data = np.load(cache_path, allow_pickle=True)
        return data["audio_embs"], data["visual_embs"], data["label_ids"], data["label_names"]

    log.info("No cache found — parsing TFRecords from %s", tfrecord_dir)
    tfrecords = find_tfrecords(tfrecord_dir)
    log.info("Found %d tfrecord shard(s)", len(tfrecords))

    audio_list, visual_list, label_ids, label_names = [], [], [], []
    for clip in iter_parsed_clips(tfrecords, max_samples=max_samples):
        audio_list.append(clip.audio_emb)
        visual_list.append(clip.visual_emb)
        label_ids.append(clip.label_index)
        label_names.append(clip.label_name)

    if not audio_list:
        raise RuntimeError(f"No clips were parsed from {tfrecord_dir} — check the archive contents.")

    audio_embs = np.stack(audio_list).astype(np.float32)
    visual_embs = np.stack(visual_list).astype(np.float32)
    label_ids = np.array(label_ids, dtype=np.int64)
    label_names = np.array(label_names, dtype=object)

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        audio_embs=audio_embs,
        visual_embs=visual_embs,
        label_ids=label_ids,
        label_names=label_names,
    )
    log.info("Cached %d parsed clips -> %s", len(audio_embs), cache_path)
    return audio_embs, visual_embs, label_ids, label_names


class AVMITDataset(Dataset):
    """Simple in-memory tensor dataset over pre-pooled (audio, visual, label) triples."""

    def __init__(self, audio_embs: np.ndarray, visual_embs: np.ndarray, label_ids: np.ndarray):
        assert len(audio_embs) == len(visual_embs) == len(label_ids)
        self.audio = torch.from_numpy(audio_embs).float()
        self.visual = torch.from_numpy(visual_embs).float()
        self.labels = torch.from_numpy(label_ids).long()

    def __len__(self) -> int:
        return len(self.audio)

    def __getitem__(self, idx: int):
        return self.audio[idx], self.visual[idx], self.labels[idx]


def make_splits(
    audio_embs: np.ndarray,
    visual_embs: np.ndarray,
    label_ids: np.ndarray,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict:
    """Stratified 75/15/10-style split (fractions are configurable)."""
    n = len(audio_embs)
    idx = np.arange(n)

    # Stratification requires every class to have >=2 members; fall back to a
    # plain random split if the label distribution is too sparse (e.g. tiny
    # quick-test subsets).
    _, class_counts = np.unique(label_ids, return_counts=True)
    stratify = label_ids if class_counts.min() >= 2 else None

    train_val_idx, test_idx = train_test_split(
        idx, test_size=test_fraction, random_state=seed, stratify=stratify
    )
    stratify_tv = label_ids[train_val_idx] if stratify is not None else None
    relative_val = val_fraction / (1.0 - test_fraction)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=relative_val, random_state=seed, stratify=stratify_tv
    )

    log.info(
        "Split sizes -> train: %d, val: %d, test: %d", len(train_idx), len(val_idx), len(test_idx)
    )

    def subset(indices: np.ndarray) -> AVMITDataset:
        return AVMITDataset(audio_embs[indices], visual_embs[indices], label_ids[indices])

    return {
        "train": subset(train_idx),
        "val": subset(val_idx),
        "test": subset(test_idx),
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }
