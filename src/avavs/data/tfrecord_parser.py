"""Parse gzip-compressed `SequenceExample` TFRecords from the AVMIT dataset.

Schema (per the dataset spec):
    context:
        filename     : bytes  — clip identifier
        label_index  : int64  — integer class label
        label_name   : bytes  — human-readable class name
    feature_lists:
        audio_embedding             : float, shape (T, 128) — VGGish per-frame
        rgb / visual_embedding      : float, shape (T, 512) — VGG16 fc per-frame

Each clip's temporal embeddings are mean-pooled across time into a single
flat (128,) audio vector and (512,) visual vector, matching the original
pipeline's pooling strategy.
"""
from __future__ import annotations

import glob
from dataclasses import dataclass
from typing import Iterator, List

import numpy as np
import tensorflow as tf

from avavs.utils.logging import get_logger

log = get_logger(__name__)

_CONTEXT_FEATURES = {
    "filename": tf.io.FixedLenFeature([], dtype=tf.string),
    "label_index": tf.io.FixedLenFeature([], dtype=tf.int64),
    "label_name": tf.io.FixedLenFeature([], dtype=tf.string),
}

# The visual feature key differs across AVMIT export versions; we try both.
_VISUAL_KEY_CANDIDATES = ("visual_embedding", "rgb")


@dataclass
class ParsedClip:
    filename: str
    label_index: int
    label_name: str
    audio_emb: np.ndarray   # (128,) mean-pooled
    visual_emb: np.ndarray  # (512,) mean-pooled


def _sequence_feature_spec(visual_key: str) -> dict:
    return {
        "audio_embedding": tf.io.FixedLenSequenceFeature([128], dtype=tf.float32),
        visual_key: tf.io.FixedLenSequenceFeature([512], dtype=tf.float32),
    }


def _detect_visual_key(raw_record: bytes) -> str:
    _, sequence = tf.io.parse_single_sequence_example(
        raw_record,
        context_features=_CONTEXT_FEATURES,
        sequence_features={
            k: tf.io.FixedLenSequenceFeature([512], dtype=tf.float32)
            for k in _VISUAL_KEY_CANDIDATES
        }
        | {"audio_embedding": tf.io.FixedLenSequenceFeature([128], dtype=tf.float32)},
    )
    for key in _VISUAL_KEY_CANDIDATES:
        if sequence[key].shape[0] > 0:
            return key
    # Fall back to the first candidate — extraction errors will surface loudly
    # downstream rather than being silently swallowed here.
    return _VISUAL_KEY_CANDIDATES[0]


def find_tfrecords(root_dir: str) -> List[str]:
    files = sorted(glob.glob(f"{root_dir}/**/*.tfrecord", recursive=True))
    if not files:
        raise FileNotFoundError(f"No .tfrecord files found under {root_dir}")
    return files


def iter_parsed_clips(tfrecord_paths: List[str], max_samples: int = -1) -> Iterator[ParsedClip]:
    """Stream-parse clips, mean-pooling temporal audio/visual embeddings."""
    dataset = tf.data.TFRecordDataset(tfrecord_paths, compression_type="GZIP")

    visual_key = "visual_embedding"
    count = 0
    for raw_record in dataset:
        if max_samples > 0 and count >= max_samples:
            break

        record_bytes = raw_record.numpy()
        if count == 0:
            visual_key = _detect_visual_key(record_bytes)
            log.info("Detected visual feature key: %s", visual_key)

        context, sequence = tf.io.parse_single_sequence_example(
            record_bytes,
            context_features=_CONTEXT_FEATURES,
            sequence_features=_sequence_feature_spec(visual_key),
        )

        audio = sequence["audio_embedding"].numpy()   # (T, 128)
        visual = sequence[visual_key].numpy()          # (T, 512)
        if audio.shape[0] == 0 or visual.shape[0] == 0:
            continue  # skip malformed/empty clips rather than crash the run

        yield ParsedClip(
            filename=context["filename"].numpy().decode("utf-8", errors="replace"),
            label_index=int(context["label_index"].numpy()),
            label_name=context["label_name"].numpy().decode("utf-8", errors="replace"),
            audio_emb=audio.mean(axis=0).astype(np.float32),
            visual_emb=visual.mean(axis=0).astype(np.float32),
        )
        count += 1
