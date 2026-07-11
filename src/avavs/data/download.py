"""Download and extract the AVMIT_VGGish_VGG16 dataset.

Replaces the manual "run this cell to download, run this cell to extract"
steps from the original notebook with a single idempotent function that is
safe to call repeatedly (skips work that's already done).
"""
from __future__ import annotations

import tarfile
from pathlib import Path

from avavs.utils.logging import get_logger

log = get_logger(__name__)

AVMIT_DRIVE_FILE_ID = "1ZwqD_Y2aYOSlkLQLEIJ9PWCWcBZOtC1j"


def download_avmit(dest_path: str, drive_file_id: str = AVMIT_DRIVE_FILE_ID) -> str:
    """Download the AVMIT tar archive from Google Drive if not already present."""
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0:
        log.info("Dataset archive already present at %s — skipping download.", dest)
        return str(dest)

    try:
        import gdown
    except ImportError as e:
        raise ImportError(
            "gdown is required to download the dataset. Install with `pip install gdown`."
        ) from e

    url = f"https://drive.google.com/uc?id={drive_file_id}"
    log.info("Downloading AVMIT dataset from %s -> %s", url, dest)
    gdown.download(url, str(dest), quiet=False)

    if not dest.exists():
        raise RuntimeError(f"Download appears to have failed — {dest} was not created.")
    return str(dest)


def extract_tar(tar_path: str, extract_dir: str) -> str:
    """Extract the AVMIT tar archive if the target dir isn't already populated."""
    out = Path(extract_dir)
    out.mkdir(parents=True, exist_ok=True)

    already_extracted = any(out.rglob("*.tfrecord"))
    if already_extracted:
        log.info("TFRecords already extracted under %s — skipping extraction.", out)
        return str(out)

    log.info("Extracting %s -> %s", tar_path, out)
    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(out)  # noqa: S202 — trusted, user-provided dataset archive
    return str(out)
