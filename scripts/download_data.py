#!/usr/bin/env python
"""Standalone dataset download: `python scripts/download_data.py`."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from avavs.data.download import download_avmit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=str, default="./data/AVMIT_VGGish_VGG16.tar")
    args = parser.parse_args()
    path = download_avmit(args.dest)
    print(f"Dataset ready at: {path}")


if __name__ == "__main__":
    main()
