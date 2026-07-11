#!/usr/bin/env python
"""Convenience wrapper: `python scripts/run_pipeline.py --config configs/default.yaml`.

Equivalent to `avavs run --config ...` once the package is pip-installed;
this script lets you run the pipeline straight from a git checkout without
installing anything, as long as `src/` is on PYTHONPATH (see README).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from avavs.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "run"] + sys.argv[1:]
    main()
