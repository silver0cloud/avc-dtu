# Project Notes — issues found & discussion items

This tracks what was wrong with the original `.ipynb`-only project, what
this rewrite fixes, and what's still worth discussing/improving next.

## Fixed in this rewrite

- **No notebook, period.** All logic moved into `src/avavs/` as an
  installable package; `.ipynb` is gitignored so one can't creep back in.
- **No reproducibility guarantees.** Notebooks let you run cells out of
  order and get results that don't match a fresh run. The CLI stages
  (`download → prepare-data → train → evaluate → visualize`) always run in
  the same order with the same seeded config.
- **No caching discipline.** Re-parsing every `.tfrecord` on every notebook
  restart is slow and easy to forget to redo after a data change. Now
  `avmit_embeddings.npz` is the single cache artifact, and it's easy to
  force a rebuild (delete the cache dir).
- **No tests.** Added `pytest` coverage for model shapes/losses, metrics,
  config round-tripping, and the splitter — all runnable without the full
  dataset.
- **Config scattered across cells.** Now a single `Config` dataclass,
  loadable from YAML, overridable from the CLI, and saved alongside every
  run's outputs for provenance.
- **Duplicated training-loop code** (AMP handling, grad clipping, early
  stopping, checkpointing, history logging) was likely copy-pasted per
  method in the notebook. Consolidated into `training/common.py`.
- **Plots lost on kernel restart.** Every figure is now written to disk
  under `outputs/`.

## Open items — worth a discussion

These weren't in scope for the "get code out of the notebook" pass but are
the natural next round of improvements:

1. **Experiment tracking.** Right now results are JSON files on disk. Worth
   wiring up Weights & Biases / MLflow / TensorBoard for run comparison,
   especially once we start sweeping hyperparameters.
2. **Data versioning.** The AVMIT tar is pulled from a hardcoded Google
   Drive file ID (`data/download.py`). That's fragile — no checksum
   verification, no versioning if the upstream file changes. Consider DVC
   or at least a SHA256 pin + verification step.
3. **Hyperparameter search.** `configs/default.yaml` hand-picks LR, dropout,
   KD temperature/alpha/beta, etc. None of these have been swept. Optuna or
   a simple grid search harness would help justify the current values.
4. **CI.** `tests/` exist but nothing runs them automatically yet. A
   GitHub Actions workflow (`pytest` + `ruff` + `black --check` on every PR)
   would catch regressions before merge.
5. **Distributed / multi-GPU training.** Current trainers assume a single
   device. Fine for AVMIT-scale data; would need `DistributedDataParallel`
   or `accelerate` if the dataset grows.
6. **Evaluation gallery size.** `retrieval_metrics` currently ranks against
   the full test-set gallery (index-aligned). For very large test sets this
   is O(N²) memory for the similarity matrix — fine at AVMIT scale, but
   worth chunking if the dataset grows substantially.
7. **ImageBind baseline honesty.** The current "ImageBind-style" baseline is
   an *untrained* linear projection, explicitly documented as a
   zero-shot-style reference point, not a faithful reproduction of the real
   ImageBind model. If we want a true ImageBind comparison, that means
   pulling in the actual checkpoint and thinking through licensing/size
   implications.
8. **Class imbalance.** `make_splits` falls back to a non-stratified split
   when any class has <2 examples — worth checking how common that is in
   the real AVMIT label distribution and whether some classes need to be
   merged/dropped.
9. **Model card / write-up.** Once hyperparameters and baselines are
   settled, worth writing a short model card documenting intended use,
   known failure modes, and the exact metric numbers per method.

Bring any of these up whenever — happy to dig into any one of them next.
