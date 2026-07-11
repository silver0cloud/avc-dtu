# AVAVS — Audio-Visual Approximation of Video Semantic Space

Learns to approximate a full video's semantic embedding (VGG16, 512-d) from
just its audio track (VGGish, 128-d) — and compares that against
cross-modal distillation, a naive late-fusion baseline, and an
ImageBind-style zero-shot projection ceiling.

This repository is a **script-based, no-notebook** rewrite of the original
`AVAVSS_pipeline.ipynb`. Every pipeline stage — data prep, training,
evaluation, visualization — is now a testable Python module you can run
from the CLI or import directly, instead of a linear sequence of notebook
cells you have to re-run top-to-bottom by hand.

## What changed vs. the notebook

| Before (notebook) | Now |
|---|---|
| One 100+-cell `.ipynb`, run top-to-bottom | `src/avavs/` installable package + `avavs` CLI |
| Hyperparameters hand-edited inside a cell | `configs/*.yaml`, overridable from the CLI |
| No tests — correctness only checked by eyeballing cell output | `tests/` with `pytest`, runs in CI |
| Re-parsing all TFRecords every time you reopen the notebook | Embeddings cached once to `avmit_cache/*.npz` |
| Global mutable state shared across cells (easy to silently break) | Pure functions + explicit `Config` object, no hidden state |
| Plots rendered inline, lost when the kernel restarts | Every figure written to `outputs/*.png` / `.html` |
| Hard to diff, hard to code-review, hard to reuse pieces | Plain `.py` files — normal git diffs, normal imports |

## Project layout

```
src/avavs/
  config.py                 # single source of truth for all hyperparameters
  data/
    download.py              # dataset download + tar extraction
    tfrecord_parser.py        # SequenceExample -> pooled (audio, visual) vectors
    dataset.py                 # embedding cache, torch Dataset, train/val/test split
  models/
    fusion_mlp.py             # Method 1: Latent Space Regression
    student_teacher.py        # Method 2: Cross-Modal Distillation (teacher + student)
    baselines.py               # Naive Late Fusion + ImageBind-style zero-shot ceiling
  training/
    train_regression.py, train_distillation.py, train_baseline.py
    common.py                  # shared AMP/early-stop/checkpoint/history loop
  evaluation/
    metrics.py                 # cosine sim, MSE, Recall@K, MedR
    efficiency.py               # latency, param count, FLOPs
  visualization/
    dim_reduction.py            # PCA / t-SNE / UMAP
    dashboard.py                  # training curves + final comparative dashboard
  cli.py                        # `avavs {download,prepare-data,train,evaluate,visualize,run}`
scripts/
  run_pipeline.py               # `python scripts/run_pipeline.py` (no install needed)
  download_data.py
configs/
  default.yaml, quick_test.yaml
tests/                          # pytest unit tests (models, metrics, config, splits)
docs/
  PROJECT_NOTES.md               # rewrite rationale + open discussion items
```

## Setup

```bash
git clone https://github.com/silver0cloud/avavs-dtu.git
cd avavs-dtu
pip install -e .              # installs the `avavs` package + CLI
# or, for a plain venv without editable install:
pip install -r requirements.txt
```

## Usage

```bash
# Full pipeline: download -> prepare data -> train all 3 methods -> evaluate -> visualize
avavs run --config configs/default.yaml

# Fast smoke test on a 2,000-sample subset, 5 epochs each (a few minutes on CPU)
avavs run --config configs/quick_test.yaml

# Or run stages individually
avavs download --config configs/default.yaml
avavs prepare-data --config configs/default.yaml
avavs train --config configs/default.yaml
avavs evaluate --config configs/default.yaml
avavs visualize --config configs/default.yaml

# CLI overrides for a one-off run without editing YAML
avavs run --config configs/default.yaml --max-samples 5000 --output-dir ./outputs_5k
```

Without installing the package, everything also works straight from a checkout:

```bash
python scripts/run_pipeline.py --config configs/quick_test.yaml
```

### Outputs

Everything lands under `{output_dir}` (default `./outputs`):
- `*_model.pt`, `*_model_final.pt` — best-val-loss and final checkpoints per method
- `*_history.json` — per-epoch train/val loss, cosine similarity, MSE
- `full_comparative_results.json` — CosSim / MSE / R@1,5,10 / MedR / latency / params / MACs for all 4 methods
- `comparative_dashboard.png` — 3-row summary figure across all methods
- `pca_2d_train.png`, `tsne_2d_train.png`, `umap_2d_train.png`, `umap_3d.html` — embedding-space visualizations
- `*_training_curves.png` — per-method loss/CosSim/MSE curves

## Methods compared

1. **Naive Late Fusion** (baseline) — 2-layer MLP on `[audio; visual]` concat, no normalization. Minimum viable baseline; every other method should beat it.
2. **Latent Space Regression** — audio-only MLP predicting the visual embedding directly (`0.5·MSE + 0.5·(1−CosSim)`). No visual input at inference time — this is the genuinely audio-only method.
3. **Cross-Modal Distillation** — a linear probe teacher (visual → class logits) distills into a student that sees `[audio; visual]` and is trained with `α·KD + β·MSE + (1−α−β)·(1−CosSim)`.
4. **ImageBind-style zero-shot** — untrained linear-projection ceiling (no gradient steps), included as a reference point rather than a competitor.

Full hyperparameters live in `src/avavs/config.py` / `configs/default.yaml`.

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Tests cover model shapes/losses, retrieval and similarity metrics, config
save/load round-tripping, and the stratified train/val/test splitter — all
runnable without the actual (large) dataset present.

## Known limitations / open items for discussion

See [`docs/PROJECT_NOTES.md`](./docs/PROJECT_NOTES.md) for the full list of issues
found in the original notebook and what's still open for the next round of
improvements (data versioning, experiment tracking, distributed training,
hyperparameter sweeps, CI, model card, etc).
