"""Command-line entry point for the full AVAVS pipeline.

This is the single script that replaces every notebook cell. Typical usage:

    avavs run --config configs/default.yaml
    avavs run --config configs/quick_test.yaml   # fast smoke test

Or run individual stages:

    avavs download --config configs/default.yaml
    avavs train --config configs/default.yaml
    avavs evaluate --config configs/default.yaml
    avavs visualize --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from avavs.config import Config
from avavs.data.dataset import load_or_build_embeddings, make_splits
from avavs.data.download import download_avmit, extract_tar
from avavs.evaluation.efficiency import measure_efficiency
from avavs.evaluation.metrics import cosine_similarity, mse, retrieval_metrics
from avavs.models.baselines import ImageBindProjection, NaiveLateFusionMLP
from avavs.models.fusion_mlp import FusionMLP
from avavs.models.student_teacher import StudentMLP
from avavs.training.train_baseline import train_naive_baseline
from avavs.training.train_distillation import train_distillation
from avavs.training.train_regression import train_regression
from avavs.utils.logging import get_logger
from avavs.utils.seed import set_seed
from avavs.visualization.dashboard import plot_comparative_dashboard, plot_training_curves
from avavs.visualization.dim_reduction import plot_pca, plot_tsne, plot_umap_2d, plot_umap_3d

log = get_logger("avavs.cli")


# ------------------------------------------------------------------ helpers
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AVAVS end-to-end pipeline")
    parser.add_argument(
        "stage",
        choices=["download", "prepare-data", "train", "evaluate", "visualize", "run"],
        help="Which stage to run. 'run' executes the full pipeline end-to-end.",
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--max-samples", type=int, default=None, help="Override Config.max_samples")
    parser.add_argument("--tar-path", type=str, default=None, help="Override Config.tar_path")
    parser.add_argument("--output-dir", type=str, default=None, help="Override Config.output_dir")
    parser.add_argument("--skip-download", action="store_true", help="Assume the tar archive already exists")
    return parser.parse_args()


def _load_config(args: argparse.Namespace) -> Config:
    overrides = {
        "max_samples": args.max_samples,
        "tar_path": args.tar_path,
        "output_dir": args.output_dir,
    }
    cfg = Config.from_yaml_with_overrides(args.config, overrides)
    cfg.ensure_dirs()
    return cfg


def _prepare_data(cfg: Config):
    extract_dir = str(Path(cfg.tar_path).with_suffix(""))
    extract_tar(cfg.tar_path, extract_dir)
    audio, visual, labels, label_names = load_or_build_embeddings(
        extract_dir, cfg.cache_dir, max_samples=cfg.max_samples
    )
    splits = make_splits(audio, visual, labels, cfg.val_fraction, cfg.test_fraction, cfg.seed)
    num_classes = int(labels.max()) + 1
    return splits, num_classes, label_names


def _sample_batch_for_efficiency(device: str):
    audio = torch.randn(1, 128, device=device)
    visual = torch.randn(1, 512, device=device)
    return audio, visual


def _evaluate_method(name: str, pred: np.ndarray, gt: np.ndarray, ks) -> dict:
    result = {"cos_sim": cosine_similarity(pred, gt), "mse": mse(pred, gt)}
    result.update(retrieval_metrics(pred, gt, ks))

    pred_n = pred / (np.linalg.norm(pred, axis=-1, keepdims=True) + 1e-8)
    gt_n = gt / (np.linalg.norm(gt, axis=-1, keepdims=True) + 1e-8)
    result["cos_sim_distribution"] = np.sum(pred_n * gt_n, axis=-1).tolist()
    log.info("[%s] cos_sim=%.4f mse=%.6f R@1=%.1f MedR=%.1f", name, result["cos_sim"], result["mse"], result["R@1"], result["MedR"])
    return result


# ------------------------------------------------------------------ stages
def stage_download(cfg: Config) -> None:
    download_avmit(cfg.tar_path)


def stage_train(cfg: Config):
    set_seed(cfg.seed)
    device = cfg.resolve_device()
    log.info("Using device: %s", device)

    splits, num_classes, _ = _prepare_data(cfg)
    train_ds, val_ds = splits["train"], splits["val"]

    log.info("=== Training Baseline 1: Naive Late Fusion ===")
    naive_model = train_naive_baseline(cfg, train_ds, val_ds, device)

    log.info("=== Training Method 1: Latent Space Regression (FusionMLP) ===")
    reg_model = train_regression(cfg, train_ds, val_ds, device)

    log.info("=== Training Method 2: Cross-Modal Distillation ===")
    dist_model = train_distillation(cfg, train_ds, val_ds, num_classes, device)

    return {"naive": naive_model, "regression": reg_model, "distillation": dist_model}


def stage_evaluate(cfg: Config, models: dict | None = None) -> dict:
    set_seed(cfg.seed)
    device = cfg.resolve_device()
    splits, num_classes, _ = _prepare_data(cfg)
    test_ds = splits["test"]

    audio = test_ds.audio.to(device)
    visual = test_ds.visual.to(device)
    gt = visual.detach().cpu().numpy()

    if models is None:
        models = {}
        naive = NaiveLateFusionMLP(cfg.audio_dim, cfg.visual_dim, cfg.latent_dim).to(device)
        naive.load_state_dict(torch.load(f"{cfg.output_dir}/naive_baseline_model.pt", map_location=device))
        models["naive"] = naive

        reg = FusionMLP(cfg.audio_dim, cfg.latent_dim).to(device)
        reg.load_state_dict(torch.load(f"{cfg.output_dir}/reg_model.pt", map_location=device))
        models["regression"] = reg

        dist = StudentMLP(cfg.audio_dim, cfg.visual_dim, cfg.latent_dim, num_classes).to(device)
        dist.load_state_dict(torch.load(f"{cfg.output_dir}/distill_model.pt", map_location=device))
        models["distillation"] = dist

    results: dict = {}

    with torch.no_grad():
        models["naive"].eval()
        pred = models["naive"](audio, visual).cpu().numpy()
        results["NaiveLateFusion"] = _evaluate_method("NaiveLateFusion", pred, gt, cfg.retrieval_k)
        eff = measure_efficiency(models["naive"], _sample_batch_for_efficiency(device), cfg.latency_runs, cfg.latency_warmup, device)
        results["NaiveLateFusion"].update(eff)

        models["regression"].eval()
        pred = models["regression"](audio).cpu().numpy()
        results["LatentRegression"] = _evaluate_method("LatentRegression", pred, gt, cfg.retrieval_k)
        eff = measure_efficiency(models["regression"], (_sample_batch_for_efficiency(device)[0],), cfg.latency_runs, cfg.latency_warmup, device)
        results["LatentRegression"].update(eff)

        models["distillation"].eval()
        pred, _ = models["distillation"](audio, visual)
        pred = pred.cpu().numpy()
        results["CrossModalDistillation"] = _evaluate_method("CrossModalDistillation", pred, gt, cfg.retrieval_k)
        eff = measure_efficiency(models["distillation"], _sample_batch_for_efficiency(device), cfg.latency_runs, cfg.latency_warmup, device)
        results["CrossModalDistillation"].update(eff)

        # ImageBind-style zero-shot ceiling — untrained linear projection, no gradient steps
        imagebind = ImageBindProjection(cfg.audio_dim, cfg.visual_dim, cfg.latent_dim).to(device)
        imagebind.eval()
        pred = imagebind(audio, visual).cpu().numpy()
        results["ImageBindZeroShot"] = _evaluate_method("ImageBindZeroShot", pred, gt, cfg.retrieval_k)
        eff = measure_efficiency(imagebind, _sample_batch_for_efficiency(device), cfg.latency_runs, cfg.latency_warmup, device)
        results["ImageBindZeroShot"].update(eff)

    out_path = f"{cfg.output_dir}/full_comparative_results.json"
    serializable = {
        k: {kk: vv for kk, vv in v.items() if kk != "cos_sim_distribution"} for k, v in results.items()
    }
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    log.info("Saved evaluation results -> %s", out_path)

    return results


def stage_visualize(cfg: Config, eval_results: dict | None = None) -> None:
    splits, num_classes, label_names_full = _prepare_data(cfg)
    train_idx = splits["train_idx"]
    train_ds = splits["train"]

    av_concat = np.concatenate([train_ds.audio.numpy(), train_ds.visual.numpy()], axis=1)
    labels = train_ds.labels.numpy()
    label_names = label_names_full[train_idx]

    pca_reduced = plot_pca(av_concat, labels, cfg.output_dir)
    plot_tsne(pca_reduced, labels, cfg.output_dir)
    plot_umap_2d(av_concat, labels, cfg.output_dir)
    plot_umap_3d(av_concat, labels, label_names, cfg.output_dir)

    for name, fname, title in [
        ("regression_history.json", "regression_training_curves.png", "Latent Space Regression"),
        ("distillation_history.json", "distillation_training_curves.png", "Cross-Modal Distillation"),
        ("naive_history.json", "naive_late_fusion.png", "Naive Late Fusion"),
    ]:
        hpath = Path(cfg.output_dir) / name
        if hpath.exists():
            with open(hpath) as f:
                history = json.load(f)
            plot_training_curves(history, title, f"{cfg.output_dir}/{fname}")

    if eval_results is not None:
        plot_comparative_dashboard(eval_results, cfg.output_dir)


def stage_run(cfg: Config) -> None:
    stage_download(cfg)
    models = stage_train(cfg)
    results = stage_evaluate(cfg, models)
    stage_visualize(cfg, results)
    log.info("Pipeline complete. All outputs written to %s", cfg.output_dir)


# ------------------------------------------------------------------ main
def main() -> None:
    args = _parse_args()
    cfg = _load_config(args)

    if args.stage == "download":
        stage_download(cfg)
    elif args.stage == "prepare-data":
        _prepare_data(cfg)
    elif args.stage == "train":
        stage_train(cfg)
    elif args.stage == "evaluate":
        stage_evaluate(cfg)
    elif args.stage == "visualize":
        stage_visualize(cfg)
    elif args.stage == "run":
        stage_run(cfg)


if __name__ == "__main__":
    main()
