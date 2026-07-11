"""Per-method training curves + the final 3-row comparative dashboard."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from avavs.utils.logging import get_logger

log = get_logger(__name__)


def plot_training_curves(history: Dict[str, List[float]], title: str, out_path: str) -> None:
    keys = sorted({k.split("train_")[-1].split("val_")[-1] for k in history if "loss" in k or "cos_sim" in k or "mse" in k})
    metric_names = [m for m in ["loss", "cos_sim", "mse"] if any(m in k for k in history)]

    fig, axes = plt.subplots(1, len(metric_names), figsize=(6 * len(metric_names), 4))
    if len(metric_names) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metric_names):
        train_key, val_key = f"train_{metric}", f"val_{metric}"
        if train_key in history:
            ax.plot(history[train_key], label="train")
        if val_key in history:
            ax.plot(history[val_key], label="val")
        ax.set_title(metric)
        ax.set_xlabel("epoch")
        ax.legend()

    fig.suptitle(title)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info("Saved %s", out_path)


def plot_comparative_dashboard(results: Dict[str, dict], output_dir: str) -> None:
    """3-row dashboard: (1) latent+retrieval metrics, (2) efficiency, (3) CosSim distributions."""
    methods = list(results.keys())
    fig, axes = plt.subplots(3, 4, figsize=(22, 14))

    # Row 1 — CosSim | MSE | Recall@K bars | MedR
    cos_vals = [results[m]["cos_sim"] for m in methods]
    mse_vals = [results[m]["mse"] for m in methods]
    axes[0, 0].bar(methods, cos_vals, color="steelblue")
    axes[0, 0].set_title("Cosine Similarity (higher better)")
    axes[0, 0].tick_params(axis="x", rotation=45)

    axes[0, 1].bar(methods, mse_vals, color="indianred")
    axes[0, 1].set_title("MSE (lower better)")
    axes[0, 1].tick_params(axis="x", rotation=45)

    ks = sorted({int(k.split("@")[1]) for m in methods for k in results[m] if k.startswith("R@")})
    width = 0.8 / max(len(ks), 1)
    for i, k in enumerate(ks):
        vals = [results[m].get(f"R@{k}", 0.0) for m in methods]
        x = np.arange(len(methods)) + i * width
        axes[0, 2].bar(x, vals, width=width, label=f"R@{k}")
    axes[0, 2].set_xticks(np.arange(len(methods)) + width * (len(ks) - 1) / 2)
    axes[0, 2].set_xticklabels(methods, rotation=45)
    axes[0, 2].set_title("Recall@K (higher better)")
    axes[0, 2].legend()

    medr_vals = [results[m].get("MedR", np.nan) for m in methods]
    axes[0, 3].bar(methods, medr_vals, color="darkorange")
    axes[0, 3].set_title("Median Rank (lower better)")
    axes[0, 3].tick_params(axis="x", rotation=45)

    # Row 2 — Latency | Params | FLOPs | Summary table
    latency_vals = [results[m].get("latency_ms", np.nan) for m in methods]
    axes[1, 0].bar(methods, latency_vals, color="seagreen")
    axes[1, 0].set_title("Inference Latency (ms/sample)")
    axes[1, 0].tick_params(axis="x", rotation=45)

    param_vals = [results[m].get("params", np.nan) for m in methods]
    axes[1, 1].bar(methods, param_vals, color="slateblue")
    axes[1, 1].set_title("Parameter Count")
    axes[1, 1].tick_params(axis="x", rotation=45)

    macs_vals = [results[m].get("macs", np.nan) for m in methods]
    axes[1, 2].bar(methods, macs_vals, color="goldenrod")
    axes[1, 2].set_title("MACs / FLOPs")
    axes[1, 2].tick_params(axis="x", rotation=45)

    axes[1, 3].axis("off")
    table_data = [[m, f"{results[m]['cos_sim']:.3f}", f"{results[m].get('R@1', float('nan')):.1f}"] for m in methods]
    axes[1, 3].table(cellText=table_data, colLabels=["Method", "CosSim", "R@1"], loc="center")
    axes[1, 3].set_title("Summary")

    # Row 3 — CosSim distribution (box/violin) per method
    for i, m in enumerate(methods):
        dist = results[m].get("cos_sim_distribution")
        ax = axes[2, i] if i < 4 else None
        if ax is not None and dist is not None:
            ax.violinplot(dist, showmeans=True)
            ax.set_title(f"{m} — CosSim distribution")
    for i in range(len(methods), 4):
        axes[2, i].axis("off")

    plt.tight_layout()
    out_path = f"{output_dir}/comparative_dashboard.png"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info("Saved %s", out_path)
