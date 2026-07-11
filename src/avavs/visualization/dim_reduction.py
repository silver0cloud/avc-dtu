"""PCA / t-SNE / UMAP visualisations of the AV-concat embedding space."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe: never depends on a running display/notebook kernel
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from avavs.utils.logging import get_logger

log = get_logger(__name__)


def _scatter(coords: np.ndarray, labels: np.ndarray, title: str, out_path: str) -> None:
    plt.figure(figsize=(8, 7))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab20", s=8, alpha=0.7)
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.colorbar(scatter, label="class id")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info("Saved %s", out_path)


def plot_pca(av_concat: np.ndarray, labels: np.ndarray, output_dir: str) -> np.ndarray:
    pca = PCA(n_components=min(50, av_concat.shape[1]))
    reduced = pca.fit_transform(av_concat)

    _scatter(reduced[:, :2], labels, "PCA (2D) — AV concat embeddings", f"{output_dir}/pca_2d_train.png")

    plt.figure(figsize=(7, 5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker="o", markersize=3)
    plt.xlabel("# components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA Scree Plot")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_scree.png", dpi=150)
    plt.close()
    log.info("Saved %s/pca_scree.png", output_dir)
    return reduced


def plot_tsne(pca_reduced: np.ndarray, labels: np.ndarray, output_dir: str) -> None:
    tsne = TSNE(n_components=2, random_state=42, init="pca", perplexity=min(30, max(5, len(pca_reduced) // 10)))
    reduced = tsne.fit_transform(pca_reduced)
    _scatter(reduced, labels, "t-SNE (2D) — PCA-reduced AV embeddings", f"{output_dir}/tsne_2d_train.png")


def plot_umap_2d(av_concat: np.ndarray, labels: np.ndarray, output_dir: str) -> None:
    try:
        import umap
    except ImportError:
        log.warning("umap-learn not installed — skipping UMAP 2D plot.")
        return
    reducer = umap.UMAP(n_components=2, random_state=42)
    reduced = reducer.fit_transform(av_concat)
    _scatter(reduced, labels, "UMAP (2D) — AV concat embeddings", f"{output_dir}/umap_2d_train.png")


def plot_umap_3d(av_concat: np.ndarray, labels: np.ndarray, label_names: np.ndarray, output_dir: str) -> None:
    try:
        import umap
        import plotly.express as px
    except ImportError:
        log.warning("umap-learn/plotly not installed — skipping interactive UMAP 3D plot.")
        return

    reducer = umap.UMAP(n_components=3, random_state=42)
    reduced = reducer.fit_transform(av_concat)

    fig = px.scatter_3d(
        x=reduced[:, 0], y=reduced[:, 1], z=reduced[:, 2],
        color=label_names.astype(str), title="UMAP (3D) — AV concat embeddings",
        opacity=0.7,
    )
    out_path = f"{output_dir}/umap_3d.html"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path)
    log.info("Saved %s", out_path)
