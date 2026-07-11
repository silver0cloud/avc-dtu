import numpy as np

from avavs.evaluation.metrics import cosine_similarity, mse, retrieval_metrics


def test_cosine_similarity_identical_vectors_is_one():
    x = np.random.randn(10, 32)
    assert abs(cosine_similarity(x, x) - 1.0) < 1e-5


def test_mse_zero_for_identical():
    x = np.random.randn(10, 32)
    assert mse(x, x) == 0.0


def test_retrieval_metrics_perfect_alignment_gives_top_rank():
    # predictions == gallery -> every query's best match is itself -> rank 1 always
    gallery = np.eye(20, dtype=np.float32)
    pred = gallery.copy()
    result = retrieval_metrics(pred, gallery, ks=[1, 5, 10])
    assert result["R@1"] == 100.0
    assert result["MedR"] == 1.0


def test_retrieval_metrics_shuffled_is_worse_than_perfect():
    rng = np.random.default_rng(0)
    gallery = rng.standard_normal((50, 16)).astype(np.float32)
    noisy_pred = gallery + rng.standard_normal((50, 16)).astype(np.float32) * 5.0
    result = retrieval_metrics(noisy_pred, gallery, ks=[1, 5, 10])
    perfect = retrieval_metrics(gallery, gallery, ks=[1, 5, 10])
    assert result["R@1"] <= perfect["R@1"]
