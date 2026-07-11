import numpy as np

from avavs.config import Config
from avavs.data.dataset import AVMITDataset, make_splits


def test_config_defaults():
    cfg = Config()
    assert cfg.audio_dim == 128
    assert cfg.visual_dim == 512
    assert cfg.resolve_device() in ("cpu", "cuda")


def test_config_save_and_load_roundtrip(tmp_path):
    cfg = Config(batch_size=64, seed=7)
    path = tmp_path / "cfg.yaml"
    cfg.save(str(path))
    loaded = Config.load(str(path))
    assert loaded.batch_size == 64
    assert loaded.seed == 7


def test_avmit_dataset_len_and_getitem():
    audio = np.random.randn(10, 128).astype(np.float32)
    visual = np.random.randn(10, 512).astype(np.float32)
    labels = np.random.randint(0, 3, size=10).astype(np.int64)
    ds = AVMITDataset(audio, visual, labels)
    assert len(ds) == 10
    a, v, l = ds[0]
    assert a.shape == (128,)
    assert v.shape == (512,)
    assert l.dim() == 0


def test_make_splits_sizes_sum_to_total():
    n = 200
    audio = np.random.randn(n, 128).astype(np.float32)
    visual = np.random.randn(n, 512).astype(np.float32)
    labels = np.random.randint(0, 5, size=n).astype(np.int64)

    splits = make_splits(audio, visual, labels, val_fraction=0.15, test_fraction=0.10, seed=42)
    total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    assert total == n
    assert len(splits["test"]) == round(n * 0.10) or abs(len(splits["test"]) - n * 0.10) <= 2
