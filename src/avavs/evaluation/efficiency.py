"""Computational-efficiency measurement: latency, parameter count, FLOPs/MACs."""
from __future__ import annotations

import time
from typing import Callable, Dict, Tuple

import torch


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_latency(
    forward_fn: Callable[[], None], runs: int = 200, warmup: int = 20, device: str = "cpu"
) -> float:
    """Returns mean latency in milliseconds per forward call (batch=1)."""
    for _ in range(warmup):
        forward_fn()
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(runs):
        forward_fn()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return (elapsed / runs) * 1000.0


def measure_flops(model: torch.nn.Module, sample_inputs: Tuple[torch.Tensor, ...]) -> float:
    try:
        from thop import profile
    except ImportError:
        return float("nan")

    with torch.no_grad():
        macs, _ = profile(model, inputs=sample_inputs, verbose=False)
    return float(macs)


def measure_efficiency(
    model: torch.nn.Module,
    sample_inputs: Tuple[torch.Tensor, ...],
    runs: int = 200,
    warmup: int = 20,
    device: str = "cpu",
) -> Dict[str, float]:
    model.eval()

    def _forward():
        with torch.no_grad():
            model(*sample_inputs)

    return {
        "params": count_params(model),
        "latency_ms": measure_latency(_forward, runs=runs, warmup=warmup, device=device),
        "macs": measure_flops(model, sample_inputs),
    }
