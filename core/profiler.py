"""Simple, robust latency measurement utilities.

This module provides GPU-friendly profilers with warmup, multiple repeats,
median/percentile reporting, and optional outlier rejection via MAD.

Design goals:
  - Family-agnostic: take a callable `forward(model, x)` or rely on HF `.forward`
  - Deterministic when desired; avoids autograd by default
  - Works with CUDA or CPU; uses `torch.cuda.Event` for accurate GPU timing

Key APIs:
  - measure_latency_ms(model, input_shape | input_tensor, ...)
  - profile(model, sample, settings) -> {mean, p50, p90, p95, p99}
  - LatencyProfiler(settings).measure(...)
  - profile_many_shapes(model, shapes, settings)
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

import contextlib
import math
import time

import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

@dataclass
class ProfileSettings:
    warmup: int = 10
    iters: int = 50
    percentile: Sequence[int] = (50, 90, 95, 99)
    sync_each_iter: bool = True
    use_inference_mode: bool = True
    cuda_graph: bool = False  # advanced users can enable with static shapes
    reject_outliers_mad: float = 0.0  # e.g., 3.5 to drop extreme spikes
    cudnn_benchmark: bool = True
    deterministic: bool = False  # sets cudnn.deterministic


# -----------------------------------------------------------------------------
# Context helpers
# -----------------------------------------------------------------------------

@contextlib.contextmanager
def _torch_backend_ctx(settings: ProfileSettings):
    prev_bench = torch.backends.cudnn.benchmark
    prev_det = torch.backends.cudnn.deterministic
    try:
        torch.backends.cudnn.benchmark = bool(settings.cudnn_benchmark)
        torch.backends.cudnn.deterministic = bool(settings.deterministic)
        yield
    finally:
        torch.backends.cudnn.benchmark = prev_bench
        torch.backends.cudnn.deterministic = prev_det


def _percentiles(sorted_vals: Sequence[float], qs: Sequence[int]) -> Dict[int, float]:
    n = len(sorted_vals)
    if n == 0:
        return {q: float("nan") for q in qs}
    out = {}
    for q in qs:
        if n == 1:
            out[q] = sorted_vals[0]
            continue
        k = (q / 100.0) * (n - 1)
        f = math.floor(k)
        c = min(n - 1, f + 1)
        if f == c:
            out[q] = sorted_vals[int(k)]
        else:
            d0 = sorted_vals[f] * (c - k)
            d1 = sorted_vals[c] * (k - f)
            out[q] = d0 + d1
    return out


def _apply_mad_filter(vals: Sequence[float], thresh: float) -> Sequence[float]:
    if thresh <= 0 or len(vals) < 5:
        return vals
    med = median(vals)
    dev = [abs(v - med) for v in vals]
    mad = median(dev) or 1e-12
    keep = [v for v, d in zip(vals, dev) if (d / mad) <= thresh]
    return keep if keep else vals


# -----------------------------------------------------------------------------
# Core measurement
# -----------------------------------------------------------------------------

@torch.inference_mode()
def measure_latency_ms(
    model: nn.Module,
    sample: torch.Tensor | Tuple[int, ...],
    *,
    settings: Optional[ProfileSettings] = None,
    device: str = "cuda",
    forward_fn: Optional[Callable[[nn.Module, torch.Tensor], torch.Tensor]] = None,
) -> Tuple[float, float]:
    """Return (mean_ms, p95_ms) over `iters` measurements.

    If `sample` is a shape tuple, a random tensor is created on-device.
    The default forward calls `model(pixel_values=x)` if available, else `model(x)`.
    """
    cfg = settings or ProfileSettings()

    with _torch_backend_ctx(cfg):
        m = model.to(device).eval()
        if isinstance(sample, torch.Tensor):
            x = sample.to(device)
        else:
            x = torch.randn(*sample, device=device)

        # Default forward
        def _fwd(mod, inp):
            if hasattr(mod, "forward"):
                try:
                    return mod(pixel_values=inp)
                except TypeError:
                    return mod(inp)
            return mod(inp)

        fn = forward_fn or _fwd

        # Warmup
        if torch.cuda.is_available() and device.startswith("cuda"):
            for _ in range(cfg.warmup):
                _ = fn(m, x)
            torch.cuda.synchronize()
        else:
            for _ in range(cfg.warmup):
                _ = fn(m, x)
            torch.cuda.synchronize() if torch.cuda.is_available() else None

        times: list[float] = []
        if torch.cuda.is_available() and device.startswith("cuda"):
            for _ in range(cfg.iters):
                t0 = torch.cuda.Event(enable_timing=True)
                t1 = torch.cuda.Event(enable_timing=True)
                t0.record()
                _ = fn(m, x)
                t1.record()
                if cfg.sync_each_iter:
                    torch.cuda.synchronize()
                times.append(t0.elapsed_time(t1))  # milliseconds
        else:
            for _ in range(cfg.iters):
                t0 = time.perf_counter()
                _ = fn(m, x)
                if cfg.sync_each_iter and torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000.0)

        times = sorted(_apply_mad_filter(times, cfg.reject_outliers_mad))
        mean_ms = sum(times) / max(1, len(times))
        p = _percentiles(times, cfg.percentile)
        p95 = p.get(95, times[int(0.95 * (len(times) - 1))] if times else float("nan"))
        return mean_ms, p95


# Higher level wrapper returning multiple percentiles
@torch.inference_mode()
def profile(
    model: nn.Module,
    sample: torch.Tensor | Tuple[int, ...],
    *,
    settings: Optional[ProfileSettings] = None,
    device: str = "cuda",
    forward_fn: Optional[Callable[[nn.Module, torch.Tensor], torch.Tensor]] = None,
) -> Dict[str, float]:
    cfg = settings or ProfileSettings()
    mean_ms, _ = measure_latency_ms(model, sample, settings=cfg, device=device, forward_fn=forward_fn)
    # Re-run percentile calc on same settings for consistency
    m = model.to(device).eval()
    if isinstance(sample, torch.Tensor):
        x = sample.to(device)
    else:
        x = torch.randn(*sample, device=device)

    if torch.cuda.is_available() and device.startswith("cuda"):
        times = []
        for _ in range(cfg.iters):
            t0 = torch.cuda.Event(True); t1 = torch.cuda.Event(True)
            t0.record(); _ = (forward_fn or (lambda a, b: a(pixel_values=b)))(m, x); t1.record();
            if cfg.sync_each_iter: torch.cuda.synchronize()
            times.append(t0.elapsed_time(t1))
    else:
        times = []
        for _ in range(cfg.iters):
            t0 = time.perf_counter(); _ = (forward_fn or (lambda a, b: a(pixel_values=b)))(m, x); t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    times = sorted(_apply_mad_filter(times, cfg.reject_outliers_mad))
    percs = _percentiles(times, cfg.percentile)
    out = {"mean": sum(times) / max(1, len(times))}
    out.update({f"p{q}": v for q, v in percs.items()})
    return out


class LatencyProfiler:
    """Reusable profiler with fixed settings."""

    def __init__(self, settings: Optional[ProfileSettings] = None, device: str = "cuda"):
        self.settings = settings or ProfileSettings()
        self.device = device

    def measure(self, model: nn.Module, sample: torch.Tensor | Tuple[int, ...], *, forward_fn: Optional[Callable[[nn.Module, torch.Tensor], torch.Tensor]] = None) -> Tuple[float, float]:
        return measure_latency_ms(model, sample, settings=self.settings, device=self.device, forward_fn=forward_fn)

    def profile(self, model: nn.Module, sample: torch.Tensor | Tuple[int, ...], *, forward_fn: Optional[Callable[[nn.Module, torch.Tensor], torch.Tensor]] = None) -> Dict[str, float]:
        return profile(model, sample, settings=self.settings, device=self.device, forward_fn=forward_fn)


@torch.inference_mode()
def profile_many_shapes(
    model: nn.Module,
    shapes: Iterable[Tuple[int, ...]],
    *,
    settings: Optional[ProfileSettings] = None,
    device: str = "cuda",
    forward_fn: Optional[Callable[[nn.Module, torch.Tensor], torch.Tensor]] = None,
) -> Dict[Tuple[int, ...], Dict[str, float]]:
    out: Dict[Tuple[int, ...], Dict[str, float]] = {}
    for shp in shapes:
        out[tuple(shp)] = profile(model, shp, settings=settings, device=device, forward_fn=forward_fn)
    return out
