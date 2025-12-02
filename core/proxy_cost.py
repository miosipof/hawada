# core/proxy_cost.py
"""Latency proxy models and a tiny LUT for hardware correction.

This file defines a family-agnostic interface plus concrete proxies (ViT, ResNet, LLM)
that estimate latency from *soft structure* (gates) and input size. All proxies accept
the trainer's `(model, batch) -> ms` call signature directly (batches may be dict/tuple/tensor).
A small, in-memory LUT can be populated from real measurements during training to correct
analytic estimates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union, List

import torch
import torch.nn as nn

from .gates import iter_gates, _as_like


# -----------------------------------------------------------------------------
# Small batch helpers (shared)
# -----------------------------------------------------------------------------

TensorOrBatch = Union[torch.Tensor, Tuple, List, Dict[str, Any]]

def _first_tensor(batch: TensorOrBatch) -> torch.Tensor:
    """Find the first tensor inside a batch-like structure."""
    if torch.is_tensor(batch):
        return batch
    if isinstance(batch, dict):
        # Common keys across tasks
        for k in ("input_ids", "pixel_values", "images", "x"):
            v = batch.get(k, None)
            if torch.is_tensor(v):
                return v
        # fallback: first tensor value
        for v in batch.values():
            if torch.is_tensor(v):
                return v
        raise ValueError("Batch dict has no tensor field I recognize.")
    if isinstance(batch, (list, tuple)):
        for v in batch:
            if torch.is_tensor(v):
                return v
        # torchvision pattern: ([aug1, aug2], label)
        if len(batch) and isinstance(batch[0], (list, tuple)):
            for v in batch[0]:
                if torch.is_tensor(v):
                    return v
    raise ValueError("Cannot find a tensor in the provided batch.")

def _ids_from_batch(batch: TensorOrBatch) -> torch.Tensor:
    """Return a 2D [B,S] tensor representing token ids for LLMs."""
    if isinstance(batch, dict) and "input_ids" in batch and torch.is_tensor(batch["input_ids"]):
        return batch["input_ids"]
    t = _first_tensor(batch)
    if t.dim() >= 2:
        return t
    raise ValueError("Cannot infer [B,S] from batch; need 'input_ids' or a 2D tensor.")

def _nchw_from_batch(batch: TensorOrBatch) -> Tuple[int, int, int, int]:
    """Return NCHW shape from a batch or an explicit (N,C,H,W) tuple/list/tensor."""
    if isinstance(batch, (tuple, list)) and len(batch) == 4 and all(isinstance(x, int) for x in batch):
        return tuple(batch)  # type: ignore[return-value]
    x = _first_tensor(batch)
    if x.dim() != 4:
        raise ValueError(f"Expected NCHW tensor for CNN proxy; got tensor with shape {tuple(x.shape)}")
    N, C, H, W = map(int, x.shape)
    return (N, C, H, W)


# -----------------------------------------------------------------------------
# Base proxy + LUT
# -----------------------------------------------------------------------------

class LatencyProxy(nn.Module):
    """Abstract proxy producing a scalar latency-like value (ms).

    Subclasses implement `_predict_raw` and may define `_signature` keys used by
    a LUT to refine estimates with real measurements. Proxies accept either a
    batch-like object (dict/tuple/tensor) or an explicit shape tuple.
    """

    def __init__(self):
        super().__init__()

    def predict(
        self,
        model: nn.Module,
        sample: TensorOrBatch,
        *,
        policy=None,
        step: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Batch-friendly entry point. `sample` may be a batch or explicit shape."""
        return self._predict_raw(model, sample, policy=policy, step=step, **kwargs)

    def _predict_raw(
        self,
        model: nn.Module,
        sample: TensorOrBatch,
        *,
        policy=None,
        step: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def signature(
        self,
        model: nn.Module,
        sample: TensorOrBatch,
        *,
        policy=None,
        step: Optional[int] = None
    ) -> Tuple:
        """Return a hashable signature describing the workload shape."""
        if torch.is_tensor(sample):
            shp = tuple(sample.shape)
        elif isinstance(sample, (tuple, list)):
            shp = tuple(sample)
        elif isinstance(sample, dict):
            # summarize the shapes of any tensors in dict
            shp = tuple((k, tuple(v.shape)) for k, v in sample.items() if torch.is_tensor(v))
        else:
            shp = (str(type(sample)),)
        return (type(self).__name__, shp)


class LatencyLUT:
    """Tiny LUT mapping `(signature) -> measured_ms`."""

    def __init__(self):
        self._table: Dict[Tuple[Any, ...], float] = {}

    def update(self, signature: Tuple[Any, ...], measured_ms: float) -> None:
        self._table[signature] = float(measured_ms)

    def get(self, signature: Tuple[Any, ...]) -> Optional[float]:
        return self._table.get(signature)

    def blend(self, raw_estimate: torch.Tensor, signature: Tuple[Any, ...]) -> torch.Tensor:
        val = self.get(signature)
        if val is None:
            return raw_estimate
        # Put on same device/dtype as raw_estimate
        return _as_like(raw_estimate, val)