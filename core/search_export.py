"""Export-parameter search (hardware-aware).

This module performs a small grid search over export rounding/multiple knobs and
picks the configuration that minimizes *measured* latency for the target batch
shape. It is family-agnostic; adapters provide the export function.

For ViT, see `vit_search_best_export` which scans per-head multiples and FFN
snap group sizes, mirroring kernel-friendly widths.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import copy
import itertools

import torch
import torch.nn as nn

from .export import ExportPolicy as CoreExportPolicy, Rounding as CoreRounding
from .profiler import measure_latency_ms, ProfileSettings


# Type alias: adapter export function
ExportFn = Callable[[nn.Module, object, int], nn.Module]


@dataclass
class SearchResult:
    best_model: nn.Module
    best_params: dict
    trials: List[dict]


def grid_search_latency(
    model_with_gates: nn.Module,
    export_fn: ExportFn,
    *,
    head_multiples: Sequence[int],
    ffn_snaps: Sequence[int],
    step: int,
    batch_shape: Tuple[int, int, int, int],  # (B,C,H,W)
    measure_settings: Optional[ProfileSettings] = None,
    device: str = "cuda",
    make_policy: Optional[Callable[[int, int], object]] = None,
) -> SearchResult:
    """Generic grid search over (head_multiple, ffn_snap_groups).

    - `make_policy(h_mult, ffn_snap)` must return an adapter-acceptable export policy.
      If not provided, falls back to a single-rounding `CoreExportPolicy` using
      `multiple_groups=head_multiple` for both heads and FFN.
    """
    trials: List[dict] = []
    best = None

    to_try = itertools.product(head_multiples, ffn_snaps)
    for i, (hm, fs) in enumerate(to_try):
        policy = make_policy(hm, fs) if make_policy is not None else CoreExportPolicy(
            warmup_steps=0,
            rounding=CoreRounding(floor_groups=1, multiple_groups=int(hm), min_keep_ratio=0.0),
        )
        slim = export_fn(model_with_gates, policy, step)
        mean_ms, p95_ms, _ = measure_latency_ms(slim, batch_shape, settings=measure_settings, device=device)
        rec = {"head_multiple": int(hm), "ffn_snap": int(fs), "mean_ms": float(mean_ms), "p95_ms": float(p95_ms)}
        print(f"[{i}/{len(list(to_try))}] head_multiple {int(hm)} | ffn_snap {int(fs)} | mean_ms = {float(mean_ms)}")
        trials.append(rec)
        if best is None or mean_ms < best[0]:
            best = (mean_ms, hm, fs, slim)

    assert best is not None
    _, hm_best, fs_best, slim_best = best
    return SearchResult(best_model=slim_best, best_params={"head_multiple": int(hm_best), "ffn_snap": int(fs_best)}, trials=trials)



