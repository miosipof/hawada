"""Core export utilities for hard-pruning and kernel-aligned rounding.

This module is *family-agnostic*. Adapters (e.g., ViT, ResNet, LLM) should:
  1) decide which gates map to which structural dims (heads, hidden groups, channels),
  2) obtain KEEP indices using helpers in this file, and
  3) rebuild family-specific modules with the sliced weights.

Provided here:
  - Rounding policies and helpers (floors, multiples, warmup keep-all)
  - KEEP index selection from a `Gate` (or gate-like) object
  - Generic weight slicers for Linear / Conv2d / Embedding
  - Small safe-guards for dtype/device and shape checks

The library avoids touching family internals here. Exporters in adapters should
use these primitives to assemble a clean pruned model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Tuple
import torch
import torch.nn as nn

from .gates import Gate, expand_group_indices

# -----------------------------------------------------------------------------
# Policies & rounding
# -----------------------------------------------------------------------------

@dataclass
class Rounding:
    """Rounding policy for a single gated axis.

    Attributes
    ----------
    floor_groups : int
        Minimum number of groups to keep after rounding.
    multiple_groups : int
        Snap the number of groups kept down to a multiple of this (>=1).
    min_keep_ratio : float
        Optional fractional lower bound on expected keep; applied before rounding.
    """

    floor_groups: int = 1
    multiple_groups: int = 1
    min_keep_ratio: float = 0.0


@dataclass
class ExportPolicy:
    """Export-time policy shared by families.

    - `warmup_steps`: if current `step < warmup_steps`, keep-all.
    - `rounding`: default rounding used unless adapter overrides per-axis.
    """

    warmup_steps: int = 0
    rounding: Rounding = field(default_factory=Rounding)


def _round_down_mult(n: int, m: int) -> int:
    if m is None or m <= 1:
        return max(1, int(n))
    n = int(n)
    return max(m, (n // m) * m)


def _compute_keep_k(
    expected_kept: float,
    total_groups: int,
    *,
    rounding: Rounding,
) -> int:
    # Start from nearest-integer expectation
    k = int(round(expected_kept))
    # Apply ratio floor, then absolute floor, then multiple snapping
    k = max(k, int(rounding.min_keep_ratio * total_groups))
    k = max(k, int(rounding.floor_groups))
    k = min(k, total_groups)
    k = _round_down_mult(k, int(rounding.multiple_groups))
    return max(1, min(k, total_groups))


# -----------------------------------------------------------------------------
# KEEP index selection from a gate
# -----------------------------------------------------------------------------

@torch.no_grad()
def keep_group_indices_from_gate(
    gate: Gate,
    *,
    policy: ExportPolicy,
    step: Optional[int] = None,
    custom_rounding: Optional[Rounding] = None,
) -> torch.Tensor:
    """Return sorted indices of groups to KEEP based on `gate` and policy.

    If `step < warmup_steps`, returns all indices (keep-all). Otherwise, the
    number of groups to keep is computed from the *expected keep* under the
    current logits and snapped according to the rounding policy.
    """
    G = int(gate.num_groups)
    if step is not None and step < int(policy.warmup_steps):
        return torch.arange(G, device=gate.logits.device)

    if not hasattr(policy, "rounding") or custom_rounding is not None:
        rounding = custom_rounding
    else:
        rounding = policy.rounding
        
    
    p = torch.sigmoid(gate.logits.detach().float() / float(gate.tau))
    k = _compute_keep_k(expected_kept=float(p.sum()), total_groups=G, rounding=rounding)
    idx = torch.topk(p, k, largest=True).indices.sort().values
    return idx.to(torch.long)


@torch.no_grad()
def keep_element_indices_from_gate(
    gate: Gate,
    *,
    policy: ExportPolicy,
    step: Optional[int] = None,
    custom_rounding: Optional[Rounding] = None,
) -> torch.Tensor:
    """Expand kept *group* indices into element indices using `group_size`."""
    grp_idx = keep_group_indices_from_gate(gate, policy=policy, step=step, custom_rounding=custom_rounding)
    return expand_group_indices(grp_idx, gate.group_size)


# -----------------------------------------------------------------------------
# Generic slicers
# -----------------------------------------------------------------------------

@torch.no_grad()
def slice_linear(mat: nn.Linear, keep_in: Optional[Sequence[int]] = None, keep_out: Optional[Sequence[int]] = None) -> nn.Linear:
    """Create a new Linear with selected input/output features preserved.

    - `keep_out` selects rows (output features)
    - `keep_in`  selects columns (input features)
    """
    W = mat.weight.detach()
    b = mat.bias.detach() if mat.bias is not None else None

    if keep_out is not None:
        W = W.index_select(0, torch.as_tensor(keep_out, device=W.device))
        if b is not None:
            b = b.index_select(0, torch.as_tensor(keep_out, device=b.device))
    if keep_in is not None:
        W = W.index_select(1, torch.as_tensor(keep_in, device=W.device))

    out_f, in_f = W.shape
    new = nn.Linear(in_f, out_f, bias=(b is not None)).to(W.device)
    new.weight.copy_(W)
    if b is not None:
        new.bias.copy_(b)
    return new


@torch.no_grad()
def slice_conv2d(conv: nn.Conv2d, keep_in: Optional[Sequence[int]] = None, keep_out: Optional[Sequence[int]] = None) -> nn.Conv2d:
    """Create a new Conv2d with selected in/out channels preserved.

    Only supports standard conv2d (no groups/depthwise changes). For grouped
    convs, the adapter should handle group alignment before calling this.
    """
    W = conv.weight.detach()
    b = conv.bias.detach() if conv.bias is not None else None

    if keep_out is not None:
        W = W.index_select(0, torch.as_tensor(keep_out, device=W.device))
        if b is not None:
            b = b.index_select(0, torch.as_tensor(keep_out, device=b.device))
    if keep_in is not None:
        W = W.index_select(1, torch.as_tensor(keep_in, device=W.device))

    out_c, in_c = W.shape[:2]
    new = nn.Conv2d(
        in_c,
        out_c,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=1,
        bias=(b is not None),
        padding_mode=conv.padding_mode,
    ).to(W.device)
    new.weight.copy_(W)
    if b is not None:
        new.bias.copy_(b)
    return new


@torch.no_grad()
def slice_embedding(emb: nn.Embedding, keep_rows: Optional[Sequence[int]] = None, keep_dim: Optional[Sequence[int]] = None) -> nn.Embedding:
    """Create a new Embedding with selected rows (vocab) and/or dims kept."""
    W = emb.weight.detach()
    if keep_rows is not None:
        W = W.index_select(0, torch.as_tensor(keep_rows, device=W.device))
    if keep_dim is not None:
        W = W.index_select(1, torch.as_tensor(keep_dim, device=W.device))
    num, dim = W.shape
    new = nn.Embedding(num, dim, padding_idx=emb.padding_idx, max_norm=emb.max_norm, norm_type=emb.norm_type, scale_grad_by_freq=emb.scale_grad_by_freq, sparse=emb.sparse, device=W.device, dtype=W.dtype)
    new.weight.copy_(W)
    return new


# -----------------------------------------------------------------------------
# Small helpers for adapters
# -----------------------------------------------------------------------------

@torch.no_grad()
def concat_index_ranges(ranges: Sequence[Tuple[int, int]]) -> torch.Tensor:
    """Given [(start, end_exclusive), ...], return concatenated 1D indices."""
    parts = [torch.arange(a, b, dtype=torch.long) for a, b in ranges if b > a]
    return torch.cat(parts, dim=0) if parts else torch.empty(0, dtype=torch.long)


@torch.no_grad()
def block_indices_from_groups(groups: Sequence[int], group_size: int) -> torch.Tensor:
    """Convert sorted group ids to expanded feature indices."""
    groups = torch.as_tensor(groups, dtype=torch.long)
    return expand_group_indices(groups, int(group_size))
