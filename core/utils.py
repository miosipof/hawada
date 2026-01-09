"""Shared utilities used across core and adapters.

Consolidates helpers that are generic (device/dtype, seeding, shapes, rounding,
parameter grouping, model copying, etc.). Keep this file dependency-light.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import copy
import random

import numpy as np
import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# YAML & batch helpers
# -----------------------------------------------------------------------------

def load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _images_from_batch(batch):
    # (images, labels) or [images, labels]
    if isinstance(batch, (tuple, list)):
        return batch[0]
    # dict-style datasets
    if isinstance(batch, dict):
        for k in ("pixel_values", "images", "inputs"):
            v = batch.get(k, None)
            if torch.is_tensor(v):
                return v
        # fallback to first tensor value
        for v in batch.values():
            if torch.is_tensor(v):
                return v
        raise TypeError("Batch dict has no tensor-like image field")
    # plain tensor
    if torch.is_tensor(batch):
        return batch
    raise TypeError(f"Unsupported batch type for images: {type(batch)}")
    
# -----------------------------------------------------------------------------
# Device / dtype helpers
# -----------------------------------------------------------------------------

def as_like(x: torch.Tensor, val) -> torch.Tensor:
    """Create a scalar/tensor constant on same device/dtype as `x`."""
    return torch.as_tensor(val, device=x.device, dtype=x.dtype)


def first_param(module: nn.Module) -> torch.Tensor:
    for p in module.parameters(recurse=True):
        return p
    return torch.tensor(0.0)


def to_device_dtype(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return x.to(device=ref.device, dtype=ref.dtype)


# -----------------------------------------------------------------------------
# Seeding & determinism
# -----------------------------------------------------------------------------

def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
# Model parameter helpers
# -----------------------------------------------------------------------------

def freeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)


def unfreeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(True)


def count_parameters(module: nn.Module, *, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


# -----------------------------------------------------------------------------
# Shape/signature helpers
# -----------------------------------------------------------------------------

def input_spec_vision(sample) -> Tuple[int, int, int]:
    """Accept either a 4D tensor [B,3,H,W] or a 4-tuple (B,3,H,W). Returns (B,H,W)."""
    if isinstance(sample, torch.Tensor):
        B, C, H, W = sample.shape
        return int(B), int(H), int(W)
    if isinstance(sample, (tuple, list)) and len(sample) == 4:
        B, C, H, W = sample
        return int(B), int(H), int(W)
    raise ValueError("sample must be a tensor [B,3,H,W] or a 4-tuple (B,3,H,W)")


# -----------------------------------------------------------------------------
# Rounding / multiples
# -----------------------------------------------------------------------------

def round_down_multiple(n: int, m: int) -> int:
    if m is None or m <= 1:
        return max(1, int(n))
    n = int(n)
    return max(m, (n // m) * m)


def clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(int(v), hi))


# -----------------------------------------------------------------------------
# Slicing helpers
# -----------------------------------------------------------------------------

@torch.no_grad()
def slice_linear(mat: nn.Linear, keep_in: Optional[Sequence[int]] = None, keep_out: Optional[Sequence[int]] = None) -> nn.Linear:
    W = mat.weight.detach()
    b = mat.bias.detach() if mat.bias is not None else None
    if keep_out is not None:
        idx_out = torch.as_tensor(keep_out, device=W.device)
        W = W.index_select(0, idx_out)
        if b is not None:
            b = b.index_select(0, idx_out)
    if keep_in is not None:
        idx_in = torch.as_tensor(keep_in, device=W.device)
        W = W.index_select(1, idx_in)
    out_f, in_f = W.shape
    new = nn.Linear(in_f, out_f, bias=(b is not None)).to(W.device)
    new.weight.copy_(W)
    if b is not None:
        new.bias.copy_(b)
    return new


# -----------------------------------------------------------------------------
# Copying & detaching models
# -----------------------------------------------------------------------------

def deepcopy_eval_cpu(module: nn.Module) -> nn.Module:
    m = copy.deepcopy(module).cpu().eval()
    return m


# -----------------------------------------------------------------------------
# Gradient utilities
# -----------------------------------------------------------------------------

def zero_if_any(params: Iterable[torch.Tensor]) -> None:
    for p in params:
        if p.grad is not None:
            p.grad = None


def any_grad(params: Iterable[torch.Tensor]) -> bool:
    for p in params:
        if p.grad is not None:
            return True
    return False

# -----------------------------------------------------------------------------
# For fine-tuning
# -----------------------------------------------------------------------------

def ensure_trainable_parameters(module: nn.Module, *, requires_grad: bool = True) -> nn.Module:
    """
    Rebuild all parameters as fresh nn.Parameter tensors (detach+clone),
    which drops any 'inference tensor' tag and re-enables autograd.
    """
    for mod in module.modules():
        for name, p in list(mod._parameters.items()):
            if p is None:
                continue
            new_p = nn.Parameter(p.detach().clone(), requires_grad=requires_grad)
            setattr(mod, name, new_p)
    return module


# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------

@dataclass
class ExportRounding:
    head_floor_post: int = 1
    head_multiple_post: int = 1
    ffn_min_keep_ratio_post: float = 0.0
    ffn_snap_groups_post: int = 1


def shape_signature_vit(cfg, sample_shape: Tuple[int, int, int, int]) -> Tuple:
    B, C, H, W = sample_shape
    return (
        "ViT",
        sample_shape,
        int(getattr(cfg, "num_attention_heads", 12)),
        int(getattr(cfg, "hidden_size", 768)),
        int(getattr(cfg, "intermediate_size", 3072)),
        int(getattr(cfg, "patch_size", 16)) if not isinstance(getattr(cfg, "patch_size", 16), (tuple, list)) else tuple(getattr(cfg, "patch_size", (16, 16))),
    )
