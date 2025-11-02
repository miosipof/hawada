"""Core gating primitives for hardware-aware model optimization.

This module defines:
  - Base `Gate` interface (nn.Module) with a small, consistent API
  - Concrete gates: HeadGate, GroupGate, LayerGate
  - Straight-Through (ST) relaxed Bernoulli with Gumbel noise
  - Penalties/regularizers commonly used during training
  - Constraint projection helpers

Design goals:
  - TorchScript-friendly where possible
  - Minimal assumptions about model family (ViT, ResNet, LLM)
  - Gates operate on *groups* of units; group_size controls expansion
  - No direct knowledge of attention/FFN/etc. — adapters wire masks

Typical usage (adapter side):
  >>> gate = GroupGate(num_groups=H, group_size=Dh, tau=1.5, init_logit=3.0)
  >>> m = gate.mask(training=self.training)           # [H * Dh]
  >>> tensor = tensor * m.view(1, H, 1, Dh)           # example broadcast

Penalties scan the module tree for objects exposing `.logits` and `.tau`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _as_like(x: torch.Tensor, val) -> torch.Tensor:
    return torch.as_tensor(val, device=x.device, dtype=x.dtype)


def _gumbel_like(x: torch.Tensor) -> torch.Tensor:
    # Uniform(0,1) clamped for numerical stability
    u = torch.rand_like(x).clamp_(1e-6, 1 - 1e-6)
    return u.log().neg_() - (1 - u).log().neg_()  # log(u) - log(1-u)


# -----------------------------------------------------------------------------
# Base Gate
# -----------------------------------------------------------------------------

class Gate(nn.Module):
    """Abstract gate over *groups*.

    A gate controls `num_groups` binary decisions, typically expanded by
    `group_size` when applied to tensors. For example, gating ViT MLP hidden
    units in groups of 16: `num_groups = hidden // 16`, `group_size = 16`.

    Subclasses may override `sample_mask` for custom relaxations.
    """

    def __init__(
        self,
        num_groups: int,
        *,
        group_size: int = 1,
        tau: float = 1.5,
        init_logit: float = 3.0,
        hard_during_eval: bool = True,
    ) -> None:
        super().__init__()
        assert num_groups > 0 and group_size > 0
        self.num_groups = int(num_groups)
        self.group_size = int(group_size)
        self.tau = float(tau)
        self.hard_during_eval = bool(hard_during_eval)
        self.logits = nn.Parameter(torch.full((self.num_groups,), float(init_logit)))

    # ----- probabilities & stats ------------------------------------------------
    def probs(self) -> torch.Tensor:
        """Return per-group keep probabilities (sigmoid(logit / tau))."""
        # Using /tau here makes `tau` affect both train and eval statistics
        return torch.sigmoid(self.logits / self.tau)

    def expected_kept(self) -> torch.Tensor:
        """Expected *elements* kept (groups × group_size)."""
        return self.probs().sum() * _as_like(self.logits, self.group_size)

    # ----- masks ----------------------------------------------------------------
    def _hard_mask(self) -> torch.Tensor:
        m = (self.logits > 0).to(self.logits.dtype)
        return m.repeat_interleave(self.group_size)

    def _soft_st_mask(self) -> torch.Tensor:
        # Straight-through relaxed Bernoulli via Gumbel-sigmoid
        s = _gumbel_like(self.logits)
        y = torch.sigmoid((self.logits + s) / self.tau)
        y_hard = (y > 0.5).to(y.dtype)
        m = (y_hard - y).detach() + y
        return m.repeat_interleave(self.group_size)

    def mask(self, training: Optional[bool] = None) -> torch.Tensor:
        """Return a 1D mask of length `num_groups * group_size`.

        - Training: straight-through relaxed mask
        - Eval: hard (thresholded) mask if `hard_during_eval` else probs expanded
        """
        if training is None:
            training = self.training
        if training:
            return self._soft_st_mask()
        if self.hard_during_eval:
            return self._hard_mask()
        p = self.probs()
        return p.repeat_interleave(self.group_size)

    # ----- export helpers -------------------------------------------------------
    @torch.no_grad()
    def topk_indices(self, k: int) -> torch.Tensor:
        k = int(max(1, min(k, self.num_groups)))
        return torch.topk(self.logits, k, largest=True).indices.sort().values

    @torch.no_grad()
    def threshold_count(self) -> int:
        # Rounds to the nearest integer expectation, then clamps
        p = self.probs()
        k = int(torch.round(p.sum()).item())
        return max(1, min(k, self.num_groups))


# -----------------------------------------------------------------------------
# Concrete gates
# -----------------------------------------------------------------------------

class HeadGate(Gate):
    """Per-head gate. Often used with attention where group_size=head_dim."""

    def __init__(self, num_heads: int, *, head_dim: int = 1, **kw):
        super().__init__(num_groups=num_heads, group_size=head_dim, **kw)


class GroupGate(Gate):
    """Generic group gate (e.g., MLP hidden grouped by `group_size`)."""

    pass


class LayerGate(Gate):
    """One bit per layer (group_size=1)."""

    def __init__(self, num_layers: int, **kw):
        super().__init__(num_groups=num_layers, group_size=1, **kw)


# -----------------------------------------------------------------------------
# Penalties / Regularizers
# -----------------------------------------------------------------------------

@dataclass
class PenaltyWeights:
    """Scalars to blend regularization terms.

    Attributes
    ----------
    l0 : float
        Weight for the L0-like sparsity term (sum of keep probs).
    keep_floor_ratio : float
        Soft constraint: expected kept groups >= floor_ratio * groups.
    bimodality : float
        Encourages probabilities away from 0.5.
    """

    l0: float = 0.0
    keep_floor_ratio: float = 0.0
    bimodality: float = 0.0


def iter_gates(module: nn.Module) -> Iterable[Gate]:
    for m in module.modules():
        if isinstance(m, Gate):
            yield m
        else:
            # Duck-typing compatibility: any module with `.logits` and `.tau`
            if hasattr(m, "logits") and hasattr(m, "tau"):
                logits = getattr(m, "logits")
                if isinstance(logits, torch.Tensor) and logits.dim() == 1:
                    # Wrap view: expose basic API via adapter shim
                    g = _TensorBackedGateShim(m)
                    yield g


class _TensorBackedGateShim:
    """Lightweight adapter exposing .logits, .tau, .group_size, .num_groups.

    It is intentionally NOT an nn.Module and NOT a Gate subclass to avoid
    ctor/signature constraints and registration side-effects. It's only used
    by projection/regularization utilities that read/update .logits.
    """
    __slots__ = ("host", "logits", "tau", "group_size", "num_groups")

    def __init__(self, host):
        self.host = host
        # logits must be a Tensor/Parameter on the host
        self.logits = getattr(host, "logits")
        # default tau=1.5 if not present
        self.tau = float(getattr(host, "tau", 1.5))
        # support either group_size or group attribute names
        self.group_size = int(getattr(host, "group_size", getattr(host, "group", 1)))
        self.num_groups = int(self.logits.numel())

    def forward(self, *args, **kwargs):  # pragma: no cover - shim is not used as a layer
        raise RuntimeError("Gate shim is not a callable layer")


def l0_like_sparsity(module: nn.Module) -> torch.Tensor:
    """Sum of keep probabilities across all gates (acts like L0/L1)."""
    val = _as_like(next(module.parameters(), torch.tensor(0.0, device="cpu")), 0.0)
    out = torch.as_tensor(0.0, device=val.device, dtype=val.dtype)
    for g in iter_gates(module):
        out = out + g.probs().sum()
    return out


def keep_floor(module: nn.Module, floor_ratio: float) -> torch.Tensor:
    """Soft penalty if expected-kept falls below a fraction per gate.

    For each gate with G groups, penalize relu(floor*G - sum(p)).
    """
    if floor_ratio <= 0:
        return torch.tensor(0.0, device=next(module.parameters(), torch.tensor(0.0)).device)
    floor_ratio = float(floor_ratio)
    val = _as_like(next(module.parameters(), torch.tensor(0.0, device="cpu")), 0.0)
    out = torch.as_tensor(0.0, device=val.device, dtype=val.dtype)
    for g in iter_gates(module):
        G = _as_like(val, g.num_groups)
        floor_groups = _as_like(val, max(1.0, floor_ratio * float(g.num_groups)))
        out = out + F.relu(floor_groups - g.probs().sum())
    return out


def bimodality(module: nn.Module) -> torch.Tensor:
    """Sum over p*(1-p) to push probs away from 0.5 (minimum at 0 or 1)."""
    val = _as_like(next(module.parameters(), torch.tensor(0.0, device="cpu")), 0.0)
    out = torch.as_tensor(0.0, device=val.device, dtype=val.dtype)
    for g in iter_gates(module):
        p = g.probs()
        out = out + (p * (1.0 - p)).sum()
    return out


def combined_penalty(
    module: nn.Module,
    weights: PenaltyWeights,
) -> torch.Tensor:
    out = torch.tensor(0.0, device=next(module.parameters(), torch.tensor(0.0)).device)
    if weights.l0:
        out = out + weights.l0 * l0_like_sparsity(module)
    if weights.keep_floor_ratio:
        out = out + keep_floor(module, weights.keep_floor_ratio)
    if weights.bimodality:
        out = out + weights.bimodality * bimodality(module)
    return out


# -----------------------------------------------------------------------------
# Constraint projection
# -----------------------------------------------------------------------------

@dataclass
class Constraints:
    """High-level feasibility constraints.

    * min_keep_ratio: per-gate minimum fraction of groups to keep (soft cap via
      projection onto [min_k, G]).
    * min_groups: absolute lower bound per gate (after rounding).
    * max_groups_drop: optional ceiling on groups dropped per gate.
    """

    min_keep_ratio: float = 0.0
    min_groups: int = 1
    max_groups_drop: Optional[int] = None


@torch.no_grad()
def project_gates_into_constraints(module: nn.Module, cons: Constraints) -> None:
    """Project gate logits so that expected kept groups respect constraints.

    We rescale logits by an additive bias to achieve a target sum of probs when
    violating the lower/upper bounds. This is a light-touch projection that
    keeps relative ordering intact.
    """
    for g in iter_gates(module):
        p = torch.sigmoid(g.logits / g.tau)
        G = p.numel()
        # Lower bound
        min_keep = max(cons.min_groups, int(cons.min_keep_ratio * G))
        if p.sum().item() < min_keep:
            # Additive bias to increase sum(p)
            bias = torch.tensor(2.0, device=p.device, dtype=p.dtype)
            # Increase iteratively but cheaply
            for _ in range(6):
                p = torch.sigmoid((g.logits + bias) / g.tau)
                if p.sum().item() >= min_keep:
                    break
                bias = bias * 2
            g.logits.add_(bias)
        # Optional upper bound on drops
        if cons.max_groups_drop is not None:
            max_drop = int(cons.max_groups_drop)
            max_keep = max(1, G - max_drop)
            if p.sum().item() > max_keep:
                bias = torch.tensor(-2.0, device=p.device, dtype=p.dtype)
                for _ in range(6):
                    p = torch.sigmoid((g.logits + bias) / g.tau)
                    if p.sum().item() <= max_keep:
                        break
                    bias = bias * 2
                g.logits.add_(bias)


# -----------------------------------------------------------------------------
# Export helpers (indices from gates)
# -----------------------------------------------------------------------------

@torch.no_grad()
def topk_group_indices(g: Gate, keep_k: Optional[int] = None) -> torch.Tensor:
    """Return sorted group indices to KEEP based on logits/probs.

    If `keep_k` is None, use nearest-integer of expected kept.
    """
    if keep_k is None:
        keep_k = g.threshold_count()
    idx = torch.topk(g.logits, int(keep_k), largest=True).indices
    return idx.sort().values


@torch.no_grad()
def expand_group_indices(idx: torch.Tensor, group_size: int) -> torch.Tensor:
    """Expand group indices into element indices by `group_size` blocks."""
    if group_size == 1:
        return idx.clone()
    starts = idx * group_size
    parts = [torch.arange(s, s + group_size, device=idx.device) for s in starts]
    return torch.cat(parts, dim=0).long()


# -----------------------------------------------------------------------------
# Parameter utilities
# -----------------------------------------------------------------------------

def collect_gate_params(module: nn.Module) -> List[nn.Parameter]:
    return [g.logits for g in iter_gates(module) if isinstance(g.logits, torch.Tensor)]


def collect_param_groups(
    module: nn.Module,
    *,
    lr_gate: float = 1e-2,
    lr_linear: float = 1e-4,
    lr_affine: float = 3e-4,
    wd_linear: float = 1e-4,
) -> List[dict]:
    """Convenience grouping matching common training setups.

    Group 0: gate logits (no weight decay)
    Group 1: linear weights (with weight decay)
    Group 2: linear biases (no decay)
    Group 3: norm affine params (no decay)
    """
    gates, ln_affine, linear_w, linear_b = [], [], [], []
    for n, p in module.named_parameters():
        if not p.requires_grad:
            continue
        if n.endswith((".logits", ".head_gate", ".channel_gate")):
            gates.append(p)
            continue
        is_linear_path = (".weight" in n or ".bias" in n) and (
            ".dense" in n or ".query" in n or ".key" in n or ".value" in n or ".proj" in n
        )
        if n.endswith(".weight") and is_linear_path:
            linear_w.append(p)
        elif n.endswith(".bias") and is_linear_path:
            linear_b.append(p)
        elif "layernorm" in n.lower() or "layer_norm" in n.lower() or "LayerNorm" in n:
            ln_affine.append(p)
    return [
        {"params": gates, "lr": lr_gate, "weight_decay": 0.0},
        {"params": linear_w, "lr": lr_linear, "weight_decay": wd_linear},
        {"params": linear_b, "lr": lr_linear, "weight_decay": 0.0},
        {"params": ln_affine, "lr": lr_affine, "weight_decay": 0.0},
    ]
