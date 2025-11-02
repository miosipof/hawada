"""Latency proxy models and a tiny LUT for hardware correction.

This file defines a family-agnostic interface plus a concrete ViT proxy that
estimates latency from *soft structure* (gates) and input size. It supports a
small, in-memory LUT that can be populated from real measurements during
training to correct analytic estimates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .gates import iter_gates, _as_like


# -----------------------------------------------------------------------------
# Base proxy + LUT
# -----------------------------------------------------------------------------

class LatencyProxy(nn.Module):
    """Abstract proxy producing a scalar latency-like value (ms).

    Subclasses implement `_predict_raw` and may define `_signature` keys used by
    a LUT to refine estimates with real measurements.
    """

    def __init__(self):
        super().__init__()

    def predict(self, model: nn.Module, sample: torch.Tensor | Tuple[int, ...], *, policy=None, step: Optional[int] = None) -> torch.Tensor:
        return self._predict_raw(model, sample, policy=policy, step=step)

    def _predict_raw(self, model: nn.Module, sample: torch.Tensor | Tuple[int, ...], *, policy=None, step: Optional[int] = None) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def signature(self, model: nn.Module, sample: torch.Tensor | Tuple[int, ...], *, policy=None, step: Optional[int] = None) -> Tuple:
        """Return a hashable signature describing the workload shape.

        Used by `LatencyLUT` to cache/lookup corrections. Subclasses should
        override for better fidelity.
        """
        if isinstance(sample, torch.Tensor):
            shp = tuple(sample.shape)
        else:
            shp = tuple(sample)
        return (type(self).__name__, shp)


class LatencyLUT:
    """Tiny LUT mapping `(signature) -> measured_ms`.

    Use `blend(raw_estimate, signature)` to return `measured_ms` when available
    or fall back to `raw_estimate`.
    """

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


# -----------------------------------------------------------------------------
# ViT proxy (analytic + gates), with scale and per-term weights
# -----------------------------------------------------------------------------

@dataclass
class ViTProxyConfig:
    scale_ms: float = 1.0
    alpha_qkv: float = 1.0
    alpha_scores: float = 1.0
    alpha_out: float = 1.0
    alpha_mlp: float = 1.0


class ViTLatencyProxy(LatencyProxy):
    """Latency proxy for ViT models.

    Cost per block (up to a global scale):
      qkv:        3 * S * D * D_kept
      scores:     S^2 * heads_kept * d_head
      out_proj:   S * D_kept * D
      mlp:        2 * S * D * hidden_kept

    `hidden_kept` and `heads_kept` are *soft expectations* derived from gates
    if present; otherwise defaults from config are used.
    """

    def __init__(self, cfg: Optional[ViTProxyConfig] = None, lut: Optional[LatencyLUT] = None):
        super().__init__()
        self.cfg = cfg or ViTProxyConfig()
        self.lut = lut or LatencyLUT()

    # ---- helpers -------------------------------------------------------------
    @staticmethod
    def _input_spec(sample) -> Tuple[int, int, int]:
        if isinstance(sample, torch.Tensor):
            B, C, H, W = sample.shape
            return int(B), int(H), int(W)
        if isinstance(sample, (tuple, list)) and len(sample) == 4:
            B, C, H, W = sample
            return int(B), int(H), int(W)
        raise ValueError("sample must be a tensor [B,3,H,W] or a 4-tuple (B,3,H,W)")

    @staticmethod
    def _patch_hw(cfg) -> Tuple[int, int]:
        patch = getattr(cfg, "patch_size", 16)
        if isinstance(patch, (tuple, list)):
            return int(patch[0]), int(patch[1])
        return int(patch), int(patch)

    @staticmethod
    def _soft_heads_from_block(blk) -> Optional[torch.Tensor]:
        # Prefer a nested attention with kept_heads_soft()
        attn = getattr(getattr(blk, "attention", None), "attention", None)
        if attn is not None and hasattr(attn, "kept_heads_soft"):
            return attn.kept_heads_soft()
        return None

    @staticmethod
    def _find_ffn_gate(blk):
        inter = getattr(blk, "intermediate", None)
        if inter is None:
            return None
        # Common attribute names
        for nm in ("neuron_gate", "gate", "ffn_gate"):
            g = getattr(inter, nm, None)
            if g is not None and hasattr(g, "logits") and hasattr(g, "tau"):
                return g
        # Last resort: scan children
        for m in blk.modules():
            if hasattr(m, "logits") and hasattr(m, "tau"):
                return m
        return None

    # ---- proxy ---------------------------------------------------------------
    def _predict_raw(self, model: nn.Module, sample: torch.Tensor | Tuple[int, ...], *, policy=None, step: Optional[int] = None) -> torch.Tensor:
        anchor = next((p for p in model.parameters()), torch.tensor(0.0))
        device = anchor.device
        dtype = anchor.dtype

        B, H_img, W_img = self._input_spec(sample)
        cfg = getattr(model, "config", None)
        if cfg is None:
            raise ValueError("Model must expose a HuggingFace-like .config for ViT proxy")
        ph, pw = self._patch_hw(cfg)

        S = _as_like(anchor, 1 + (H_img // ph) * (W_img // pw))
        D = _as_like(anchor, int(getattr(cfg, "hidden_size", 768)))
        Hh = _as_like(anchor, int(getattr(cfg, "num_attention_heads", 12)))
        Dh = D // Hh

        warm = False
        if policy is not None and step is not None:
            warm = (step < int(getattr(policy, "warmup_steps", 0)))

        total_qkv = _as_like(anchor, 0.0)
        total_scores = _as_like(anchor, 0.0)
        total_out = _as_like(anchor, 0.0)
        total_mlp = _as_like(anchor, 0.0)

        default_hidden = _as_like(anchor, int(getattr(cfg, "intermediate_size", 4 * int(D))))

        for blk in model.encoder.layer:
            if warm:
                heads_soft = Hh
            else:
                heads_soft = self._soft_heads_from_block(blk) or Hh

            # FFN hidden expectation
            if warm:
                hidden_soft = default_hidden
            else:
                g = self._find_ffn_gate(blk)
                if g is None:
                    hidden_soft = default_hidden
                else:
                    probs = torch.sigmoid(g.logits / g.tau)
                    group = int(getattr(g, "group", getattr(g, "group_size", 16)))
                    hidden_soft = probs.sum() * _as_like(anchor, group)

            D_kept = heads_soft * Dh

            total_qkv += 3 * S * D * D_kept
            total_scores += (S * S) * heads_soft * Dh
            total_out += S * D_kept * D
            total_mlp += 2 * S * D * hidden_soft

        raw = (
            self.cfg.alpha_qkv * total_qkv
            + self.cfg.alpha_scores * total_scores
            + self.cfg.alpha_out * total_out
            + self.cfg.alpha_mlp * total_mlp
        )
        raw_ms = raw * _as_like(anchor, float(self.cfg.scale_ms))

        # optional LUT correction
        sig = self.signature(model, sample, policy=policy, step=step)
        return self.lut.blend(raw_ms, sig)

    # A reasonable default signature for ViT workloads
    def signature(self, model: nn.Module, sample, *, policy=None, step: Optional[int] = None) -> Tuple:
        if isinstance(sample, torch.Tensor):
            shp = tuple(sample.shape)
        else:
            shp = tuple(sample)
        cfg = getattr(model, "config", None)
        heads = int(getattr(cfg, "num_attention_heads", 12))
        hidden = int(getattr(cfg, "hidden_size", 768))
        inter = int(getattr(cfg, "intermediate_size", 3072))
        return ("ViT", shp, heads, hidden, inter)


# -----------------------------------------------------------------------------
# Calibration helpers
# -----------------------------------------------------------------------------

@torch.inference_mode()
def calibrate_scale(proxy: ViTLatencyProxy, model: nn.Module, sample: torch.Tensor | Tuple[int, ...], measure_fn, *, device: str = "cuda") -> float:
    """Set proxy scale so that keep-all student matches measured ms.

    `measure_fn(model, shape_or_tensor)` should return `(mean_ms, p95_ms)`.
    """
    if isinstance(sample, torch.Tensor):
        sample_t = sample
        shape = tuple(sample.shape)
    else:
        shape = tuple(sample)
        sample_t = torch.randn(*shape, device=device)

    model = model.to(device).eval()
    # infer keep-all latency using the passed measure function
    mean_ms, _ = measure_fn(model, shape, device=device)
    soft_ms = proxy.predict(model, sample_t).item()
    proxy.cfg.scale_ms = float(mean_ms / max(soft_ms, 1e-9))
    return proxy.cfg.scale_ms



# ------------------------------ ResNet Proxy ------------------------------

@dataclass
class ResNetProxyConfig:
    scale_ms: float = 1.0
    alpha_conv: float = 1.0   # weight for conv FLOPs term


def _as_const_like_resnet(x_like: torch.Tensor, val):
    return torch.as_tensor(val, device=x_like.device, dtype=x_like.dtype)


def _find_anchor_param(model: nn.Module) -> torch.Tensor:
    # Prefer any gate-like parameter; otherwise any parameter; else cpu scalar
    for m in model.modules():
        for nm in ("logits", "head_gate"):
            t = getattr(m, nm, None)
            if isinstance(t, torch.Tensor):
                return t
    for p in model.parameters():
        return p
    return torch.tensor(0.0)


def _kept_from_gate(module, anchor: torch.Tensor) -> Optional[torch.Tensor]:
    """Return expected kept channels for a BN gate: probs.sum() * group_size.
    If no gate is found, return None.
    """
    # Look for common gate attribute names on the BN wrapper or sibling
    g = None
    for nm in ("gate", "neuron_gate", "channel_gate", "bn_gate"):
        if hasattr(module, nm):
            g = getattr(module, nm)
            break
    # Also allow the module itself to have logits/tau/group_size (e.g., custom wrapper)
    if g is None and hasattr(module, "logits") and hasattr(module, "tau"):
        g = module

    if g is None or not hasattr(g, "logits"):
        return None
    logits = g.logits
    tau = float(getattr(g, "tau", 1.5))
    group = int(getattr(g, "group", getattr(g, "group_size", 1)))
    if group <= 0:
        group = 1
    probs = torch.sigmoid(logits / tau)
    return probs.sum() * _as_const_like_resnet(anchor, group)


class ResNetLatencyProxy:
    """Latency proxy for ResNet-like backbones with BN gates.

    Approximates latency with a FLOPs-style sum over convs, using the *expected*
    kept channels after each BN gate (probs.sum()*group_size). Falls back to the
    full channel count when a gate is not found.

    Assumptions:
    - Model has attributes: conv1, bn1, layer1..layer4 (torchvision-style).
    - Each BasicBlock has conv1,bn1,conv2,bn2, and optional downsample of (conv,bn).
    - Stride/padding are taken from the conv modules to update H,W.
    """

    def __init__(self, cfg: Optional[ResNetProxyConfig] = None):
        self.cfg = cfg or ResNetProxyConfig()

    def _add_cost(self, cost_like: torch.Tensor, oc, ic, k, stride, H, W):
        alpha = _as_const_like_resnet(cost_like, self.cfg.alpha_conv)
        # update spatial dims with conv stride (roughly, ignoring padding effects)
        H = (H + stride - 1) // stride
        W = (W + stride - 1) // stride
        flops = _as_const_like_resnet(cost_like, oc) * _as_const_like_resnet(cost_like, ic) * (k * k) * _as_const_like_resnet(cost_like, H) * _as_const_like_resnet(cost_like, W)
        return cost_like + alpha * flops, H, W

    def predict(self, model: nn.Module, sample_shape: Tuple[int, int, int, int]):
        # sample_shape: (B,C,H,W)
        _, C_in, H0, W0 = sample_shape
        anchor = _find_anchor_param(model)
        cost = _as_const_like_resnet(anchor, 0.0)
        H = _as_const_like_resnet(anchor, int(H0))
        W = _as_const_like_resnet(anchor, int(W0))

        # Stem
        conv1 = getattr(model, "conv1")
        bn1 = getattr(model, "bn1", None)
        k = conv1.kernel_size[0]
        s = conv1.stride[0]
        kept_out = None
        if bn1 is not None:
            kept = _kept_from_gate(bn1, anchor)
            if kept is not None:
                kept_out = kept
        oc_eff = kept_out if kept_out is not None else _as_const_like_resnet(anchor, conv1.out_channels)
        cost, H, W = self._add_cost(cost, oc_eff, _as_const_like_resnet(anchor, C_in), k, s, H, W)
        in_ch = oc_eff

        def _block_cost(block, in_ch, H, W, cost):
            # conv1 -> bn1
            c1 = block.conv1
            b1 = block.bn1 if hasattr(block, "bn1") else None
            k1, s1 = c1.kernel_size[0], c1.stride[0]
            oc1_eff = _kept_from_gate(b1, anchor) or _as_const_like_resnet(anchor, c1.out_channels)
            cost, H, W = self._add_cost(cost, oc1_eff, in_ch, k1, s1, H, W)

            # conv2 -> bn2
            c2 = block.conv2
            b2 = block.bn2 if hasattr(block, "bn2") else None
            k2, s2 = c2.kernel_size[0], c2.stride[0]
            oc2_eff = _kept_from_gate(b2, anchor) or _as_const_like_resnet(anchor, c2.out_channels)
            cost, H, W = self._add_cost(cost, oc2_eff, oc1_eff, k2, s2, H, W)

            # downsample path affects the residual width; use bn2 kept as next in_ch
            return oc2_eff, H, W, cost

        # Layers
        for lname in ("layer1", "layer2", "layer3", "layer4"):
            layer = getattr(model, lname, None)
            if layer is None:
                continue
            for blk in layer:
                in_ch, H, W, cost = _block_cost(blk, in_ch, H, W, cost)

        scale = _as_const_like_resnet(anchor, self.cfg.scale_ms)
        return cost * scale

    @torch.no_grad()
    def calibrate(self, model: nn.Module, keepall_export_fn, profiler_fn, sample_shape: Tuple[int, int, int, int], device: str = "cuda") -> float:
        """Calibrate `scale_ms` so proxy(model_keepall) ~= real latency in ms."""
        keep = keepall_export_fn(model)
        mean_ms, _ = profiler_fn(keep, sample_shape, device=device)
        soft = float(self.predict(model, sample_shape).detach().cpu())
        self.cfg.scale_ms = mean_ms / max(soft, 1e-9)
        return mean_ms

