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

from .gates import iter_gates, _as_like  # _as_like is used by ViT proxy


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

def _vit_layers(m):
    enc = getattr(m, "encoder", None)
    if enc is not None and hasattr(enc, "layer"):
        return enc.layer
    vit = getattr(m, "vit", None)
    if vit is not None and hasattr(vit, "encoder") and hasattr(vit.encoder, "layer"):
        return vit.encoder.layer
    raise TypeError("Expected a HF ViT with *.encoder.layer (ViTModel or ViTForImageClassification).")




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
    g = None
    for nm in ("gate", "neuron_gate", "channel_gate", "bn_gate"):
        if hasattr(module, nm):
            g = getattr(module, nm)
            break
    if g is None and hasattr(module, "logits") and hasattr(module, "tau"):
        g = module

    if g is None or not hasattr(g, "logits"):
        return None
    logits = g.logits
    tau = float(getattr(g, "tau", 1.5))
    group = int(getattr(g, "group", getattr(g, "group_size", 1)))
    if group <= 0: group = 1
    probs = torch.sigmoid(logits / tau)
    return probs.sum() * _as_const_like_resnet(anchor, group)


class ResNetLatencyProxy(LatencyProxy):
    """Latency proxy for ResNet-like backbones with BN gates.

    Approximates latency with a FLOPs-style sum over convs, using the *expected*
    kept channels after each BN gate (probs.sum()*group_size). Falls back to the
    full channel count when a gate is not found.

    Accepts a batch or an explicit (N,C,H,W) shape.
    """

    def __init__(self, cfg: Optional[ResNetProxyConfig] = None):
        super().__init__()
        self.cfg = cfg or ResNetProxyConfig()

    def _add_cost(self, cost_like: torch.Tensor, oc, ic, k, stride, H, W):
        alpha = _as_const_like_resnet(cost_like, self.cfg.alpha_conv)
        # update spatial dims with conv stride (roughly, ignoring padding effects)
        H = (H + stride - 1) // stride
        W = (W + stride - 1) // stride
        flops = _as_const_like_resnet(cost_like, oc) * _as_const_like_resnet(cost_like, ic) * (k * k) * _as_const_like_resnet(cost_like, H) * _as_const_like_resnet(cost_like, W)
        return cost_like + alpha * flops, H, W

    def _predict_raw(self, model: nn.Module, sample: TensorOrBatch, **_) -> torch.Tensor:
        N, C_in, H0, W0 = _nchw_from_batch(sample)
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
    def calibrate(self, model: nn.Module, keepall_export_fn, profiler_fn, sample: TensorOrBatch, device: str = "cuda") -> float:
        """Calibrate `scale_ms` so proxy(model_keepall) ~= real latency in ms."""
        keep = keepall_export_fn(model)
        sample_shape = _nchw_from_batch(sample)
        mean_ms, _ = profiler_fn(keep, sample_shape, device=device)
        soft = float(self.predict(model, sample).detach().cpu())
        self.cfg.scale_ms = mean_ms / max(soft, 1e-9)
        return mean_ms


# -----------------------------------------------------------------------------
# LLM proxy
# -----------------------------------------------------------------------------

"""
LatencyProxyLLM
---------------
A lightweight latency proxy for decoder-only HF LLMs (LLaMA/Mistral style).

- Estimates end-to-end latency (ms-like scalar) for a given (B, S, T):
    * Prefill on S tokens (build KV cache)
    * Cached decode for T steps
- Uses soft gate expectations:
    * Attention heads (HeadGate on GatedSelfAttentionLLM)
    * FFN hidden (SwiGLUWidthGate via .mlp.neuron_gate)
- Calibrate .scale_ms so proxy ≈ real latency of a keep-all model.

Public API
----------
- LatencyProxyLLM(...).predict(model, batch_or_shape)     # trainer entry
- LatencyProxyLLM(...).predict(model, B=?, S=?, T=?)      # explicit entry
- LatencyProxyLLM(...).debug_layer_view(...)
- calibrate_proxy_llm(...), calibrate_proxy_llm_from_batch(...)
"""

# ------------------------------------------------------------
# Shared tiny utils (device/dtype-safe constants)
# ------------------------------------------------------------
def _find_gate_param_or_fallback(model: nn.Module) -> torch.Tensor:
    """
    Return a tensor to anchor device/dtype for proxy constants.
    Prefer gate logits; else any parameter; else CPU fp32 scalar.
    """
    for m in model.modules():
        if hasattr(m, "head_gate") and hasattr(getattr(m, "head_gate"), "logits"):
            return m.head_gate.logits
        if hasattr(m, "neuron_gate") and hasattr(m.neuron_gate, "logits"):
            return m.neuron_gate.logits
        if hasattr(m, "logits") and isinstance(getattr(m, "logits"), torch.Tensor):
            return m.logits
    for p in model.parameters():
        return p
    return torch.tensor(0.0)

def _as_const_like(x_like: torch.Tensor, val):
    return torch.as_tensor(val, device=x_like.device, dtype=x_like.dtype)


# ------------------------------------------------------------
# Proxy
# ------------------------------------------------------------
@dataclass
class _WarmupOnlyPolicy:
    """Tiny policy shim so you can pass warmup_steps to .predict()."""
    warmup_steps: int = 0

class LatencyProxyLLM(LatencyProxy):
    """
    LLM latency proxy (ms ~ weighted FLOPs/bandwidth terms) for prefill + cached decode.
    Accepts either a batch or explicit B,S,T.
    """

    def __init__(
        self,
        *,
        scale_ms: float = 1.0,
        alpha_qkv: float = 1.0,
        alpha_scores: float = 1.0,
        alpha_out: float = 1.0,
        alpha_mlp: float = 1.0,
        gate_kv_in_proxy: bool = False,
        default_T: int = 128,
    ):
        super().__init__()
        self.scale_ms = float(scale_ms)
        self.alpha_qkv = float(alpha_qkv)
        self.alpha_scores = float(alpha_scores)
        self.alpha_out = float(alpha_out)
        self.alpha_mlp = float(alpha_mlp)
        self.gate_kv_in_proxy = bool(gate_kv_in_proxy)
        self.default_T = int(default_T)

    # ---------- gate discovery ----------
    @staticmethod
    def _soft_heads_from_block_llm(blk) -> Optional[torch.Tensor]:
        attn = getattr(blk, "self_attn", None)
        if attn is None:
            return None
        if hasattr(attn, "kept_heads_soft") and callable(attn.kept_heads_soft):
            return attn.kept_heads_soft()
        logits, tau = None, None
        if hasattr(attn, "head_gate") and hasattr(attn.head_gate, "logits"):
            logits = attn.head_gate.logits
            tau = float(getattr(attn.head_gate, "tau", getattr(attn, "tau", 1.5)))
        elif hasattr(attn, "logits"):
            logits = attn.logits
            tau = float(getattr(attn, "tau", 1.5))
        if logits is None:
            return None
        return torch.sigmoid(logits / tau).sum()

    @staticmethod
    def _find_ffn_gate_llm(blk):
        mlp = getattr(blk, "mlp", None)
        g = getattr(mlp, "neuron_gate", None) if mlp is not None else None
        if g is not None and hasattr(g, "logits") and hasattr(g, "tau"):
            return g
        return None

    def _soft_hidden_from_block_llm(self, blk, default_hidden, anchor, warm=False):
        if warm:
            return default_hidden
        g = self._find_ffn_gate_llm(blk)
        if g is None:
            return default_hidden
        probs = torch.sigmoid(g.logits / float(g.tau))  # [#groups]
        group = int(getattr(g, "group", getattr(g, "group_size", 128)))
        kept_hidden = probs.sum() * _as_const_like(anchor, group)
        return kept_hidden

    # ---------- main ----------
    def predict(  # trainer entry and explicit-shape entry unified
        self,
        model: nn.Module,
        sample: Optional[TensorOrBatch] = None,
        *,
        B: Optional[int] = None,
        S: Optional[int] = None,
        T: Optional[int] = None,
        policy: Optional[object] = None,
        step: Optional[int] = None,
        return_terms: bool = False,
    ):
        # Allow explicit B,S,(T) path
        if B is not None and S is not None:
            ids_B, ids_S = int(B), int(S)
            ids_T = int(T) if T is not None else int(self.default_T)
        else:
            if sample is None:
                raise ValueError("LatencyProxyLLM.predict needs either a batch sample or explicit B,S.")
            if isinstance(sample, (tuple, list)) and len(sample) in (2, 3) and all(isinstance(x, int) for x in sample):
                # explicit (B,S) or (B,S,T)
                ids_B, ids_S = int(sample[0]), int(sample[1])
                ids_T = int(sample[2]) if len(sample) == 3 else int(self.default_T)
            else:
                ids = _ids_from_batch(sample)
                ids_B, ids_S = int(ids.size(0)), int(ids.size(1))
                ids_T = int(self.default_T) if T is None else int(T)

        anchor = _find_gate_param_or_fallback(model)

        # scalar tensors (same device/dtype)
        B_t = _as_const_like(anchor, ids_B)
        S_t = _as_const_like(anchor, ids_S)
        T_t = _as_const_like(anchor, ids_T)

        cfg = model.config
        D  = _as_const_like(anchor, int(cfg.hidden_size))
        Hh = _as_const_like(anchor, int(cfg.num_attention_heads))
        Hkv = _as_const_like(anchor, int(getattr(cfg, "num_key_value_heads", int(Hh))))
        Dh = D // Hh

        warmup_steps = int(getattr(policy, "warmup_steps", 0)) if policy is not None else 0
        warm = bool(step is not None and step < warmup_steps)

        total_qkv = anchor.new_zeros(())
        total_scores = anchor.new_zeros(())
        total_out = anchor.new_zeros(())
        total_mlp = anchor.new_zeros(())

        default_hidden = _as_const_like(anchor, int(getattr(cfg, "intermediate_size", 4 * int(D))))

        layers = getattr(getattr(model, "model", model), "layers", [])
        for blk in layers:
            heads_soft = Hh if warm else (self._soft_heads_from_block_llm(blk) or Hh)
            Dq = heads_soft * Dh
            # K/V effective width
            if self.gate_kv_in_proxy:
                Dkv = heads_soft * Dh
            else:
                Dkv = Hkv * Dh
            hidden_soft = self._soft_hidden_from_block_llm(blk, default_hidden, anchor, warm=warm)

            # Prefill + decode (simplified aggregation)
            Seff = S_t + T_t

            # q/k/v linear FLOP-like terms
            total_qkv = total_qkv + (
                # q
                B_t * Seff * D * Dq +
                # k + v
                2 * B_t * Seff * D * Dkv
            )
            # attention scores (prefill SxS + decode triangular)
            total_scores = total_scores + (
                B_t * (S_t * S_t) * heads_soft * Dh +
                B_t * heads_soft * Dh * (T_t * S_t + (T_t * (T_t + 1)) // 2)
            )
            # out proj
            total_out = total_out + B_t * Seff * Dq * D
            # mlp
            total_mlp = total_mlp + B_t * Seff * 2 * D * hidden_soft

        flops_like = (
            self.alpha_qkv * total_qkv
            + self.alpha_scores * total_scores
            + self.alpha_out * total_out
            + self.alpha_mlp * total_mlp
        )

        ms = flops_like * _as_const_like(anchor, self.scale_ms)
        if return_terms:
            return ms, {
                "qkv": float((self.alpha_qkv * total_qkv).detach().cpu()),
                "scores": float((self.alpha_scores * total_scores).detach().cpu()),
                "out": float((self.alpha_out * total_out).detach().cpu()),
                "mlp": float((self.alpha_mlp * total_mlp).detach().cpu()),
            }
        return ms

    # ---------- per-layer debug ----------
    @torch.no_grad()
    def debug_layer_view(
        self,
        model: nn.Module,
        *,
        B: int,
        S: int,
        T: int,
        policy: Optional[object] = None,
        step: Optional[int] = None,
    ) -> list:
        anchor = _find_gate_param_or_fallback(model)
        cfg = getattr(model, "config", None)
        D   = _as_const_like(anchor, int(getattr(cfg, "hidden_size", 0)))
        Hq  = _as_const_like(anchor, int(getattr(cfg, "num_attention_heads", 0)))
        Hkv = _as_const_like(anchor, int(getattr(cfg, "num_key_value_heads", int(Hq))))
        Dh  = D // Hq

        warm = False
        if policy is not None and step is not None:
            warm = (int(step) < int(getattr(policy, "warmup_steps", 0)))

        rows = []
        layers = getattr(getattr(model, "model", model), "layers", None) or []
        for i, blk in enumerate(layers):
            heads_soft = Hq if warm else (self._soft_heads_from_block_llm(blk) or Hq)
            Dq = heads_soft * Dh
            Dkv = (heads_soft * Dh) if self.gate_kv_in_proxy else (Hkv * Dh)
            hidden_soft = self._soft_hidden_from_block_llm(
                blk, _as_const_like(anchor, int(getattr(cfg, "intermediate_size", 4 * int(D)))), anchor, warm=warm
            )
            rows.append({
                "layer": i,
                "heads_soft": float(heads_soft.detach().cpu()),
                "Dq≈heads*Dh": float(Dq.detach().cpu()),
                "Dkv_used": float(Dkv.detach().cpu()),
                "ffn_hidden_soft": float(hidden_soft.detach().cpu()),
            })
        return rows


# ------------------------------------------------------------
# Calibration helpers for LLM
# ------------------------------------------------------------
@torch.inference_mode()
def calibrate_proxy_llm(
    proxy: LatencyProxyLLM,
    model: nn.Module,
    *,
    B: int,
    S: int,
    T: int,
    export_keepall_fn,
    device: str = "cuda",
    warmup: int = 10,
    iters: int = 30,
) -> float:
    """
    Calibrate proxy.scale_ms so proxy.predict(...) matches real keep-all latency for (B,S,T).
    Returns the measured real mean latency in ms.
    """
    keepall = export_keepall_fn(model).to(device).eval()

    # Measure real latency (prefill + decode)
    from core.measure import measure_latency_text_ms as _measure  # adjust if your path differs
    real_ms, _ = _measure(keepall, B=B, S=S, T=T, warmup=warmup, iters=iters, device=device)

    # Soft/proxy latency on *gated* model
    ms_like = proxy.predict(model, B=B, S=S, T=T)
    soft_ms = float(ms_like.detach().item()) if torch.is_tensor(ms_like) else float(ms_like)

    proxy.scale_ms = float(real_ms / max(soft_ms, 1e-9))
    return real_ms


@torch.inference_mode()
def calibrate_proxy_llm_from_batch(
    proxy: LatencyProxyLLM,
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    *,
    T: int,
    export_keepall_fn,
    device: str = "cuda",
    warmup: int = 10,
    iters: int = 30,
) -> Tuple[int, int, int, float]:
    """
    Infers (B,S) from a batch like {'input_ids': [B,S], ...},
    calibrates for (B,S,T), and returns (B,S,T, real_ms).
    """
    input_ids = batch["input_ids"]
    B, S = int(input_ids.size(0)), int(input_ids.size(1))
    ms = calibrate_proxy_llm(
        proxy, model, B=B, S=S, T=T, export_keepall_fn=export_keepall_fn,
        device=device, warmup=warmup, iters=iters
    )
    return B, S, T, ms
