"""HuggingFace ViT adapter

Bridges the family-agnostic core (gates/export/proxy/train) to ViT-like models
from Hugging Face (`ViTModel`, `ViTForImageClassification`, DeiT, etc.).

Responsibilities
----------------
- Attach gates to attention heads and MLP hidden in groups
- Provide logits getters for student/teacher
- Export helpers: keep-all (remove gates), and pruned (slice weights + metadata)

This adapter intentionally keeps the core unaware of ViT internals.
"""
from __future__ import annotations

# Ensure repo root on sys.path for absolute imports (core, adapters, data)
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from dataclasses import dataclass
from typing import Optional

import copy
import torch
import torch.nn as nn

# NOTE: absolute imports so running `-m examples.run_vit_optimize` works without package install
from core.gates import HeadGate, GroupGate
from core.export import (
    ExportPolicy as CoreExportPolicy,
    Rounding as CoreRounding,
    keep_group_indices_from_gate,
    keep_element_indices_from_gate,
    slice_linear,
    Rounding as CoreRounding,
)

from core.utils import deepcopy_eval_cpu
from core.search_export import grid_search_latency

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass
class ViTGatingConfig:
    tau: float = 1.5
    init_logit: float = 3.0
    head_gating: bool = True
    ffn_group: int = 16
    ffn_gating: bool = True
    hard_eval: bool = True  # use hard masks in eval mode during forward



def _encoder_layers(m: nn.Module):
    """
    Return the sequence of Transformer blocks for HF ViT.
    Supports:
      - ViTModel:                 m.encoder.layer
      - ViTForImageClassification: m.vit.encoder.layer
    """
    # ViTModel path
    enc = getattr(m, "encoder", None)
    if enc is not None and hasattr(enc, "layer"):
        return enc.layer

    # ViTForImageClassification path
    vit = getattr(m, "vit", None)
    if vit is not None and hasattr(vit, "encoder") and hasattr(vit.encoder, "layer"):
        return vit.encoder.layer

    raise ValueError("Provided model does not look like a HF ViT (missing *.encoder.layer)")



# -----------------------------------------------------------------------------
# Gated attention wrapper
# -----------------------------------------------------------------------------

class GatedSelfAttentionHF(nn.Module):
    """A thin wrapper around HF ViT self-attention that multiplies per-head gates.

    It keeps references to the underlying query/key/value `nn.Linear` layers and
    the output projection, while exposing a `HeadGate` in `head_gate`.
    """

    def __init__(self, attn_container: nn.Module, num_heads: int, head_dim: int, cfg: ViTGatingConfig):
        super().__init__()
        base_attn = attn_container.attention  # ViTSdpaSelfAttention or ViTSelfAttention
        out_proj = attn_container.output.dense

        self.base_attn = base_attn
        self.out_proj = out_proj

        self.q_proj = base_attn.query
        self.k_proj = base_attn.key
        self.v_proj = base_attn.value

        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.drop_p = getattr(base_attn, "dropout", nn.Dropout(0.0)).p

        self.head_gate = HeadGate(num_heads=self.num_heads, head_dim=self.head_dim, tau=cfg.tau, init_logit=cfg.init_logit, hard_during_eval=cfg.hard_eval)

    @property
    def logits(self) -> torch.Tensor:
        return self.head_gate.logits

    def kept_heads_soft(self) -> torch.Tensor:
        return self.head_gate.probs().sum()

    def forward(self, hidden_states, head_mask=None):
        B, N, _ = hidden_states.shape
        H, Dh = self.num_heads, self.head_dim

        wdev = self.q_proj.weight.device
        if hidden_states.device != wdev:
            hidden_states = hidden_states.to(wdev, non_blocking=True)

        q_lin = self.q_proj(hidden_states)
        k_lin = self.k_proj(hidden_states)
        v_lin = self.v_proj(hidden_states)

        q = q_lin.view(B, N, H, Dh).transpose(1, 2)
        k = k_lin.view(B, N, H, Dh).transpose(1, 2)
        v = v_lin.view(B, N, H, Dh).transpose(1, 2)

        logits = self.head_gate.logits
        tau = float(self.head_gate.tau)
        if self.training:
            u = torch.rand_like(logits).clamp_(1e-6, 1-1e-6)
            s = u.log() - (1 - u).log()
            y = torch.sigmoid((logits + s) / tau)
            g_head = ((y > 0.5).to(y.dtype) - y).detach() + y
        else:
            if getattr(self.head_gate, 'hard_during_eval', True):
                g_head = (logits > 0).to(logits.dtype)
            else:
                g_head = torch.sigmoid(logits / tau)
        g = g_head.view(1, H, 1, 1)

        q = q * g; k = k * g; v = v * g

        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=self.drop_p if self.training else 0.0
        )  # [B, H, N, Dh]

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, H * Dh)
        attn_out = self.out_proj(attn_out)
        return attn_out, None


# -----------------------------------------------------------------------------
# Adapter
# -----------------------------------------------------------------------------

class ViTAdapter:
    def __init__(self, model: nn.Module):
        self.model = model
        _ = _encoder_layers(model)

    # ---------- Gating attachment ----------
    def attach_gates(self, cfg: ViTGatingConfig) -> nn.Module:
        m = self.model
        H = int(getattr(m.config, "num_attention_heads", 12))
        D = int(getattr(m.config, "hidden_size", 768))
        Dh = D // H

        for layer in _encoder_layers(m):
            # Attention heads
            if cfg.head_gating:
                attn_container = layer.attention
                if not isinstance(getattr(attn_container, "attention", None), GatedSelfAttentionHF):
                    gated = GatedSelfAttentionHF(attn_container, H, Dh, cfg)
                    attn_container.attention = gated

            # FFN hidden (grouped)
            if cfg.ffn_gating:
                inter = layer.intermediate
                d_ff = int(inter.dense.out_features)
                assert d_ff % cfg.ffn_group == 0, f"FFN size {d_ff} not divisible by group {cfg.ffn_group}"
                if not hasattr(inter, "neuron_gate"):
                    inter.neuron_gate = GroupGate(num_groups=d_ff // cfg.ffn_group, group_size=cfg.ffn_group, tau=cfg.tau, init_logit=cfg.init_logit, hard_during_eval=cfg.hard_eval)
                # Monkey-patch forward to apply mask after activation (keeps HF shapes)
                if not hasattr(inter, "_orig_forward"):
                    inter._orig_forward = inter.forward

                    def _gated_forward(this, x):
                        h = this.dense(x)
                        h = this.intermediate_act_fn(h)
                        msk = this.neuron_gate.mask(this.training).view(1, 1, -1)
                        return h * msk

                    inter.forward = _gated_forward.__get__(inter, inter.__class__)
        return m

    # ---------- Logits helpers ----------
    @staticmethod
    def get_logits(model: nn.Module, x: torch.Tensor, *, head: Optional[nn.Module] = None) -> torch.Tensor:
        out = model(pixel_values=x)
        if hasattr(out, "logits"):
            return out.logits                        # ViTForImageClassification path
        if hasattr(out, "last_hidden_state"):        # ViTModel path (needs external head)
            if head is None:
                raise ValueError("Provide a classification head when using ViTModel without logits.")
            cls_tok = out.last_hidden_state[:, 0, :]
            if next(head.parameters(), torch.tensor([], device=cls_tok.device)).device != cls_tok.device:
                head = head.to(cls_tok.device)
            return head(cls_tok)
        raise ValueError("Model output lacks logits and last_hidden_state.")


    # ---------- Exporters ----------
    @staticmethod
    @torch.no_grad()
    def export_keepall(model_with_gates: nn.Module) -> nn.Module:
        slim = deepcopy_eval_cpu(model_with_gates)
        for layer in _encoder_layers(slim):
            # Attention: unwrap gate
            attn_container = layer.attention
            if isinstance(getattr(attn_container, "attention", None), GatedSelfAttentionHF):
                gat = attn_container.attention
                new_attn = copy.deepcopy(gat.base_attn)
                # restore HF metadata if present
                if hasattr(new_attn, "num_attention_heads"):
                    new_attn.num_attention_heads = int(gat.num_heads)
                if hasattr(new_attn, "attention_head_size"):
                    new_attn.attention_head_size = int(gat.head_dim)
                if hasattr(new_attn, "all_head_size"):
                    new_attn.all_head_size = int(gat.num_heads * gat.head_dim)
                attn_container.attention = new_attn
            # FFN: restore original forward and drop gate
            inter = layer.intermediate
            if hasattr(inter, "_orig_forward"):
                inter.forward = inter._orig_forward
                delattr(inter, "_orig_forward")
            if hasattr(inter, "neuron_gate"):
                delattr(inter, "neuron_gate")
        return slim

    @staticmethod
    @torch.no_grad()
    def export_pruned(model_with_gates: nn.Module, policy, step: int) -> nn.Module:
        # Support both CoreExportPolicy (single rounding) and ViTExportPolicy (per-axis)
        if isinstance(policy, ViTExportPolicy):
            head_rounding = policy.head_rounding
            ffn_rounding = policy.ffn_rounding
            warmup_steps = policy.warmup_steps
        else:
            # fallback to single rounding for both
            head_rounding = getattr(policy, "rounding", None)
            ffn_rounding = getattr(policy, "rounding", None)
            warmup_steps = int(getattr(policy, "warmup_steps", 0))
    
        slim = deepcopy_eval_cpu(model_with_gates)
        warm = (step < warmup_steps)
    
        for layer in _encoder_layers(slim):
            # --- Attention heads ---
            attn_container = layer.attention
            gat = getattr(attn_container, "attention", None)
            if isinstance(gat, GatedSelfAttentionHF):
                # choose rounding
                rnd = head_rounding
                # decide head indices via our helper; honor warmup if needed by passing step
                grp_idx = keep_group_indices_from_gate(
                    gat.head_gate,
                    policy=policy,
                    step=step,
                    custom_rounding=rnd,
                )
                H_keep = int(grp_idx.numel())
                Dh = int(gat.head_dim)
    
                ch_idx = torch.cat([torch.arange(h * Dh, (h + 1) * Dh) for h in grp_idx]).long()
                gat.q_proj = slice_linear(gat.q_proj, keep_out=ch_idx)
                gat.k_proj = slice_linear(gat.k_proj, keep_out=ch_idx)
                gat.v_proj = slice_linear(gat.v_proj, keep_out=ch_idx)
                attn_container.output.dense = slice_linear(attn_container.output.dense, keep_in=ch_idx)
    
                new_attn = copy.deepcopy(gat.base_attn)
                new_attn.query = gat.q_proj
                new_attn.key = gat.k_proj
                new_attn.value = gat.v_proj
                if hasattr(new_attn, "num_attention_heads"):
                    new_attn.num_attention_heads = H_keep
                if hasattr(new_attn, "attention_head_size"):
                    new_attn.attention_head_size = Dh
                if hasattr(new_attn, "all_head_size"):
                    new_attn.all_head_size = H_keep * Dh
                attn_container.attention = new_attn
    
            # --- FFN groups ---
            inter, out = layer.intermediate, layer.output
            g = getattr(inter, "neuron_gate", None)
            if g is not None:
                rnd = ffn_rounding
                grp_idx = keep_group_indices_from_gate(
                    g,
                    policy=policy,
                    step=step,
                    custom_rounding=rnd,
                )
                group = int(g.group_size)
                keep_exp = torch.cat([torch.arange(i * group, (i + 1) * group) for i in grp_idx]).long()
                inter.dense = slice_linear(inter.dense, keep_out=keep_exp)
                out.dense = slice_linear(out.dense, keep_in=keep_exp)
    
                # # restore clean forward & drop gate
                # if hasattr(inter, "_orig_forward"):
                #     def _clean_forward(this, x):
                #         h = this.dense(x)
                #         return this.intermediate_act_fn(h)
                #     inter.forward = _clean_forward.__get__(inter, inter.__class__)
                #     delattr(inter, "_orig_forward")
                # if hasattr(inter, "neuron_gate"):
                #     delattr(inter, "neuron_gate")

                inter.forward = inter.__class__.forward.__get__(inter, inter.__class__)
                if hasattr(inter, "neuron_gate"):
                    delattr(inter, "neuron_gate")
                if hasattr(inter, "_orig_forward"):
                    delattr(inter, "_orig_forward")                
               
    
        return slim


# -----------------------------------------------------------------------------
# Export policy
# -----------------------------------------------------------------------------
"""ViT-specific export policy that allows different rounding for heads vs FFN."""    
@dataclass
class ViTExportPolicy:
    warmup_steps: int = 0
    head_rounding: CoreRounding = CoreRounding()
    ffn_rounding: CoreRounding = CoreRounding()


@dataclass
class ViTGrid:
    head_multiple_grid: Optional[Sequence[int]] = (2, 4, 8)
    ffn_snap_grid: Sequence[int] = (1, 8) 
    # head_multiple_grid: Optional[Sequence[int]] = None  # default --> 1..num_heads    
    # ffn_snap_grid: Sequence[int] = (1, 2, 4, 8, 16) 


def vit_search_best_export(
    model_with_gates: nn.Module,
    *,
    export_fn: ExportFn,
    num_heads: int,
    step: int,
    batch_shape: Tuple[int, int, int, int],
    grid: Optional[ViTGrid] = None,
    device: str = "cuda",
    measure_settings: Optional[ProfileSettings] = None,
    make_policy: Optional[Callable[[int, int], object]] = None,
) -> SearchResult:
    """Convenience wrapper for ViT-style search.

    If `make_policy` is not provided, the caller's adapter should accept a
    policy with separate head/FFN rounding; see `adapters.huggingface.vit.ViTExportPolicy`.
    """
    g = grid or ViTGrid()
    head_grid = g.head_multiple_grid or list(range(1, int(num_heads) + 1))
    ffn_grid = list(g.ffn_snap_grid)

    return grid_search_latency(
        model_with_gates,
        export_fn,
        head_multiples=head_grid,
        ffn_snaps=ffn_grid,
        step=step,
        batch_shape=batch_shape,
        measure_settings=measure_settings,
        device=device,
        make_policy=make_policy,
    )
    

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
    """Latency proxy for ViT models. Accepts batches or (N,C,H,W) tuples."""

    def __init__(self, cfg: Optional[ViTProxyConfig] = None, lut: Optional[LatencyLUT] = None):
        super().__init__()
        self.cfg = cfg or ViTProxyConfig()
        self.lut = lut or LatencyLUT()

    # ---- helpers -------------------------------------------------------------
    @staticmethod
    def _input_spec(sample: TensorOrBatch) -> Tuple[int, int, int]:
        if isinstance(sample, (tuple, list)) and len(sample) == 4 and all(isinstance(x, int) for x in sample):
            B, C, H, W = sample
            return int(B), int(H), int(W)
        x = _first_tensor(sample)
        if x.dim() != 4:
            raise ValueError("ViTLatencyProxy expects a tensor [B,3,H,W] or a 4-tuple (B,3,H,W)")
        B, C, H, W = x.shape
        return int(B), int(H), int(W)

    @staticmethod
    def _vit_layers(m):
        enc = getattr(m, "encoder", None)
        if enc is not None and hasattr(enc, "layer"):
            return enc.layer
        vit = getattr(m, "vit", None)
        if vit is not None and hasattr(vit, "encoder") and hasattr(vit.encoder, "layer"):
            return vit.encoder.layer
        raise TypeError("Expected a HF ViT with *.encoder.layer (ViTModel or ViTForImageClassification).")            

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
    def _predict_raw(
        self,
        model: nn.Module,
        sample: TensorOrBatch,
        *,
        policy=None,
        step: Optional[int] = None
    ) -> torch.Tensor:
        anchor = next((p for p in model.parameters()), torch.tensor(0.0))

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

        layers = _vit_layers(model)
        for blk in layers:
            heads_soft = Hh if warm else (self._soft_heads_from_block(blk) or Hh)

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
        if torch.is_tensor(sample):
            shp = tuple(sample.shape)
        elif isinstance(sample, (tuple, list)):
            shp = tuple(sample)
        elif isinstance(sample, dict):
            shp = tuple((k, tuple(v.shape)) for k, v in sample.items() if torch.is_tensor(v))
        else:
            shp = (str(type(sample)),)
        cfg = getattr(model, "config", None)
        heads = int(getattr(cfg, "num_attention_heads", 12))
        hidden = int(getattr(cfg, "hidden_size", 768))
        inter = int(getattr(cfg, "intermediate_size", 3072))
        return ("ViT", shp, heads, hidden, inter)

    @torch.no_grad()
    def calibrate(self, model: nn.Module, shape: tuple, measure_fn, *, device: str = "cuda") -> float:
        """Set proxy scale so that keep-all student matches measured ms.
    
        `measure_fn(model, shape_or_tensor)` should return `(mean_ms, p95_ms)`.
        """
        
        sample_t = torch.randn(shape, device=device)
    
        sample_t = sample_t.to(device)
        model = model.to(device).eval()
        mean_ms, _ = measure_fn(model, shape, device=device)
        soft_ms = self.predict(model, sample_t).item()
        self.cfg.scale_ms = float(mean_ms / max(soft_ms, 1e-9))
        return self.cfg.scale_ms    