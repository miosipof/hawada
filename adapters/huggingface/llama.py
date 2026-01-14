"""HuggingFace LLaMA/Mistral adapter

Bridges the family-agnostic core (gates/export/proxy/train) to HF causal LMs
(LlamaForCausalLM / MistralForCausalLM, etc.).

Responsibilities
----------------
- Attach gates to attention Q heads (and optional KV) + grouped MLP (SwiGLU)
- Provide a logits getter (student/teacher)
- Exporters:
    * keep-all (unwrap gates, restore clean HF modules)
    * pruned (slice q_proj/o_proj and SwiGLU up/gate/down; update HF metadata)
- Grid-search wrapper for post-export rounding/snap params

This adapter intentionally keeps the core unaware of LLaMA internals.
"""
from __future__ import annotations

# Ensure repo root on sys.path for absolute imports (core, adapters, data)
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Tuple, Dict, Any

import copy
from copy import deepcopy
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import re

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from transformers import PreTrainedModel
from transformers import AutoConfig, AutoModelForCausalLM


# Core (absolute imports so running `-m examples.run_llama_optimize` works)
from core.gates import HeadGate, GroupGate
from core.export import (
    ExportPolicy as CoreExportPolicy,
    Rounding as CoreRounding,
    keep_group_indices_from_gate,
    slice_linear,
)
from core.utils import deepcopy_eval_cpu
from core.search_export import grid_search_latency
from core.proxy_cost import LatencyProxy

# -------------------------------------------------------------------------
# Configs
# -------------------------------------------------------------------------

@dataclass
class LlamaGatingConfig:
    tau: float = 1.5
    init_logit: float = 3.0
    head_gating: bool = True
    gate_kv: bool = False          # optional: gate KV along with Q
    ffn_group: int = 128           # SwiGLU groups
    ffn_gating: bool = True
    hard_eval: bool = True         # use hard gates in eval forward


# -------------------------------------------------------------------------
# Helpers (GQA, rotary, cache-safe)
# -------------------------------------------------------------------------


def _last_nonpad_index(attn_mask: Optional[torch.Tensor], seq_len: int, device) -> torch.Tensor:
    if attn_mask is None:
        return torch.full((1,), seq_len - 1, device=device, dtype=torch.long)  # will be expanded per-batch later
    # attn_mask: [B, S] in {0,1}; works for left/right padding
    return (attn_mask.sum(dim=1) - 1).clamp(min=0).long()
    
def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    B, Hkv, T, Dh = x.shape
    return x.unsqueeze(2).expand(B, Hkv, n_rep, T, Dh).reshape(B, Hkv * n_rep, T, Dh)

try:
    from transformers.cache_utils import Cache
except Exception:
    class Cache:  # type: ignore
        pass


# -------------------------------------------------------------------------
# Gated attention wrapper (Llama/Mistral ready)
# -------------------------------------------------------------------------

class GatedSelfAttentionLLM(nn.Module):
    """
    Thin wrapper around HF Llama/Mistral attention module.

    - Uses the base module's q_proj/k_proj/v_proj/o_proj
    - Applies per-Q-head gates (and optional KV gates)
    - Handles rotary and cache (tuple or HF Cache)
    - Runs SDPA directly, then o_proj
    """
    def __init__(self, attn_container: nn.Module,
                 num_q_heads: int, num_kv_heads: int, head_dim: int,
                 cfg: LlamaGatingConfig, layer_idx: int):
        super().__init__()
        self.base_attn = attn_container
        self.q_proj = attn_container.q_proj
        self.k_proj = attn_container.k_proj
        self.v_proj = attn_container.v_proj
        self.o_proj = getattr(attn_container, "o_proj", getattr(attn_container, "out_proj", None))

        self.num_q_heads = int(num_q_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.gate_kv = bool(cfg.gate_kv)
        self.drop_p = float(getattr(attn_container, "attention_dropout",
                                    getattr(attn_container, "attn_dropout",
                                            getattr(attn_container, "dropout", 0.0))))
        self.head_gate = HeadGate(num_heads=self.num_q_heads,
                                  head_dim=self.head_dim,
                                  tau=cfg.tau, init_logit=cfg.init_logit,
                                  hard_during_eval=cfg.hard_eval)

        # rotary helpers if present on base
        self.rotary_emb = getattr(attn_container, "rotary_emb", None)
        self.apply_rotary_pos_emb = getattr(attn_container, "apply_rotary_pos_emb", None)
        self.layer_idx = int(layer_idx)

    @property
    def logits(self) -> torch.Tensor:
        return self.head_gate.logits

    def kept_heads_soft(self) -> torch.Tensor:
        p = self.head_gate.probs().detach().float().view(-1)
        if p.numel() == self.num_q_heads * self.head_dim:
            p = p.view(self.num_q_heads, self.head_dim).mean(dim=1)
        return p.sum()


    def forward(
        self,
        hidden_states: torch.Tensor,                   # [B,T,D]
        attention_mask: Optional[torch.Tensor] = None, # additive mask [B,1,Tq,Tk] or None
        position_ids: Optional[torch.Tensor] = None,
        past_key_value = None,                         # tuple, list, Cache or None
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        B, T, D = hidden_states.shape
        Hq, Hkv, Dh = self.num_q_heads, self.num_kv_heads, self.head_dim
        assert Hq * Dh == D, "hidden_size must equal num_heads * head_dim"
        n_rep = max(1, Hq // Hkv)

        # qkv projections
        q = self.q_proj(hidden_states).view(B, T, Hq, Dh).transpose(1, 2)   # [B,Hq,T,Dh]
        k = self.k_proj(hidden_states).view(B, T, Hkv, Dh).transpose(1, 2)  # [B,Hkv,T,Dh]
        v = self.v_proj(hidden_states).view(B, T, Hkv, Dh).transpose(1, 2)  # [B,Hkv,T,Dh]

        # rotary
        if (self.rotary_emb is not None) and (self.apply_rotary_pos_emb is not None):
            Tpast = 0
            if isinstance(past_key_value, (tuple, list)) and len(past_key_value) == 2:
                Tpast = int(past_key_value[0].size(2))
            elif isinstance(past_key_value, Cache):
                Tpast = int(cache_position.max().item() if cache_position is not None else 0)
            seq_len = Tpast + T
            try:
                cos, sin = self.rotary_emb(v, seq_len=seq_len)
            except TypeError:
                cos, sin = self.rotary_emb(q, seq_len=seq_len)
            # try rich signature first
            try:
                q, k = self.apply_rotary_pos_emb(
                    q, k, cos, sin,
                    position_ids=position_ids,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings
                )
            except TypeError:
                try:
                    q, k = self.apply_rotary_pos_emb(q, k, cos, sin, position_ids=position_ids)
                except TypeError:
                    q, k = self.apply_rotary_pos_emb(q, k, cos, sin)

        # cache merge
        present = None
        if past_key_value is None or (isinstance(past_key_value, (tuple, list)) and len(past_key_value) == 0):
            pass
        elif isinstance(past_key_value, (tuple, list)):
            pk, pv = past_key_value   # [B,Hkv,Tpast,Dh]
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
            present = (k, v) if use_cache else None
        elif isinstance(past_key_value, Cache):
            k, v = past_key_value.update(k, v, self.layer_idx, cache_position)
            present = past_key_value

        # gates
        # g = self.head_gate.mask(self.training).view(1, Hq, 1, 1)
        # ---- gates (supports per-head OR per-channel HeadGate) ----
        m = self.head_gate.mask(self.training)                   # 1D tensor
        m = m.detach() if not self.training else m
        if m.numel() == Hq:
            # per-head gating
            gH = m.view(1, Hq, 1, 1)                             # [1,Hq,1,1]
            q = q * gH
            if self.gate_kv:
                if n_rep == 1:
                    k = k * gH; v = v * gH
                else:
                    g_kv = gH.view(1, Hkv, n_rep, 1, 1).amax(dim=2)
                    k = k * g_kv; v = v * g_kv
        elif m.numel() == Hq * Dh:
            # per-channel gating
            gHD = m.view(1, Hq, 1, Dh)                           # [1,Hq,1,Dh]
            q = q * gHD
            if self.gate_kv:
                # collapse to per-head for KV, then map to Hkv via amax over replicas
                gH = gHD.amax(dim=-1, keepdim=True)              # [1,Hq,1,1]
                if n_rep == 1:
                    k = k * gH; v = v * gH
                else:
                    g_kv = gH.view(1, Hkv, n_rep, 1, 1).amax(dim=2)
                    k = k * g_kv; v = v * g_kv
        else:
            raise RuntimeError(
                f"HeadGate mask has {m.numel()} elems; expected {Hq} or {Hq*Dh}"
            )


        # GQA replicate KV to Q count
        k = _repeat_kv(k, n_rep)
        v = _repeat_kv(v, n_rep)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.drop_p if self.training else 0.0,
            is_causal=True
        )
        out = attn.transpose(1, 2).contiguous().view(B, T, Hq * Dh)
        out = self.o_proj(out)

        attn_weights = None
        # HF expects (attn_output, attn_weights, present_key_value) always
        if output_attentions:
            return (out, attn_weights, present)
        else:
            return (out, None, present)



# -------------------------------------------------------------------------
# Adapter
# -------------------------------------------------------------------------

class LlamaAdapter:
    def __init__(self, model: nn.Module):
        self.model = model
        core = getattr(model, "model", model)
        if not hasattr(core, "layers"):
            raise ValueError("Provided model does not look like HF LLaMA/Mistral (missing .model.layers or .layers)")

    # ---------- Gating attachment ----------
    def attach_gates(self, cfg: LlamaGatingConfig) -> nn.Module:
        m = self.model
        core = getattr(m, "model", m)
        layers = core.layers

        Hq  = int(core.config.num_attention_heads)
        Hkv = int(getattr(core.config, "num_key_value_heads", Hq))
        Dh  = int(core.config.hidden_size // Hq)

        for li, layer in enumerate(layers):
            # Attention heads
            if cfg.head_gating:
                base = layer.self_attn
                if not isinstance(base, GatedSelfAttentionLLM):
                    gated = GatedSelfAttentionLLM(
                        attn_container=base,
                        num_q_heads=Hq,
                        num_kv_heads=Hkv,
                        head_dim=Dh,
                        cfg=cfg,
                        layer_idx=li,
                    )
                    layer.self_attn = gated  # route via our wrapper

            # MLP grouped gating (SwiGLU)
            if cfg.ffn_gating:
                mlp = layer.mlp
                I = int(mlp.up_proj.out_features)
                assert I % cfg.ffn_group == 0, f"SwiGLU size {I} not divisible by group {cfg.ffn_group}"
                if not hasattr(mlp, "neuron_gate"):
                    mlp.neuron_gate = GroupGate(
                        num_groups=I // cfg.ffn_group,
                        group_size=cfg.ffn_group,
                        tau=cfg.tau, init_logit=cfg.init_logit,
                        hard_during_eval=cfg.hard_eval,
                    )
                if not hasattr(mlp, "_orig_forward"):
                    mlp._orig_forward = mlp.forward

                    def _gated_mlp_forward(this, x):
                        # LLaMA: z = silu(up(x)) * (gate(x) * m); out = down(z)
                        u = this.up_proj(x)
                        g = this.gate_proj(x)
                        m = this.neuron_gate.mask(this.training).view(1, 1, -1)
                        z = torch.nn.functional.silu(u) * (g * m)
                        return this.down_proj(z)

                    mlp.forward = _gated_mlp_forward.__get__(mlp, mlp.__class__)
        return m

    # ---------- Logits helper ----------
    @staticmethod
    def _last_token_index(attn_mask: torch.Tensor) -> torch.Tensor:
        # attn_mask: [B, S] with 1 for tokens, 0 for padding
        # returns [B] indices of last non-pad
        # works for both bool and int masks
        if attn_mask is None:
            # no mask → use last position S-1
            return None
        if attn_mask.dtype != torch.long:
            attn_mask = attn_mask.to(torch.long)
        # idx = lengths - 1
        return (attn_mask.sum(dim=-1) - 1).clamp_min(0)

    @staticmethod
    def get_logits(model: nn.Module,
                   input_ids: torch.Tensor,
                   attention_mask: Optional[torch.Tensor] = None,
                   last_only: bool = True,
                   **forward_kwargs) -> torch.Tensor:
        """
        Returns logits. If last_only=True, computes ONLY the last-token logits by:
          1) getting hidden states from the base decoder,
          2) selecting last non-pad position per sample,
          3) projecting through lm_head on that 1 position.
        This avoids allocating [B,S,V].
        """
        # (1) run base decoder, not the full CausalLM head
        core = getattr(model, "model", None)
        if core is None:
            # fallback if the model is already a bare decoder (rare)
            core = model

        # We only need last_hidden_state; no cache; avoid building logits for all S
        # return_dict=False to grab tuple and avoid extra allocations
        outputs = core(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=False,
            **forward_kwargs
        )
        hidden = outputs[0]  # [B, S, D]

        if not last_only:
            # If someone explicitly wants all logits, fine:
            return model.lm_head(hidden)  # [B,S,V] (expensive!)

        # (2) select last token per sample
        B, S, D = hidden.shape
        if attention_mask is None:
            # simple "last index"
            idx = torch.full((B,), S - 1, device=hidden.device, dtype=torch.long)
        else:
            idx = LlamaAdapter._last_token_index(attention_mask)

        # gather last hidden: [B, D]
        last_h = hidden[torch.arange(B, device=hidden.device), idx]  # [B, D]
        # (3) project to logits for that 1 position
        last_logits = model.lm_head(last_h).unsqueeze(1)  # [B,1,V]
        return last_logits

    # ---------- Exporters ----------
    @staticmethod
    @torch.no_grad()
    def export_keepall(model_with_gates: nn.Module) -> nn.Module:
        """
        Unwrap attention wrappers; restore original MLP.forward; drop gates.
        """
        slim = deepcopy_eval_cpu(model_with_gates)
        core = getattr(slim, "model", slim)
        if not hasattr(core, "layers"):
            return slim

        for layer in core.layers:
            # attention
            attn = layer.self_attn
            if isinstance(attn, GatedSelfAttentionLLM):
                gat = attn
                new_attn = copy.deepcopy(gat.base_attn)
                # keep metadata consistent
                if hasattr(new_attn, "num_heads"):
                    new_attn.num_heads = int(gat.num_q_heads)
                if hasattr(new_attn, "num_key_value_heads"):
                    new_attn.num_key_value_heads = int(gat.num_kv_heads)
                if hasattr(new_attn, "head_dim"):
                    new_attn.head_dim = int(gat.head_dim)
                layer.self_attn = new_attn

            # mlp
            mlp = layer.mlp
            if hasattr(mlp, "_orig_forward"):
                mlp.forward = mlp._orig_forward
                delattr(mlp, "_orig_forward")
            if hasattr(mlp, "neuron_gate"):
                delattr(mlp, "neuron_gate")

        return slim
    
    @staticmethod
    @torch.no_grad()
    def export_pruned(model_with_gates: nn.Module, policy, step: int) -> nn.Module:
        """
        Produce a clean CPU eval model:
    
          - Attention:
            * Read per-head gates, rank heads.
            * Choose H_keep heads with rounding/constraints.
            * Slice q_proj rows and o_proj cols.
            * Keep k_proj / v_proj (GQA) but update num_heads / num_key_value_groups.
    
          - MLP:
            * Read GroupGate over SwiGLU expansion.
            * Choose kept groups with rounding/constraints.
            * Slice up_proj/gate_proj (out) and down_proj (in).
            * Restore original forward, drop gates.
    
        Does NOT touch:
          - hidden_size / embeddings / norms / lm_head.
        """
    
        # -------------------------------------------------------------------------
        # Unpack policy
        # -------------------------------------------------------------------------
    
        if isinstance(policy, LlamaExportPolicy):
            head_rounding = policy.head_rounding   # has: floor_groups, multiple_groups, min_keep_ratio
            ffn_rounding  = policy.ffn_rounding
            q_rounding    = getattr(policy, "q_rounding", None)  # currently unused, kept for future
            warmup_steps  = int(policy.warmup_steps)
        else:
            head_rounding = getattr(policy, "rounding", None)
            ffn_rounding  = getattr(policy, "rounding", None)
            q_rounding    = None
            warmup_steps  = int(getattr(policy, "warmup_steps", 0))
    
        # Safety defaults if rounding is None
        class _DefaultRound:
            def __init__(self):
                self.floor_groups = 1
                self.multiple_groups = 1
                self.min_keep_ratio = 0.0
    
        if head_rounding is None:
            head_rounding = _DefaultRound()
        if ffn_rounding is None:
            ffn_rounding = _DefaultRound()
    
        warm = (step < warmup_steps)
    
        # -------------------------------------------------------------------------
        # Clone model to CPU + eval
        # -------------------------------------------------------------------------
        def deepcopy_eval_cpu(m: nn.Module) -> nn.Module:
            m = deepcopy(m)
            m.eval()
            return m.cpu()
    
        slim = deepcopy_eval_cpu(model_with_gates)
        core = getattr(slim, "model", slim)
        layers = getattr(core, "layers", None)
        if layers is None:
            return slim
        
        # -------------------------------------------------------------------------
        # Helpers
        # -------------------------------------------------------------------------
        def _snap_heads(Hq: int, raw_keep: int, Hkv: int, cfg) -> int:
            """
            Snap number of kept heads to something compatible with:
              - min_keep_ratio
              - floor_groups
              - multiple_groups
              - GQA constraint (multiple of Hkv).
            """
            # minimum allowed heads from ratio / floor
            min_by_ratio = int(math.ceil(cfg.min_keep_ratio * Hq))
            min_by_floor = int(cfg.floor_groups)
            min_keep = max(min_by_ratio, min_by_floor, Hkv)  # must have at least one KV-group
    
            # base step in heads: must be multiple of Hkv
            step = int(cfg.multiple_groups)
            if step < Hkv:
                # we must respect GQA: groups = H_keep / Hkv must be integer
                step = Hkv
    
            raw_keep = max(min_keep, min(raw_keep, Hq))
            snapped = (raw_keep // step) * step
            if snapped < min_keep:
                snapped = min_keep
            if snapped > Hq:
                snapped = Hq
    
            # final guard: GQA groups integer
            if snapped % Hkv != 0:
                snapped = (snapped // Hkv) * Hkv
                snapped = max(min_keep, min(snapped, Hq))
    
            return snapped
    
        def _snap_groups(G: int, raw_keep: int, cfg) -> int:
            """
            Snap number of kept FFN groups.
            """
            min_by_ratio = int(math.ceil(cfg.min_keep_ratio * G))
            min_by_floor = int(cfg.floor_groups)
            min_keep = max(1, min_by_ratio, min_by_floor)
    
            step = max(1, int(cfg.multiple_groups))
    
            raw_keep = max(min_keep, min(raw_keep, G))
            snapped = (raw_keep // step) * step
            if snapped < min_keep:
                snapped = min_keep
            if snapped > G:
                snapped = G
            return snapped
    
        def _normalize_scores(x: torch.Tensor) -> torch.Tensor:
            """
            Normalize logits for stable ranking (optional but helps when range is tiny).
            """
            if x.numel() == 0:
                return x
            mean = x.mean()
            std = x.std()
            if float(std) < 1e-6:
                return x - mean
            return (x - mean) / std
    
        # -------------------------------------------------------------------------
        # Main loop over layers
        # -------------------------------------------------------------------------
        for li, layer in enumerate(layers):
            # ====================== ATTENTION PRUNING ============================
            attn = layer.self_attn
    
            if isinstance(attn, GatedSelfAttentionLLM):
                gat = attn
                base = gat.base_attn
    
                Hq  = int(gat.num_q_heads)
                Hkv = int(gat.num_kv_heads)
                Dh  = int(gat.head_dim)
    
                if warm:
                    # keep all heads during warmup
                    keep_idx = torch.arange(Hq)
                else:
                    base_logits = gat.head_gate.logits.detach().float().view(-1)
    
                    # per-head vs per-channel gate
                    if base_logits.numel() == Hq:
                        # per-head gate
                        proxy_gate = gat.head_gate
                        raw_keep_idx = keep_group_indices_from_gate(
                            proxy_gate, policy=policy, step=step, custom_rounding=head_rounding
                        )
                        # use logits directly as scores
                        scores = base_logits
                    elif base_logits.numel() == Hq * Dh:
                        # per-channel gate → average to per-head
                        per_head_logits = base_logits.view(Hq, Dh).mean(dim=1)
    
                        class _PerHeadProxyGate:
                            def __init__(self, logits, tau):
                                self.logits = logits
                                self.tau = tau
                                self.num_groups = logits.numel()
                                self.group_size = 1
    
                        proxy_gate = _PerHeadProxyGate(per_head_logits, float(getattr(gat.head_gate, "tau", 1.0)))
                        raw_keep_idx = keep_group_indices_from_gate(
                            proxy_gate, policy=policy, step=step, custom_rounding=head_rounding
                        )
                        scores = per_head_logits
                    else:
                        raise RuntimeError(
                            f"[export_pruned] Unexpected HeadGate logits len {base_logits.numel()} "
                            f"vs H={Hq} or H*Dh={Hq*Dh}"
                        )
    
                    # If gate helper returns everything, we still consider Hq as raw_keep
                    raw_keep = int(raw_keep_idx.numel())
                    H_keep = _snap_heads(Hq, raw_keep, Hkv, head_rounding)
    
                    # Normalize scores for a sharper top-k
                    scores = _normalize_scores(scores)
    
                    # Recompute final keep_idx as top-k by scores
                    k = int(H_keep)
                    keep_idx = torch.topk(scores, k=k, largest=True).indices.sort().values
    
                    # ---- DEBUG LOGGING (optional) ----
                    if step % 50 == 0 and li < 16:  # you can tweak this condition
                        print(
                            f"[DEBUG L{li}] Hq={Hq}, raw_keep={raw_keep}, snapped={H_keep}, "
                            f"min_logit={float(scores.min()):.4f}, max_logit={float(scores.max()):.4f}"
                        )
    
                H_keep = int(keep_idx.numel())
                assert H_keep > 0, f"[export_pruned] H_keep=0 at layer {li}"
                assert H_keep % Hkv == 0, f"[export_pruned] H_keep={H_keep} not divisible by Hkv={Hkv} at layer {li}"
    
                # channels for q/o slicing
                ch_idx = torch.cat(
                    [torch.arange(h * Dh, (h + 1) * Dh, dtype=torch.long) for h in keep_idx]
                )
    
                # slice wrapper linears (q out, o in)
                gat.q_proj = slice_linear(gat.q_proj, keep_out=ch_idx)
                gat.o_proj = slice_linear(gat.o_proj, keep_in=ch_idx)
    
                # transplant into a clean HF attention
                new_attn = deepcopy(base)
                if hasattr(new_attn, "q_proj"):
                    new_attn.q_proj = gat.q_proj
                if hasattr(new_attn, "o_proj"):
                    new_attn.o_proj = gat.o_proj
                elif hasattr(new_attn, "out_proj"):
                    new_attn.out_proj = gat.o_proj
    
                # update metadata
                if hasattr(new_attn, "num_heads"):
                    new_attn.num_heads = int(H_keep)
                if hasattr(new_attn, "num_key_value_heads"):
                    new_attn.num_key_value_heads = int(Hkv)
                if hasattr(new_attn, "head_dim"):
                    new_attn.head_dim = int(Dh)
                if hasattr(new_attn, "num_key_value_groups"):
                    new_attn.num_key_value_groups = max(1, int(H_keep) // int(Hkv))
    
                # plug back
                layer.self_attn = new_attn
    
            # ========================= MLP PRUNING ==============================
            mlp = layer.mlp
            g = getattr(mlp, "neuron_gate", None)
            if g is not None:
                # g is GroupGate over expansion dimension (e.g., 64 groups)
                G = int(g.num_groups)
                logits = g.logits.detach().float().view(-1)
                logits = _normalize_scores(logits)
    
                raw_grp_idx = keep_group_indices_from_gate(
                    g, policy=policy, step=step, custom_rounding=ffn_rounding
                )
                raw_keep_groups = int(raw_grp_idx.numel())
                keep_groups = _snap_groups(G, raw_keep_groups, ffn_rounding)
    
                # final group indices as top-k by normalized logits
                grp_scores = logits
                grp_k = int(keep_groups)
                grp_idx = torch.topk(grp_scores, k=grp_k, largest=True).indices.sort().values
    
                group_size = int(g.group_size)
                keep_exp = torch.cat(
                    [torch.arange(i * group_size, (i + 1) * group_size, dtype=torch.long) for i in grp_idx]
                )
    
                # slice SwiGLU projections
                mlp.up_proj   = slice_linear(mlp.up_proj,   keep_out=keep_exp)
                mlp.gate_proj = slice_linear(mlp.gate_proj, keep_out=keep_exp)
                mlp.down_proj = slice_linear(mlp.down_proj, keep_in=keep_exp)
    
                # Restore clean forward & drop gate
                if hasattr(mlp, "_orig_forward"):
                    mlp.forward = mlp._orig_forward
                    delattr(mlp, "_orig_forward")
                if hasattr(mlp, "neuron_gate"):
                    delattr(mlp, "neuron_gate")
    
                # ---- DEBUG LOGGING (optional) ----
                if step % 50 == 0 and li < 16:
                    print(
                        f"[DEBUG L{li}] FFN groups: initial G={G}, raw_keep={raw_keep_groups}, snapped={keep_groups}"
                    )
    
        return slim


# -------------------------------------------------------------------------
# Export policy (allow different rounding for Heads vs FFN)
# -------------------------------------------------------------------------

@dataclass
class LlamaExportPolicy:
    warmup_steps: int = 0
    head_rounding: CoreRounding = field(default_factory=CoreRounding)
    ffn_rounding:  CoreRounding = field(default_factory=CoreRounding)
    q_rounding:    CoreRounding = field(default_factory=CoreRounding)


# -------------------------------------------------------------------------
# Grid-search convenience
# -------------------------------------------------------------------------

@dataclass
class LlamaGrid:
    head_multiple_grid: Optional[Sequence[int]] = (1, 2, 4, 8)
    ffn_snap_grid: Sequence[int] = (1, 32, 64, 128)

def llama_search_best_export(
    model_with_gates: nn.Module,
    *,
    export_fn: Callable[[nn.Module, CoreExportPolicy, int], nn.Module],
    num_q_heads: int,
    num_kv_heads: int,
    step: int,
    batch_shape: Tuple[int, int],  # (B,S) for text
    grid: Optional[LlamaGrid] = None,
    device: str = "cuda",
    measure_settings=None,
    make_policy: Optional[Callable[[int, int], object]] = None,
):
    """
    Convenience wrapper for LLaMA-style search.
    Uses the same `grid_search_latency` as ViT; we just feed head/ffn grids.
    """
    g = grid or LlamaGrid()
    head_grid = g.head_multiple_grid or [1]
    ffn_grid = list(g.ffn_snap_grid)

    return grid_search_latency(
        model_with_gates,
        export_fn,
        head_multiples=head_grid,
        ffn_snaps=ffn_grid,
        step=step,
        batch_shape=batch_shape,         # adapter’s runner should interpret as (B,S)
        measure_settings=measure_settings,
        device=device,
        make_policy=make_policy,
    )



# -----------------------------------------------------------------------------
# Latency proxy
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
# Wrapper to mimic HF
# ------------------------------------------------------------



class SlimLlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig, layer_meta):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                SlimLlamaDecoderLayer(config, Hq=meta["Hq"], d_ff=meta["d_ff"])
                for meta in layer_meta
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        cache_position=None,
    ):
        # Very similar to HF's LlamaModel.forward, but we pass layer_meta-driven modules.
        if input_ids is None:
            raise ValueError("input_ids required")

        bsz, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)

        # Build causal mask + attention mask once (you already do this in your profiler)
        causal_mask = attention_mask  # reuse your existing mask building

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_past_key_values = () if use_cache else None

        # rotary precompute
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer_idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_kv = past_key_values[layer_idx] if past_key_values is not None else None

            layer_outputs = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_kv,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_past_key_values += (layer_outputs[1],)
            if output_attentions:
                all_self_attns += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        if not return_dict:
            outputs = (hidden_states,)
            if use_cache:
                outputs += (next_past_key_values,)
            if output_hidden_states:
                outputs += (all_hidden_states,)
            if output_attentions:
                outputs += (all_self_attns,)
            return outputs

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


# class SlimLlamaForCausalLM(PreTrainedModel):
#     config_class = LlamaConfig  # reuse

#     def __init__(self, config: LlamaConfig, layer_meta):
#         super().__init__(config)
#         self.model = SlimLlamaModel(config, layer_meta)
#         self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
#         self.post_init()
        
#     def get_input_embeddings(self):
#         return self.model.embed_tokens

#     def set_input_embeddings(self, new_emb):
#         self.model.embed_tokens = new_emb

#     def get_output_embeddings(self):
#         return self.lm_head

#     def set_output_embeddings(self, new_lm_head):
#         self.lm_head = new_lm_head
        
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         position_ids=None,
#         labels=None,
#         use_cache=False,
#         output_attentions=False,
#         output_hidden_states=False,
#         return_dict=True,
#         past_key_values=None,
#         cache_position=None,
#     ):
#         outputs = self.model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             cache_position=cache_position,
#         )

#         hidden_states = outputs.last_hidden_state
#         logits = self.lm_head(hidden_states)

#         if labels is not None:
#             shift_logits = logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()
#             loss_fct = nn.CrossEntropyLoss()
#             loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
#         else:
#             loss = None

#         if not return_dict:
#             out = (logits,) + outputs[1:]
#             return ((loss,) + out) if loss is not None else out

#         from transformers.modeling_outputs import CausalLMOutputWithPast
#         return CausalLMOutputWithPast(
#             loss=loss,
#             logits=logits,
#             past_key_values=outputs.past_key_values,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )


class SlimLlamaForCausalLM(PreTrainedModel):
    config_class = LlamaConfig  # reuse HF config


    def __init__(self, config: LlamaConfig, layer_meta: Optional[list] = None):
        # Make sure HF doesn't try to tie them automatically
        config.tie_word_embeddings = False
        super().__init__(config)

        if layer_meta is None:
            layer_meta = getattr(config, "layer_meta", None)
        if layer_meta is None:
            raise ValueError("SlimLlamaForCausalLM requires `layer_meta`.")

        self.model = SlimLlamaModel(config, layer_meta)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def tie_weights(self):
        # override to NO-OP: no shared tensor between embed and lm_head
        return

    # ----- Embedding helpers -----

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, new_emb):
        self.model.embed_tokens = new_emb

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_lm_head):
        self.lm_head = new_lm_head

    # ----- Forward / Causal LM head -----

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        labels=None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        past_key_values=None,
        cache_position=None,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # standard causal LM shift
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        if not return_dict:
            out = (logits,) + outputs[1:]
            return ((loss,) + out) if loss is not None else out

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
class SlimLlamaSdpaAttention(nn.Module):
    def __init__(self, config: LlamaConfig, Hq: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads  # 64
        # num_heads is derived from pruned q size:
        self.num_heads = Hq // self.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, Hq, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(Hq, self.hidden_size, bias=False)

        # rotary stays same
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.attention_dropout = config.attention_dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: torch.Tensor | None = None,
    ):
        bsz, q_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states)                     # (B, T, Hq)
        k = self.k_proj(hidden_states)                     # (B, T, n_kv * d)
        v = self.v_proj(hidden_states)

        # (B, T, n_heads, d)
        q = q.view(bsz, q_len, self.num_heads, self.head_dim)
        # (B, T, n_kv, d)
        k = k.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        v = v.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        # transpose to (B, n_heads, T, d)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # rotary
        if position_embeddings is None:
            cos, sin = self.rotary_emb(v, position_ids)
        else:
            cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # group query attention: repeat kv heads
        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # causal_mask = attention_mask  # you already pass combined mask from LlamaModel
        # if causal_mask is not None and not (
        #     causal_mask.dtype.is_floating_point or causal_mask.dtype == torch.bool
        # ):
        #     causal_mask = causal_mask.to(dtype=q.dtype)

        causal_mask = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # attention_mask: (B, S) with 1 for tokens, 0 for pads
                # -> (B, 1, 1, S) additive mask 0 / -inf
                causal_mask = attention_mask[:, None, None, :].to(q.dtype)
                causal_mask = (1.0 - causal_mask) * torch.finfo(q.dtype).min
            elif attention_mask.dim() == 4:
                # already a 4D mask, just cast to q.dtype
                causal_mask = attention_mask.to(q.dtype)
            else:
                raise ValueError(
                    f"Unsupported attention_mask dim={attention_mask.dim()} "
                    "for SlimLlamaSdpaAttention"
                )
            
        
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=causal_mask is None,
        )                                              # (B, n_heads, T, d)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.num_heads * self.head_dim)  # (B, T, Hq)
        attn_output = self.o_proj(attn_output)                                      # (B, T, hidden_size)

        outputs = (attn_output, None, None)
        if use_cache:
            # re-use HF's cache format: (k, v)
            present_key_value = (k, v)
            outputs = outputs + (present_key_value,)
        if output_attentions:
            raise NotImplementedError("attn weights not wired, but easy to add if needed")

        return outputs



class SlimLlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig, d_ff: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = d_ff

        self.gate_proj = nn.Linear(self.hidden_size, d_ff, bias=False)
        self.up_proj   = nn.Linear(self.hidden_size, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, self.hidden_size, bias=False)

        self.act_fn = nn.SiLU()  # Llama uses SiLU

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))




class SlimLlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, Hq: int, d_ff: int):
        super().__init__()
        self.self_attn = SlimLlamaSdpaAttention(config, Hq=Hq)
        self.mlp = SlimLlamaMLP(config, d_ff=d_ff)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_outputs = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + attn_outputs[0]

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)

        outputs = (hidden_states,)
        if use_cache:
            outputs = outputs + (attn_outputs[2],)  # present_key_value
        if output_attentions:
            outputs = outputs + (attn_outputs[1],)
        return outputs




class SlimLlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, Hq: int, d_ff: int):
        super().__init__()
        self.self_attn = SlimLlamaSdpaAttention(config, Hq=Hq)
        self.mlp = SlimLlamaMLP(config, d_ff=d_ff)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_outputs = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + attn_outputs[0]

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)

        outputs = (hidden_states,)
        if use_cache:
            outputs = outputs + (attn_outputs[2],)  # present_key_value
        if output_attentions:
            outputs = outputs + (attn_outputs[1],)
        return outputs





def load_slim_llama(slim_dir: str, dense_id: str, device="cuda"):
    config = AutoConfig.from_pretrained(dense_id)
    with open(os.path.join(slim_dir, "slim_meta.json"), "r") as f:
        layer_meta = json.load(f)

    model = SlimLlamaForCausalLM(config, layer_meta)
    slim = torch.load(os.path.join(slim_dir, "slim.pt"), map_location="cpu", weights_only=False)

    try:
        missing, unexpected = model.load_state_dict(slim, strict=False)
        print("missing:", missing)
        print("unexpected:", unexpected)
    except Exception as e:
        print("Unable to load as state_dict, trying to load as full model...")

        state_dict = slim.state_dict()
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("missing:", missing)
        print("unexpected:", unexpected)
        
        
    return model.to(device).eval()




def infer_slim_meta(slim_pt_path: str, output_json: str = None):
    """
    Infer per-layer slim metadata (Hq and d_ff) from slim.pt.
    Produces a list like:
      [{"Hq": 1536, "d_ff": 4096}, ...]
    """

    print(f"[load] slim state_dict: {slim_pt_path}")
    sd = torch.load(slim_pt_path, map_location="cpu", weights_only=False)

    # Detect prefix (model.layers or layers)
    prefixes = []
    for k in sd.keys():
        if k.startswith("model.layers."):
            prefixes.append("model.layers")
            break
        if k.startswith("layers."):
            prefixes.append("layers")
            break
    if not prefixes:
        raise RuntimeError("Cannot find layer prefix (model.layers or layers) in slim.pt")

    prefix = prefixes[0]
    print(f"[info] detected prefix: '{prefix}'")

    # Find number of layers
    layer_ids = set()
    layer_pat = re.compile(rf"^{prefix}\.(\d+)\.")
    for k in sd.keys():
        m = layer_pat.match(k)
        if m:
            layer_ids.add(int(m.group(1)))

    if not layer_ids:
        raise RuntimeError("Could not infer number of layers from slim state_dict")

    num_layers = max(layer_ids) + 1
    print(f"[info] detected num_layers = {num_layers}")

    layer_meta = []

    for layer in range(num_layers):
        base = f"{prefix}.{layer}"

        # --- Infer attention head count ---
        qw = sd.get(f"{base}.self_attn.q_proj.weight", None)
        if qw is None:
            raise RuntimeError(f"Missing q_proj at layer {layer}")

        Hq = qw.shape[0]      # q_proj.out_features
        # But this is channels = Hq; in LLaMA: Hq = num_heads * head_dim

        # --- Infer FFN expansion ---
        up = sd.get(f"{base}.mlp.up_proj.weight", None)
        if up is None:
            raise RuntimeError(f"Missing mlp.up_proj at layer {layer}")

        d_ff = up.shape[0]    # up_proj.out_features

        layer_meta.append({
            "Hq": int(Hq),
            "d_ff": int(d_ff)
        })

        print(f"[L{layer}] Hq={Hq}, d_ff={d_ff}")

    # Save
    if output_json is None:
        output_json = os.path.join(os.path.dirname(slim_pt_path), "slim_meta.json")

    with open(output_json, "w") as f:
        json.dump(layer_meta, f, indent=2)

    print(f"[save] wrote {output_json}")
    return layer_meta



def load_slim_for_finetune(
    dense_id: str,
    slim_dir: str,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.bfloat16,
) -> SlimLlamaForCausalLM:
    """
    Load a pruned slim model (slim.pt) into a SlimLlamaForCausalLM instance
    that is ready for training / fine-tuning.
    """

    # 1. Load dense base config + weights (we use it as a template)
    dense = AutoModelForCausalLM.from_pretrained(
        dense_id,
        torch_dtype=dtype,
    )

    # 4. Build SlimLlamaForCausalLM with pruned modules + slim weights
    slim = load_slim_llama(slim_dir, dense_id, device=device)


    # 5. Make sure training-ish flags are sane
    slim.config.use_cache = False          # we don’t want cache during training
    slim.config._attn_implementation = "sdpa"  # consistent with our forward
    slim.to(device)
    slim.train()

    return slim



def load_slim_llama(
    slim_ckpt: Path,
    base_id: str,
    device: str = "cpu",
) -> SlimLlamaForCausalLM:
    """
    Load a locally-trained slim LLaMA checkpoint and wrap it in SlimLlamaForCausalLM.

    Expected formats for `torch.load(slim_ckpt)`:

    1) dict with keys:
         - "config"      (optional, LlamaConfig or dict)
         - "layer_meta"  (list/dict with per-layer Hq, d_ff)
         - "state_dict"  (actual model SD)

    2) nn.Module (SlimLlamaForCausalLM or SlimLlamaModel wrapper)
       -> we infer layer_meta from module and take its state_dict().
    """
    obj = torch.load(slim_ckpt, map_location=device, weights_only=False)

    # Case 1: dict with metadata
    if isinstance(obj, dict) and "layer_meta" in obj:
        layer_meta = obj["layer_meta"]
        if "state_dict" in obj:
            sd = obj["state_dict"]
        elif "model" in obj and isinstance(obj["model"], torch.nn.Module):
            sd = obj["model"].state_dict()
        else:
            # assume it's already a state_dict
            sd = obj
    # Case 2: plain state_dict (no metadata in file)
    elif isinstance(obj, dict):
        raise ValueError(
            f"Slim checkpoint {slim_ckpt} is a plain state_dict without `layer_meta`.\n"
            "Please save your slim checkpoints as {'state_dict': ..., 'layer_meta': ...}."
        )
    # Case 3: nn.Module
    elif isinstance(obj, torch.nn.Module):
        raise ValueError(
            f"Got a full nn.Module in {slim_ckpt}, but no explicit layer_meta.\n"
            "For reproducible HF export, save as a dict with keys 'state_dict' and 'layer_meta'."
        )
    else:
        raise TypeError(
            f"Unexpected object type in slim_ckpt: {type(obj)}. "
            "Expected dict with 'layer_meta' and 'state_dict'."
        )

    # Build config: either from file or from base HF model
    if "config" in obj and isinstance(obj["config"], LlamaConfig):
        cfg_slim = obj["config"]
    elif "config" in obj and isinstance(obj["config"], dict):
        cfg_slim = LlamaConfig.from_dict(obj["config"])
    else:
        cfg_slim = LlamaConfig.from_pretrained(base_id)

    # Inject pruning metadata into config for HF
    cfg_slim.layer_meta = layer_meta

    # Optional: override architectures and auto_map to SlimLlamaForCausalLM
    # NOTE: module path should match how you expose SlimLlamaForCausalLM in the HF repo
    cfg_slim.architectures = ["SlimLlamaForCausalLM"]
    cfg_slim.auto_map = {
        "AutoModelForCausalLM": "slim_llama.SlimLlamaForCausalLM"
    }

    model = SlimLlamaForCausalLM(cfg_slim, layer_meta=layer_meta)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[slim] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")

    return model
