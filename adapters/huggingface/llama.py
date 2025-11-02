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

from dataclasses import dataclass
from typing import Optional, Sequence, Callable, Tuple

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
          - Read gates to choose Q heads; slice q_proj rows and o_proj cols
          - Snap kept heads to an LCM of (policy multiple, Hkv)
          - Slice SwiGLU up/gate/down by groups
          - Unwrap back to plain HF modules; update metadata
        """
        # Accept either CoreExportPolicy with per-axis rounding or family policy
        if isinstance(policy, LlamaExportPolicy):
            head_rounding = policy.head_rounding
            ffn_rounding = policy.ffn_rounding
            warmup_steps = policy.warmup_steps
        else:
            head_rounding = getattr(policy, "rounding", None)
            ffn_rounding = getattr(policy, "rounding", None)
            warmup_steps = int(getattr(policy, "warmup_steps", 0))

        slim = deepcopy_eval_cpu(model_with_gates)
        core = getattr(slim, "model", slim)
        layers = getattr(core, "layers", None)
        if layers is None:
            return slim

        warm = (step < warmup_steps)

        def _lcm(a: int, b: int) -> int:
            return abs(a * b) // math.gcd(max(a, 1), max(b, 1)) if a > 0 and b > 0 else max(a, b, 1)

        for li, layer in enumerate(layers):
            # ---- Attention (Q heads) ----
            attn = layer.self_attn
            if isinstance(attn, GatedSelfAttentionLLM):
                gat = attn
                base = gat.base_attn

                Hq  = int(gat.num_q_heads)
                Hkv = int(gat.num_kv_heads)
                Dh  = int(gat.head_dim)

                if warm:
                    keep_idx = torch.arange(Hq)
                else:
                    # Build a "per-head" proxy gate if base gate is per-channel.
                    base_logits = gat.head_gate.logits.detach().float().view(-1)
                    tau = float(getattr(gat.head_gate, "tau", 1.0))

                    if base_logits.numel() == Hq:
                        # Native per-head gate: use as-is
                        proxy_gate = gat.head_gate
                        keep_idx = keep_group_indices_from_gate(
                            proxy_gate, policy=policy, step=step, custom_rounding=head_rounding
                        )
                    elif base_logits.numel() == Hq * Dh:
                        # Collapse per-channel → per-head (mean; or use .amax for stricter)
                        per_head_logits = base_logits.view(Hq, Dh).mean(dim=1)

                        class _PerHeadProxyGate:
                            def __init__(self, logits, tau):
                                self.logits = logits
                                self.tau = tau
                                self.num_groups = logits.numel()
                                self.group_size = 1

                        proxy_gate = _PerHeadProxyGate(per_head_logits, tau)
                        keep_idx = keep_group_indices_from_gate(
                            proxy_gate, policy=policy, step=step, custom_rounding=head_rounding
                        )
                    else:
                        raise RuntimeError(
                            f"Unexpected HeadGate logits len {base_logits.numel()} vs H={Hq} or H*Dh={Hq*Dh}"
                        )

                    # Enforce LCM with GQA (Hkv) via truncation to floor-multiple
                    def _lcm(a: int, b: int) -> int:
                        import math
                        return abs(a * b) // math.gcd(max(a, 1), max(b, 1)) if a > 0 and b > 0 else max(a, b, 1)

                    pol_mult = getattr(head_rounding, "multiple_groups", 1)
                    snap = _lcm(int(pol_mult), max(1, Hkv))
                    if keep_idx.numel() % snap != 0:
                        k = (keep_idx.numel() // snap) * snap
                        k = max(snap, min(Hq, k))
                        # recompute top-k by per-head logits (ensure same criterion used above)
                        if base_logits.numel() == Hq * Dh:
                            scores = per_head_logits
                        else:
                            scores = base_logits
                        keep_idx = torch.topk(scores, k=k, largest=True).indices.sort().values


                H_keep = int(keep_idx.numel())
                # channels for q/o slicing
                ch_idx = torch.cat([torch.arange(h * Dh, (h + 1) * Dh) for h in keep_idx]).long()

                # slice wrapper linears
                gat.q_proj = slice_linear(gat.q_proj, keep_out=ch_idx)
                gat.o_proj = slice_linear(gat.o_proj, keep_in=ch_idx)

                # transplant into a clean HF attention
                new_attn = copy.deepcopy(base)
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
                if hasattr(core.config, "hidden_size"):
                    core.config.hidden_size = int(H_keep * Dh)

                layer.self_attn = new_attn  # unwrap

            # ---- MLP (SwiGLU grouped) ----
            mlp = layer.mlp
            g = getattr(mlp, "neuron_gate", None)
            if g is not None:
                grp_idx = keep_group_indices_from_gate(
                    g, policy=policy, step=step, custom_rounding=ffn_rounding,
                )
                group = int(g.group_size)  # GroupGate exposes group_size
                keep_exp = torch.cat([torch.arange(i * group, (i + 1) * group) for i in grp_idx]).long()

                mlp.up_proj   = slice_linear(mlp.up_proj,   keep_out=keep_exp)
                mlp.gate_proj = slice_linear(mlp.gate_proj, keep_out=keep_exp)
                mlp.down_proj = slice_linear(mlp.down_proj, keep_in=keep_exp)

                # Restore clean forward & drop gate
                if hasattr(mlp, "_orig_forward"):
                    mlp.forward = mlp._orig_forward
                    delattr(mlp, "_orig_forward")
                if hasattr(mlp, "neuron_gate"):
                    delattr(mlp, "neuron_gate")

        return slim


# -------------------------------------------------------------------------
# Export policy (allow different rounding for Heads vs FFN)
# -------------------------------------------------------------------------

@dataclass
class LlamaExportPolicy:
    warmup_steps: int = 0
    head_rounding: CoreRounding = CoreRounding()  # e.g., CoreRounding(floor=8, multiple=8)
    ffn_rounding:  CoreRounding = CoreRounding()  # e.g., CoreRounding(min_keep_ratio=0.8, multiple=32)


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
