"""Generic Lagrangian trainer (family-agnostic).

This module provides a light framework to optimize *gated* students against
teachers with a latency target enforced via a proxy + optional real probes.

It does not assume ViT/ResNet/LLM specifics; adapters provide tiny callables.

Key ingredients:
  - Two-phase update per step: (A) weights w.r.t. KD/task, (B) gates w.r.t. KD +
    sparsity + latency penalty with a dual variable λ.
  - Optional periodic export + real-latency probe to correct λ.
  - Constraint projection for gates after each step.

Adapters must provide:
  - get_student_logits(model, x) -> Tensor
  - get_teacher_logits(model, x) -> Tensor
  - export_keepall(model) -> nn.Module (clean copy without gates)
  - export_pruned(model, policy, step) -> nn.Module (transient copy for profiling)

Core modules used:
  - `distill.KDConfig`, `distill.kd_loss`
  - `gates.combined_penalty`, `gates.PenaltyWeights`, `gates.project_gates_into_constraints`
  - `proxy_cost.LatencyProxy`
  - `profiler.measure_latency_ms`
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional
import gc

import torch
import torch.nn as nn

from .distill import KDConfig, kd_loss
from .gates import PenaltyWeights, Constraints, combined_penalty, project_gates_into_constraints, collect_param_groups
from .proxy_cost import LatencyProxy
from .profiler import measure_latency_ms


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass
class DualConfig:
    lr: float = 0.05     # step for λ update
    ema_beta: float = 0.5  # blend proxy-driven λ and real probe λ
    clip: float = 10.0


@dataclass
class TrainerConfig:
    kd: KDConfig = KDConfig()
    penalties: PenaltyWeights = PenaltyWeights(l0=0.0, keep_floor_ratio=0.0, bimodality=0.0)
    constraints: Constraints = Constraints(min_keep_ratio=0.0, min_groups=1, max_groups_drop=None)

    latency_target_ms: float = 30.0
    real_probe_every: int = 0        # steps; 0 disables real probes
    probe_batch_override: Optional[int] = None

    amp: bool = True
    device: str = "cuda"

    # Optimizers
    lr_gate: float = 1e-2
    lr_linear: float = 1e-4
    lr_affine: float = 3e-4
    wd_linear: float = 1e-4

    # Mixed precision scaler
    use_grad_scaler: bool = True

    # Dual update
    dual: DualConfig = DualConfig()


# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------

class LagrangeTrainer:
    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        proxy: LatencyProxy,
        *,
        adapter_get_student_logits: Callable[[nn.Module, torch.Tensor], torch.Tensor],
        adapter_get_teacher_logits: Callable[[nn.Module, torch.Tensor], torch.Tensor],
        adapter_export_keepall: Callable[[nn.Module], nn.Module],
        adapter_export_pruned: Callable[[nn.Module, object, int], nn.Module],
        export_policy: object,
        cfg: TrainerConfig,
    ) -> None:
        self.student = student
        self.teacher = teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.proxy = proxy
        self.get_s = adapter_get_student_logits
        self.get_t = adapter_get_teacher_logits
        self.export_keepall = adapter_export_keepall
        self.export_pruned = adapter_export_pruned
        self.export_policy = export_policy
        self.cfg = cfg

        # Build optimizers (grouped)
        param_groups = collect_param_groups(
            student,
            lr_gate=cfg.lr_gate,
            lr_linear=cfg.lr_linear,
            lr_affine=cfg.lr_affine,
            wd_linear=cfg.wd_linear,
        )
        # gates-only optimizer uses first group
        self.opt_g = torch.optim.Adam([param_groups[0]], lr=param_groups[0]["lr"])  # type: ignore[arg-type]
        # weights optimizer for the rest
        self.opt_w = torch.optim.Adam(param_groups[1:])

        self.scaler = torch.amp.GradScaler('cuda', enabled=(cfg.amp and cfg.use_grad_scaler))
        self.lambda_: float = 0.0

    # ---- internal helpers -----------------------------------------------------
    def _zero_grads(self, params):
        for p in params:
            if p.grad is not None:
                p.grad = None

    def _has_grad(self, params) -> bool:
        for p in params:
            if p.grad is not None:
                return True
        return False

    # ---- training -------------------------------------------------------------
    def train_epoch(self, loader, *, real_policy=None, verbose_every: int = 50):
        device = self.cfg.device
        self.student.train().to(device)
        self.teacher.to(device).eval()
    
        running = 0.0
        seen = 0
        lam_real = self.lambda_
    
        for step, batch in enumerate(loader, 1):
            # Move batch to device in a type-safe way
            batch = _move_batch_to_device(batch, device)

            with torch.inference_mode():
                t_logits = self.get_t(self.teacher, batch)  # [B,1,V]
            # match AMP compute dtype to avoid upcasting later
            if self.cfg.amp:
                # infer autocast dtype from student params (bf16 or fp16)
                sparam = next(self.student.parameters())
                t_logits = t_logits.to(dtype=sparam.dtype, non_blocking=True)
            
                
            # -------- Pass A: WEIGHTS (KD only) --------
            self.opt_w.zero_grad(set_to_none=True)
    
            with torch.amp.autocast('cuda', enabled=self.cfg.amp):
                # Adapters receive the batch object (dict/tuple/tensor)
                s_logits = self.get_s(self.student, batch)
                # with torch.no_grad():
                #     t_logits = self.get_t(self.teacher, batch)
                loss_w = kd_loss(s_logits, t_logits, self.cfg.kd)
    
            self.scaler.scale(loss_w).backward()
            # Prevent gate params from changing in pass A
            gate_params = self.opt_g.param_groups[0]["params"]
            self._zero_grads(gate_params)
    
            if any(p.grad is not None for pg in self.opt_w.param_groups for p in pg["params"]):
                self.scaler.step(self.opt_w)
                self.scaler.update()
            else:
                self.opt_w.zero_grad(set_to_none=True)

            del s_logits, loss_w
            gc.collect()
            torch.cuda.empty_cache()    
    
            # -------- Pass B: GATES (KD + sparsity + λ * gap) --------
            self.opt_g.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=self.cfg.amp):
                s_logits = self.get_s(self.student, batch)
                # with torch.no_grad():
                #     t_logits = self.get_t(self.teacher, batch)
                kd_g = kd_loss(s_logits, t_logits, self.cfg.kd)
    
                # Proxy gets the batch object too; family-specific proxy can read (B,S) etc.
                o1_ms = self.proxy.predict(self.student, batch)
                gap = torch.relu(o1_ms - float(self.cfg.latency_target_ms))
                reg = combined_penalty(self.student, self.cfg.penalties)
    
                loss_g = kd_g + _to_tensor(self.lambda_, o1_ms) * gap + reg
    
            self.scaler.scale(loss_g).backward()
            # Prevent non-gate params from changing in pass B
            for pg in self.opt_w.param_groups:
                self._zero_grads(pg["params"])
    
            if self._has_grad(self.opt_g.param_groups[0]["params"]):
                self.scaler.step(self.opt_g)
                self.scaler.update()
            else:
                self.opt_g.zero_grad(set_to_none=True)

            
    
            # -------- Dual (λ) update using proxy --------
            with torch.no_grad():
                lam_proxy = max(0.0, self.lambda_ + self.cfg.dual.lr * (float(o1_ms.detach()) - self.cfg.latency_target_ms))
                self.lambda_ = 0.5 * (lam_real + lam_proxy)
    
            # -------- Constraint projection, optional real probe --------
            project_gates_into_constraints(self.student, self.cfg.constraints)
    
            if self.cfg.real_probe_every and (step % int(self.cfg.real_probe_every) == 0):
                # Build a probe shape for latency func if needed
                try:
                    from core.measure import measure_latency_text_ms  # text-friendly
                    if isinstance(batch, dict) and "input_ids" in batch and torch.is_tensor(batch["input_ids"]):
                        B, S = int(batch["input_ids"].size(0)), int(batch["input_ids"].size(1))
                    else:
                        # Fallback: try tensor-like batch
                        x0 = batch["input_ids"] if isinstance(batch, dict) else (batch[0] if isinstance(batch, (tuple, list)) else batch)
                        B = int(x0.size(0)); S = int(x0.size(1))
                    slim = self.export_pruned(self.student, real_policy or self.export_policy, step)
                    mean_ms, p95_ms = measure_latency_text_ms(slim, B=B, S=S, T=128, device=device)
                except Exception:
                    # If the project has a different profiler, retain compatibility:
                    from .profiler import measure_latency_ms
                    x0 = batch["input_ids"] if isinstance(batch, dict) else (batch[0] if isinstance(batch, (tuple, list)) else batch)
                    shape = (int(x0.size(0)), *list(x0.shape[1:]))
                    slim = self.export_pruned(self.student, real_policy or self.export_policy, step)
                    mean_ms, p95_ms = measure_latency_ms(slim, shape, device=device)
    
                with torch.no_grad():
                    lam_real = max(0.0, self.lambda_ + self.cfg.dual.lr * (mean_ms - self.cfg.latency_target_ms))
    
                if (step % verbose_every) == 0:
                    print(
                        f"Step {step}/{len(loader)} | KL={float(loss_w.item()):.4f} | Gate={float(loss_g.item()):.4f} | "
                        f"proxy={float(o1_ms.detach()):.3f}ms | real_mean={mean_ms:.3f}ms p95={p95_ms:.3f}ms | λ={self.lambda_:.4f}"
                    )
    
            running += float(loss_g.detach())
            seen += _batch_size(batch)

            del s_logits, t_logits, o1_ms, kd_g, reg, loss_g
            torch.cuda.empty_cache()
            gc.collect()
    
        print(f"Epoch loss {running / max(1, seen):.6f}")
        return self.lambda_


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _to_tensor(val: float, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(val, device=like.device, dtype=like.dtype)

def _move_batch_to_device(batch, device: str):
    """
    Supports:
      - dict with keys 'input_ids' and optional 'attention_mask'
      - (x,) or (x, y) tuples/lists -> move each tensor-like to device
      - single Tensor
    Converts attention_mask to bool (preferred by HF SDPA).
    """
    if isinstance(batch, dict):
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                v = v.to(device, non_blocking=True)
                if k == "attention_mask" and v.dtype != torch.bool:
                    v = v.to(torch.bool)
            out[k] = v
        return out

    if isinstance(batch, (tuple, list)):
        moved = []
        for v in batch:
            if torch.is_tensor(v):
                v = v.to(device, non_blocking=True)
            moved.append(v)
        return type(batch)(moved)

    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)

    # Unknown type: return as-is (adapters/proxy should handle it)
    return batch


def _batch_size(batch) -> int:
    """Best-effort batch size for logging/averages."""
    if isinstance(batch, dict) and "input_ids" in batch and torch.is_tensor(batch["input_ids"]):
        return int(batch["input_ids"].size(0))
    if torch.is_tensor(batch):
        return int(batch.size(0))
    if isinstance(batch, (tuple, list)) and len(batch) and torch.is_tensor(batch[0]):
        return int(batch[0].size(0))
    return 1