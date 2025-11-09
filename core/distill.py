"""Knowledge-distillation utilities (model-family agnostic).

This module provides:
  - Losses: KL distillation, soft cross-entropy, cosine feature loss
  - Helper to obtain logits from models with/without built-in heads
  - Lightweight classification head for backbone models (e.g., ViTModel)
  - Simple evaluators (agreement %, KL) and diagnostics

Adapters may override `adapter_get_logits(model, x)` if a family needs a
custom extraction (e.g., language models with past_key_values).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass
class KDConfig:
    temperature: float = 2.0
    alpha: float = 1.0  # multiplier for KL term; task loss handled outside


# -----------------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------------

def kl_divergence(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float = 2.0) -> torch.Tensor:
    """Batchmean KL(student/ T || teacher/ T) scaled by T^2 (Hinton-style)."""
    p_s = F.log_softmax(student_logits / T, dim=-1)
    p_t = F.softmax(teacher_logits / T, dim=-1)
    return F.kl_div(p_s, p_t, reduction="batchmean") * (T * T)

def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, cfg: KDConfig) -> torch.Tensor:
    return cfg.alpha * kl_divergence(student_logits, teacher_logits, T=cfg.temperature)

def mse_reg(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float = 2.0) -> torch.Tensor:
    mse = F.mse_loss(student_logits,teacher_logits, reduction="mean")
    return mse * (T * T)
    
def soft_ce(student_logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    """Soft cross-entropy: expects `soft_targets` already normalized."""
    logp = F.log_softmax(student_logits, dim=-1)
    return -(soft_targets * logp).sum(dim=-1).mean()

def cosine_feature_loss(student_feats: torch.Tensor, teacher_feats: torch.Tensor) -> torch.Tensor:
    """1 - cosine similarity averaged over batch and time/patch dims."""
    s = F.normalize(student_feats, dim=-1)
    t = F.normalize(teacher_feats, dim=-1)
    return (1.0 - (s * t).sum(dim=-1)).mean()



# -----------------------------------------------------------------------------
# Logit extraction
# -----------------------------------------------------------------------------

class LogitsProvider(Protocol):
    def __call__(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor: ...


class ClsHead(nn.Module):
    """Minimal classification head: LN + Linear.

    Useful when the backbone outputs hidden states (e.g., ViTModel) and you
    want logits comparable to a teacher with a classification head.
    """

    def __init__(self, hidden_size: int, num_classes: int = 1000, base_head: Optional[nn.Module] = None):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        if base_head is not None:
            # Try to load weights if shapes match (e.g., from HF classifier)
            try:
                self.load_state_dict(base_head.state_dict(), strict=False)
            except Exception:
                pass

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        return self.fc(self.norm(cls_token))


@torch.no_grad()
def infer_hidden_size(model: nn.Module, sample: torch.Tensor) -> int:
    # Run a tiny forward to inspect hidden size when unknown
    model.eval()
    out = model(pixel_values=sample)
    if hasattr(out, "last_hidden_state"):
        return int(out.last_hidden_state.shape[-1])
    if hasattr(out, "logits"):
        return int(out.logits.shape[-1])
    raise RuntimeError("Cannot infer hidden size; provide explicitly.")


def default_get_logits(model: nn.Module, x: torch.Tensor, *, head: Optional[nn.Module] = None) -> torch.Tensor:
    """Family-agnostic logits extractor.

    - If model output has `.logits`, return it.
    - Else expects `.last_hidden_state` and uses [CLS] via provided `head`.
    """
    out = model(pixel_values=x)
    if hasattr(out, "logits"):
        return out.logits
    if hasattr(out, "last_hidden_state"):
        if head is None:
            raise ValueError("Backbone returned hidden states; supply a classification head.")
        cls_tok = out.last_hidden_state[:, 0, :]
        return head(cls_tok)
    raise ValueError("Model output lacks logits and last_hidden_state.")


# -----------------------------------------------------------------------------
# Evaluators & diagnostics
# -----------------------------------------------------------------------------

@torch.inference_mode()
def logits_std(model: nn.Module, loader, *, get_logits: LogitsProvider, batches: int = 10, device: str = "cuda") -> Tuple[float, int]:
    s = 0.0
    k = 0
    for x in loader:
        if k >= batches:
            break
        x = x.to(device)
        y = get_logits(model, x)
        s += y.std().item()
        k += 1
    return (s / max(1, k), k)


@torch.inference_mode()
def agreement_metrics(
    student: nn.Module,
    teacher: nn.Module,
    loader,
    *,
    get_student_logits: LogitsProvider,
    get_teacher_logits: LogitsProvider,
    batches: int = 20,
    T: float = 1.0,
    device: str = "cuda",
) -> dict:
    kl_sum = 0.0
    n = 0
    top1 = 0
    tot = 0
    for i, x in enumerate(loader):
        if i >= batches:
            break
        x = x.to(device)
        t = get_teacher_logits(teacher, x)
        s = get_student_logits(student, x)
        p_s = F.log_softmax(s / T, dim=-1)
        p_t = F.softmax(t / T, dim=-1)
        kl_sum += (F.kl_div(p_s, p_t, reduction="batchmean") * (T * T)).item()
        top1 += (s.argmax(-1) == t.argmax(-1)).sum().item()
        tot += x.size(0)
        n += 1
    return {"kl_TT": kl_sum / max(1, n), "top1_agreement": top1 / max(1, tot)}


# -----------------------------------------------------------------------------
# Small trainer helpers
# -----------------------------------------------------------------------------

class DualEMA:
    """Simple exponential moving average for a scalar (e.g., lambda or latency)."""

    def __init__(self, beta: float = 0.9, value: float = 0.0):
        self.beta = float(beta)
        self.value = float(value)

    def update(self, x: float) -> float:
        self.value = self.beta * self.value + (1 - self.beta) * float(x)
        return self.value
