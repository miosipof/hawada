"""Post-pruning fine-tuning utilities (distillation).

This module provides a light-weight KD fine-tuner to recover accuracy after
export/pruning. It uses the same KD loss as `core/distill.py` and stays
family-agnostic via adapter-provided logits getters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn

from core.distill import KDConfig, kd_loss

from core.utils import ensure_trainable_parameters


@dataclass
class FinetuneConfig:
    epochs: int = 5
    lr: float = 3e-4
    wd: float = 0.0
    kd: KDConfig = KDConfig(temperature=2.0, alpha=1.0)
    amp: bool = True
    device: str = "cuda"
    log_every: int = 50


def finetune_student(
    student: nn.Module,
    teacher: nn.Module,
    train_loader,
    *,
    get_student_logits: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    get_teacher_logits: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    cfg: FinetuneConfig = FinetuneConfig(),
    val_loader=None,
    on_step: Optional[Callable[[int, float], None]] = None,
) -> nn.Module:
    """Fine-tune a pruned student against a frozen teacher using KD.

    Returns the fine-tuned student (in .eval() mode).
    """
    device = cfg.device
    student = student.to(device).train()
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    ensure_trainable_parameters(student, requires_grad=True)

    opt = torch.optim.AdamW((p for p in student.parameters() if p.requires_grad), lr=cfg.lr, weight_decay=cfg.wd)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.amp)

    global_step = 0
    for ep in range(cfg.epochs):
        running, seen = 0.0, 0
        for i, batch in enumerate(train_loader):
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.to(device, non_blocking=True)
            x = x.clone()

            with torch.amp.autocast('cuda', enabled=cfg.amp):
                s = get_student_logits(student, x)
                with torch.no_grad():
                    t = get_teacher_logits(teacher, x)
                loss = kd_loss(s, t, cfg.kd)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += float(loss.detach())
            seen += x.size(0)
            global_step += 1

            if cfg.log_every and (i + 1) % cfg.log_every == 0:
                print(f"Step {i+1}/{len(train_loader)} (ep {ep+1}/{cfg.epochs}): running loss = {running/max(1, seen):.4f}")
            if on_step is not None:
                on_step(global_step, float(loss.detach()))

        # Validation (optional)
        if val_loader is not None:
            val_loss, vseen = 0.0, 0
            with torch.no_grad():
                for j, vbatch in enumerate(val_loader):
                    vx = vbatch[0] if isinstance(vbatch, (tuple, list)) else vbatch
                    vx = vx.to(device, non_blocking=True)
                    vx = vx.clone()
                    with torch.amp.autocast('cuda', enabled=cfg.amp):
                        vs = get_student_logits(student, vx)
                        vt = get_teacher_logits(teacher, vx)
                        vloss = kd_loss(vs, vt, cfg.kd)
                    val_loss += float(vloss)
                    vseen += vx.size(0)
            print(f"Epoch {ep+1}/{cfg.epochs}: train={running/max(1, seen):.6f}, val={val_loss/max(1, vseen):.6f}")
        else:
            print(f"Epoch {ep+1}/{cfg.epochs}: train={running/max(1, seen):.6f}")

    student.eval()
    return student
