# core/finetune.py
"""Post-pruning fine-tuning utilities (distillation)."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Iterable

import torch
import torch.nn as nn

from core.distill import KDConfig, kd_loss, mse_reg
from core.utils import ensure_trainable_parameters


@dataclass
class FinetuneConfig:
    epochs: int = 5
    lr: float = 3e-4
    wd: float = 0.0
    kd: KDConfig = KDConfig(temperature=2.0, alpha=1.0)
    amp: bool = True
    # "auto" -> bf16 if supported else fp16; "bf16" | "fp16" | "off" also allowed
    amp_dtype: str = "auto"
    device: str = "cuda"
    log_every: int = 50
    # diagnostics
    grad_check_every: int = 50
    grad_warn_if_zero_steps: int = 2   # consecutive checks with zero grad -> warn
    mse_weight: float = 0.0


def _autocast_and_scaler(amp: bool, amp_dtype: str) -> Tuple[torch.autocast, Optional[torch.amp.GradScaler], bool, str]:
    """
    Returns (autocast_ctx, scaler_or_None, use_scaler_bool, amp_mode_str)
      - BF16 -> autocast(bfloat16), NO GradScaler
      - FP16 -> autocast(float16),  GradScaler ENABLED
      - OFF  -> disabled autocast,  NO GradScaler
    """
    if not amp or amp_dtype == "off":
        ctx = torch.amp.autocast(device_type="cuda", enabled=False)
        return ctx, None, False, "OFF"

    if amp_dtype == "auto":
        use_bf16 = torch.cuda.is_bf16_supported()
    elif amp_dtype == "bf16":
        use_bf16 = True
    elif amp_dtype == "fp16":
        use_bf16 = False
    else:
        raise ValueError(f"Unknown amp_dtype={amp_dtype!r}")

    if use_bf16:
        ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
        return ctx, None, False, "BF16"
    else:
        ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        except TypeError:
            scaler = torch.cuda.amp.GradScaler(enabled=True)
        return ctx, scaler, True, "FP16"


def _images_from_batch(batch):
    if isinstance(batch, dict):
        return batch.get("pixel_values") or batch.get("input")
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


def _param_iter_trainable(model: nn.Module) -> Iterable[torch.nn.Parameter]:
    for p in model.parameters():
        if p.requires_grad:
            yield p


def _grad_norm_and_nonzero(params: Iterable[torch.nn.Parameter]) -> Tuple[float, int]:
    total_sq, nonzero = 0.0, 0
    for p in params:
        g = p.grad
        if g is None:
            continue
        if g.is_sparse:
            g = g.coalesce().values()
        gn = float(g.detach().norm().cpu())
        if gn > 0.0:
            nonzero += 1
        total_sq += gn * gn
    return (total_sq ** 0.5), nonzero

@torch.no_grad()
def recalibrate_bn_stats(model, loader, max_batches=200, device="cuda"):
    model.train()  # use training mode to update running stats
    seen = 0
    for i, batch in enumerate(loader):
        if i >= max_batches: break
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        if not torch.is_tensor(x): continue
        x = x.to(device, non_blocking=True)
        model(x)
        seen += x.size(0)
    return seen


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
    """Fine-tune a pruned student against a frozen teacher using KD."""
    dev = cfg.device
    student = student.to(dev)
    teacher = teacher.to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    for p in student.parameters():
        p.requires_grad_(True)
    
    # Make sure we can actually train
    ensure_trainable_parameters(student, requires_grad=True)
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    if trainable == 0:
        raise RuntimeError("No trainable parameters in student — cannot finetune.")

    opt = torch.optim.AdamW(
        _param_iter_trainable(student),
        lr=cfg.lr,
        weight_decay=cfg.wd,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs*len(train_loader), eta_min=3e-5)


    autocast_ctx, scaler, use_scaler, amp_mode = _autocast_and_scaler(cfg.amp, cfg.amp_dtype)
    print(f"[AMP] Mode={amp_mode} | GradScaler={'ON' if use_scaler else 'OFF'} | "
          f"KD: T={cfg.kd.temperature} alpha={cfg.kd.alpha} | LR={cfg.lr} WD={cfg.wd} | Trainable params={trainable:,}")

    zero_grad_streak = 0
    global_step = 0

    T_max = cfg.kd.temperature
    T_min = 2.0
    kd_conf = cfg.kd

    for ep in range(cfg.epochs):
        student.train()
        running, seen = 0.0, 0

        for i, batch in enumerate(train_loader):

            step = ep*len(train_loader) + i # global step for T scheduling
            max_steps = cfg.epochs*len(train_loader)
            kd_conf.temperature = T_max - (step/max_steps)*(T_max - T_min)

            # print(f"Step {step}/{max_steps}, T_min={T_min}, T={kd_conf.temperature}, T_max={T_max}")
            
            x = _images_from_batch(batch)
            if not torch.is_tensor(x):
                raise ValueError("Train loader must yield tensors or (tensor, target) tuples.")
            x = x.to(dev, non_blocking=True)

            with torch.no_grad():
                t = get_teacher_logits(teacher, x)
                # Force numerically stable dtype for the loss
                t = t.float()

            # ---- forward student under autocast
            with autocast_ctx:
                s = get_student_logits(student, x)

            # ---- compute KD loss in FP32 (outside autocast) for stability
            s32 = s.float()
            mse = cfg.mse_weight*mse_reg(s32, t, kd_conf.temperature)
            loss = kd_loss(s32, t, kd_conf) + mse

            opt.zero_grad(set_to_none=True)
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            # ---- diagnostics
            bs = x.size(0)
            running += float(loss.detach()) * bs
            seen += bs
            global_step += 1

            if cfg.grad_check_every and (global_step % cfg.grad_check_every == 0):
                gnorm, n_nonzero = _grad_norm_and_nonzero(_param_iter_trainable(student))
                if n_nonzero == 0 or gnorm == 0.0:
                    zero_grad_streak += 1
                    if zero_grad_streak >= cfg.grad_warn_if_zero_steps:
                        print(f"[WARN] Step {global_step}: zero gradients detected "
                              f"(nonzero={n_nonzero}, grad_norm={gnorm:.3e}). "
                              f"Check get_student_logits, requires_grad, AMP settings, and data pipeline.")
                else:
                    zero_grad_streak = 0

            if cfg.log_every and (i + 1) % cfg.log_every == 0:
                print(f"Step {i+1}/{len(train_loader)} (ep {ep+1}/{cfg.epochs}): "
                      f"running loss = {running / max(1, seen):.4f}")

            if on_step is not None:
                on_step(global_step, float(loss.detach()))

            # free ASAP
            del s, s32, t, loss

        # ---- validation
        if val_loader is not None:
            _ = recalibrate_bn_stats(student, train_loader, max_batches=1000, device=cfg.device)
            student.eval()            
            val_loss, vseen = 0.0, 0
            with torch.no_grad():
                for vbatch in val_loader:
                    vx = _images_from_batch(vbatch)
                    if not torch.is_tensor(vx):
                        raise ValueError("Val loader must yield tensors or (tensor, target) tuples.")
                    vx = vx.to(dev, non_blocking=True)

                    vt = get_teacher_logits(teacher, vx).float()
                    with autocast_ctx:
                        vs = get_student_logits(student, vx)
                    vloss = kd_loss(vs.float(), vt, kd_conf)
                    val_loss += float(vloss.detach()) * vx.size(0)
                    vseen += vx.size(0)
            print(f"Epoch {ep+1}/{cfg.epochs}: T={kd_conf.temperature:.2f}, train={running / max(1, seen):.6f}, "
                  f"val={val_loss / max(1, vseen):.6f}")
        else:
            print(f"Epoch {ep+1}/{cfg.epochs}: train={running / max(1, seen):.6f}")

        scheduler.step()
        
    student.eval()
    return student
