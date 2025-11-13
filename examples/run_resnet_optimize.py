"""ResNet-18 optimization aligned with the generic core/train.py Lagrange trainer.

Flow:
  1) Build student (torchvision resnet18 + BN gates via adapter) and teacher (resnet50).
  2) Calibrate proxy on keep-all.
  3) Train with LagrangeTrainer (weights pass + gates pass, λ on latency).
  4) Kernel-aware export grid search over BN-group multiples.
  5) BN recalibration and KD fine-tuning on the slim.

Usage:
  python -m examples.run_resnet_optimize \
    --recipe recipes/resnet18_imagenet224.yaml \
    --epochs 2 \
    --outdir runs/resnet18
"""
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Sequence

import torch
import torch.nn as nn

from adapters.torchvision.resnet import ResNetAdapter, ResNetExportPolicy
from core.export import Rounding as CoreRounding
from core.proxy_cost import ResNetLatencyProxy
from core.profiler import measure_latency_ms
from core.train import LagrangeTrainer, TrainerConfig
from core.distill import KDConfig
from core.finetune import FinetuneConfig, finetune_student, recalibrate_bn_stats
from data.vision import build_imagenet_like_loaders


# ------------------------------ Utils ------------------------------

def load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _images_from_batch(batch):
    # (images, labels) or [images, labels]
    if isinstance(batch, (tuple, list)):
        return batch[0]
    # dict-style datasets
    if isinstance(batch, dict):
        for k in ("pixel_values", "images", "inputs"):
            v = batch.get(k, None)
            if torch.is_tensor(v):
                return v
        # fallback to first tensor value
        for v in batch.values():
            if torch.is_tensor(v):
                return v
        raise TypeError("Batch dict has no tensor-like image field")
    # plain tensor
    if torch.is_tensor(batch):
        return batch
    raise TypeError(f"Unsupported batch type for images: {type(batch)}")





# ------------------------------ Build pack ------------------------------

def build_from_recipe(path: str):
    cfg = load_yaml(path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Student & Teacher ---
    mcfg = cfg.get("model", {})
    gcfg = cfg.get("gates", {})

    student = ResNetAdapter.attach_gates(
        name=mcfg.get("name", "resnet18"),
        pretrained=bool(mcfg.get("pretrained", True)),
        group_size=int(gcfg.get("group_size", 16)),
        tau=float(gcfg.get("tau", 1.5)),
        init_logit=float(gcfg.get("init_logit", 3.0)),
        hard_eval=bool(gcfg.get("hard_eval", True)),
        num_classes=int(mcfg.get("num_classes", 1000)),
    ).to(device)

    # Teacher: use torchvision resnet50
    import torchvision
    teacher = torchvision.models.resnet50(weights="IMAGENET1K_V2").eval().to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)

    # --- Data ---
    dcfg = cfg.get("data", {})

    # dcfg["limit_train"] = 1000
    # dcfg["limit_val"] = 1000
    train_loader, val_loader = build_imagenet_like_loaders(dcfg)

    # --- Proxy calibration ---
    proxy = ResNetLatencyProxy()
    B = int(dcfg.get("batch_size", 64))
    img_size = int(dcfg.get("img_size", 224))
    base_ms = proxy.calibrate(student, keepall_export_fn=ResNetAdapter.export_keepall,
                              profiler_fn=measure_latency_ms, sample=torch.randn((B, 3, img_size, img_size),device=device), device=device)

    # --- Trainer config ---
    kd = KDConfig(**cfg.get("trainer", {}).get("kd", {}))

    trcfg = TrainerConfig(
        kd=kd,
        amp=bool(cfg.get("trainer", {}).get("amp", True)),
        use_grad_scaler=bool(cfg.get("trainer", {}).get("use_grad_scaler", True)),
        gate_warmup_steps = int(cfg.get("trainer", {}).get("gate_warmup_steps", 0)),
        mse_weight = float(cfg.get("trainer", {}).get("mse_weight", 0.0)),
        early_stopping_patience = int(cfg.get("trainer", {}).get("early_stopping_patience", 0)),
        early_stopping_lambda = float(cfg.get("trainer", {}).get("early_stopping_lambda", 1e-4)),
        device=device,
    )

    # latency target
    tau_scale = float(cfg.get("trainer", {}).get("lagrange", {}).get("tau_target_scale", 0.7))
    trcfg.latency_target_ms = tau_scale * float(base_ms)

    # Optional knobs mapped into TrainerConfig
    trcfg.real_probe_every = int(cfg.get("trainer", {}).get("lagrange", {}).get("real_every", 10))

    # Export policy used during periodic probes
    export_policy = ResNetExportPolicy(
        warmup_steps=0,
        rounding=CoreRounding(
            floor_groups=1,
            multiple_groups=int(cfg.get("trainer", {}).get("lagrange", {}).get("multiple_groups_probe", 1)),
            min_keep_ratio=float(cfg.get("trainer", {}).get("lagrange", {}).get("min_keep_ratio", 0.25)),
        ),
        min_keep_ratio=float(cfg.get("trainer", {}).get("lagrange", {}).get("min_keep_ratio", 0.25)),
    ) 

    pack = {
        "student": student,
        "teacher": teacher,
        "device": device,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "proxy": proxy,
        "trainer_cfg": trcfg,
        "export_policy": export_policy,
        "recipe": cfg,
        "img_size": img_size,
        "batch_size": B,
        "base_ms": float(base_ms),
    }
    return pack


# ------------------------------ Export grid search ------------------------------

def grid_search_export(student: nn.Module, *, device: str, img_size: int, B: int,
                        multiples: Sequence[int], min_keep_ratio: float = 0.0):
    trials = []
    best = None
    for i, M in enumerate(multiples):
        policy = ResNetExportPolicy(
            warmup_steps=0,
            rounding=CoreRounding(floor_groups=1, multiple_groups=int(M), min_keep_ratio=min_keep_ratio),
            min_keep_ratio=min_keep_ratio,
        )
        slim = ResNetAdapter.export_pruned(student, policy, step=9999).to(device)
        mean_ms, p95_ms = measure_latency_ms(slim, (B, 3, img_size, img_size), device=device)
        rec = {"multiple_groups": int(M), "mean_ms": float(mean_ms), "p95_ms": float(p95_ms)}
        trials.append(rec)
        if (best is None) or (rec["mean_ms"] < best["mean_ms"]):
            best = {"model": slim, "params": {"multiple_groups": int(M)}, "mean_ms": float(mean_ms), "p95_ms": float(p95_ms)}
        print(f"[{i}/{len(multiples)}] multiple_groups={M} | mean_ms={mean_ms:.3f}")
    return {"best_model": best["model"], "best_params": best["params"], "trials": trials}


# ------------------------------ Main ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--outdir", type=str, default="runs/resnet18")
    ap.add_argument("--slim", type=str, default=None)
    ap.add_argument("--finetune", type=bool, default=True)
    
    args = ap.parse_args()

    print(args)

    os.makedirs(args.outdir, exist_ok=True)
    pack = build_from_recipe(args.recipe)

    student: nn.Module = pack["student"]
    teacher: nn.Module = pack["teacher"]
    device: str = pack["device"]
    train_loader = pack["train_loader"]
    val_loader = pack["val_loader"]
    proxy = pack["proxy"]  # ProxyShim
    trcfg: TrainerConfig = pack["trainer_cfg"]
    export_policy = pack["export_policy"]

    if args.slim is None:
        # Build trainer with adapter callables (aligned with core/train.py)
        trainer = LagrangeTrainer(
            student=student,
            teacher=teacher,
            proxy=proxy,  
            adapter_get_student_logits=lambda m, batch: m(_images_from_batch(batch)),
            adapter_get_teacher_logits=lambda m, batch: m(_images_from_batch(batch)).detach(),
            adapter_export_keepall=ResNetAdapter.export_keepall,
            adapter_export_pruned=lambda m, pol, step: ResNetAdapter.export_pruned(m, pol, step),
            export_policy=export_policy,
            cfg=trcfg,
        )
    
        # --- Train ---
        lambdas = []
        for ep in range(args.epochs):
            print(f"=== Epoch {ep+1}/{args.epochs} ===")
            lam = trainer.train_epoch(train_loader)
            lambdas.append(lam)

            last = lambdas[:trcfg.early_stopping_patience]
            last = [x for x in last if x < trcfg.early_stopping_lambda]
            if len(last) == trcfg.early_stopping_patience:
                print(f"Early stopping: lambda < {trcfg.early_stopping_lambda} for last {trcfg.early_stopping_patience} epochs")
                break

                
        out_path = os.path.join(args.outdir, "resnet18_gated.pth")
        torch.save(student, out_path)
        print("Saved gated model to", out_path)
        

        # --- Export search (kernel-aware) ---
        print("Running export grid search...")
        multiples = pack["recipe"].get("export", {}).get("grid_multiple_groups", [1, 2, 4, 8, 16])
        res = grid_search_export(
            student,
            device=device,
            img_size=pack["img_size"],
            B=pack["batch_size"],
            multiples=multiples,
            min_keep_ratio=float(pack["recipe"].get("trainer", {}).get("lagrange", {}).get("min_keep_ratio", 0.25)),
        )
    
        slim = res["best_model"]
        print("Best export params:", res["best_params"])
        for rec in sorted(res["trials"], key=lambda r: r["mean_ms"])[:5]:
            print("  trial:", rec)
    
        # --- BN recalibration ---
        print("Recalibrating BN stats on the slim model...")
        ResNetAdapter.bn_recalibration(slim, train_loader, num_batches=1000, device=device)

        out_path = os.path.join(args.outdir, "resnet18_slim.pth")
        torch.save(slim, out_path)
        print("Saved pruned model to", out_path)
        
    else:
        slim = torch.load(args.slim, map_location=device, weights_only=False).to(device)
        print(f"Skipping training. Pruned model loaded from {args.slim}. [To run with training, drop '--slim' argument]")


        
    if args.finetune is True:
        # --- Fine-tune the slim model with KD ---
        ft_epochs = int(pack["recipe"].get("finetune", {}).get("epochs", 10))
        print(f"\nStarting fine tuning for {ft_epochs} epochs...")
        ft_cfg = FinetuneConfig(
            epochs=ft_epochs,
            lr=float(pack["recipe"].get("finetune", {}).get("lr", 3e-4)),
            wd=float(pack["recipe"].get("finetune", {}).get("wd", 1e-5)),
            kd=KDConfig(**pack["recipe"].get("trainer", {}).get("kd", {})),
            amp=bool(pack["recipe"].get("trainer", {}).get("amp", True)),
            mse_weight = float(pack["recipe"].get("trainer", {}).get("mse_weight", 0.0)),
            device=device,
            log_every=50,
        )
        
        recalibrate_bn_stats(slim, train_loader, max_batches=1000)
        slim = finetune_student(
            slim,
            teacher,
            train_loader,
            get_student_logits=lambda m, batch: m(_images_from_batch(batch)),
            get_teacher_logits=lambda m, batch: m(_images_from_batch(batch)).detach(),
            cfg=ft_cfg,
            val_loader=val_loader,
        )

        recalibrate_bn_stats(slim, train_loader, max_batches=1000)

        out_path = os.path.join(args.outdir, "resnet18_slim.pth")
        torch.save(slim, out_path)
        print("Saved pruned model to", out_path)
    
    else:
        print("Skipping fine-tuning. [To run fine-tuning, set '--finetuning True' or omit this argument]")

    # --- Benchmark & save ---
    
    B = pack["batch_size"]; H = W = pack["img_size"]
    print(f"Starting benchmarking with batch size = {B}...")
    mean_keep, p95_keep = measure_latency_ms(ResNetAdapter.export_keepall(student), (B, 3, H, W), device=device)
    mean_slim, p95_slim = measure_latency_ms(slim, (B, 3, H, W), device=device)
    print(f"Base: mean={mean_keep:.3f}ms p95={p95_keep:.3f}ms")
    print(f"Slim: mean={mean_slim:.3f}ms p95={p95_slim:.3f}ms\n")
    if mean_keep > 0:
        print(f"Speedup={100.0*(mean_keep-mean_slim)/mean_keep:.2f}%")




if __name__ == "__main__":
    main()
