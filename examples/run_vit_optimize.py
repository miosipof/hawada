"""Example: optimize google/vit-base-patch16-224 for a target GPU.

Usage:
  python -m examples.run_vit_optimize --recipe recipes/vit_base_patch16_224.yaml --epochs 2 --outdir runs/vit_base

This script wires together:
  - ViT adapter (gates & exporters)
  - ViT latency proxy + calibration
  - Generic Lagrangian trainer
  - ImageNet-like dataloaders (replace data paths in the recipe)

It saves a pruned student and prints latency before/after.
"""
from __future__ import annotations

import argparse
import os
from functools import partial

import torch
from transformers import ViTModel, ViTForImageClassification
import yaml

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

# Local imports (run from repo root)
from core.utils import set_seed
from core.proxy_cost import ViTLatencyProxy, ViTProxyConfig, calibrate_scale
from core.profiler import measure_latency_ms, ProfileSettings
from core.train import LagrangeTrainer, TrainerConfig
from core.distill import KDConfig, ClsHead
from core.export import ExportPolicy as CoreExportPolicy, Rounding as CoreRounding
from adapters.huggingface.vit import ViTAdapter, ViTGatingConfig, ViTExportPolicy, vit_search_best_export
from data.vision import build_imagenet_like_loaders, VisionDataConfig

from core.finetune import FinetuneConfig, finetune_student


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", type=str, default="recipes/vit_base_patch16_224.yaml")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--outdir", type=str, default="runs/vit_base")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--slim", type=str, default=None)
    ap.add_argument("--finetune", type=bool, default=True)    
    return ap.parse_args()


def build_from_recipe(recipe_path: str):
    with open(recipe_path, "r") as f:
        R = yaml.safe_load(f)

    # --- Models
    model_id = R["model"]
    device = R.get("trainer", {}).get("device", "cuda")

    student = ViTModel.from_pretrained(model_id)
    teacher = ViTModel.from_pretrained(model_id)
    base_head_model = ViTForImageClassification.from_pretrained(model_id)

    # Heads for logits when using ViTModel
    hidden = int(student.config.hidden_size)
    num_classes = int(base_head_model.config.num_labels)
    student_head = ClsHead(hidden_size=hidden, num_classes=num_classes, base_head=base_head_model.classifier).to(device)
    teacher_head = ClsHead(hidden_size=hidden, num_classes=num_classes, base_head=base_head_model.classifier).to(device)

    # --- Adapter & gates
    gate_cfg = ViTGatingConfig(**R.get("adapter", {}).get("vit_gating", {}))
    adapter = ViTAdapter(student)
    student = adapter.attach_gates(gate_cfg)

    # --- Data
    data_cfg = VisionDataConfig(**R.get("data", {}))
    train_loader, val_loader = build_imagenet_like_loaders(data_cfg)

    # --- Proxy + calibration
    proxy_cfg = ViTProxyConfig(**R.get("proxy", {}).get("vit", {}))
    proxy = ViTLatencyProxy(proxy_cfg)

    # Calibrate scale on keep-all student
    keepall = ViTAdapter.export_keepall(student)
    B = data_cfg.batch_size
    img_size = data_cfg.img_size
    calibrate_scale(proxy, keepall, (B, 3, img_size, img_size), measure_latency_ms, device=device)

    # --- Export policy
    round_cfg = CoreRounding(**R.get("export", {}).get("rounding", {}))

    # Final
    export_policy = CoreExportPolicy(
        warmup_steps=int(R.get("export", {}).get("warmup_steps", 0)), 
        rounding=round_cfg)
    
    # Eval during training
    probe_policy = CoreExportPolicy(
        warmup_steps=0,
        rounding=CoreRounding(floor_groups=1, multiple_groups=1, min_keep_ratio=0.0),
    )

    # --- Trainer config
    kd_cfg = KDConfig(**R.get("trainer", {}).get("kd", {}))
    pen = R.get("trainer", {}).get("penalties", {})
    cons = R.get("trainer", {}).get("constraints", {})
    tcfg = TrainerConfig(
        kd=kd_cfg,
        penalties=TrainerConfig.penalties.__class__(**pen),
        constraints=TrainerConfig.constraints.__class__(**cons),
        latency_target_ms=float(R.get("trainer", {}).get("latency_target_ms", 30.0)),
        real_probe_every=int(R.get("trainer", {}).get("real_probe_every", 0)),
        probe_batch_override=R.get("trainer", {}).get("probe_batch_override", None),
        amp=bool(R.get("trainer", {}).get("amp", True)),
        device=device,
        lr_gate=float(R.get("trainer", {}).get("lr_gate", 1e-2)),
        lr_linear=float(R.get("trainer", {}).get("lr_linear", 1e-4)),
        lr_affine=float(R.get("trainer", {}).get("lr_affine", 3e-4)),
        wd_linear=float(R.get("trainer", {}).get("wd_linear", 1e-4)),
    )

    # --- Adapter-specific logits providers
    get_s = lambda m, x: ViTAdapter.get_logits(m, x, head=student_head)
    get_t = lambda m, x: ViTAdapter.get_logits(m, x, head=teacher_head)

    return {
        "student": student,
        "teacher": teacher,
        "student_head": student_head,
        "teacher_head": teacher_head,
        "adapter": adapter,
        "export_policy": export_policy,
        "probe_policy": probe_policy,
        "proxy": proxy,
        "trainer_cfg": tcfg,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "get_s": get_s,
        "get_t": get_t,
        "device": device,
        "recipe": R,
    }


# Build a policy-factory so heads/FFN can use different roundings
def make_vit_policy(head_mult: int, ffn_snap: int):
    return ViTExportPolicy(
        warmup_steps=0,
        head_rounding=CoreRounding(floor_groups=1, multiple_groups=int(head_mult), min_keep_ratio=0.0),
        ffn_rounding=CoreRounding(floor_groups=1, multiple_groups=int(ffn_snap), min_keep_ratio=0.0),
    )


def main():
    args = parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)

    set_seed(args.seed)
    pack = build_from_recipe(args.recipe)

    student = pack["student"].to(pack["device"])  # type: ignore[index]
    teacher = pack["teacher"].to(pack["device"])  # type: ignore[index]
    proxy = pack["proxy"]  # type: ignore[index]
    export_policy = pack["export_policy"]  # type: ignore[index]
    probe_policy = pack["probe_policy"]  # type: ignore[index]

    if args.slim is None:
        trainer = LagrangeTrainer(
            student=student,
            teacher=teacher,
            proxy=proxy,
            adapter_get_student_logits=pack["get_s"],
            adapter_get_teacher_logits=pack["get_t"],
            adapter_export_keepall=ViTAdapter.export_keepall,
            adapter_export_pruned=ViTAdapter.export_pruned,
            export_policy=export_policy,
            cfg=pack["trainer_cfg"],
        )
    
        epochs = int(args.epochs)
        for ep in range(epochs):
            print(f"\n=== Epoch {ep+1}/{epochs} ===")
            trainer.train_epoch(pack["train_loader"], real_policy=probe_policy)
    
    
    
        # Run the grid search for the current num_heads
        num_heads = int(student.config.num_attention_heads)
        B = max(1, int(pack["recipe"]["data"]["batch_size"] // 4))
        img_size = pack["recipe"]["data"]["img_size"]
        search = vit_search_best_export(
            student,
            export_fn=ViTAdapter.export_pruned,
            num_heads=num_heads,
            step=9999,  # no warmup
            batch_shape=(B, 3, img_size, img_size),
            device=pack["device"],
            make_policy=make_vit_policy,
        )
        
        slim = search.best_model
        print("Best export params:", search.best_params)
        for rec in sorted(search.trials, key=lambda r: r["mean_ms"])[:5]:
            print("  trial:", rec)
    
        
        # # Export slim model (use a step >= warmup to enable pruning)
        # step_for_export = max(9999, int(pack["recipe"].get("export", {}).get("warmup_steps", 0)) + 1)
        # slim = ViTAdapter.export_pruned(student, export_policy, step_for_export)
    
        # Measure latency before/after on a small val batch
        img_size = pack["recipe"]["data"]["img_size"]
        B = max(1, int(pack["recipe"]["data"]["batch_size"] // 4))
        mean_keep, p95_keep = measure_latency_ms(ViTAdapter.export_keepall(student), (B, 3, img_size, img_size), device=pack["device"])  # type: ignore[index]
        mean_slim, p95_slim = measure_latency_ms(slim, (B, 3, img_size, img_size), device=pack["device"])  # type: ignore[index]
    
        print(f"Keep-all: mean={mean_keep:.3f}ms p95={p95_keep:.3f}ms | Slim: mean={mean_slim:.3f}ms p95={p95_slim:.3f}ms | \nSpeedup={(mean_keep-mean_slim)/max(1e-6,mean_keep)*100:.2f}%")
    
        # Save artifacts
        torch.save(slim, os.path.join(args.outdir, "vit_slim.pth"))
        torch.save(student, os.path.join(args.outdir, "vit_student_with_gates.pth"))
    
        print(f"Saved pruned model to {os.path.join(args.outdir, 'vit_slim.pth')}")
    
    else:
        slim = torch.load(args.slim, map_location=device, weights_only=False).to(device)
        print(f"Skipping training. Pruned model loaded from {args.slim}. [To run with training, drop '--slim' argument]")

    if args.finetune is True:
        # --- Fine-tune the selected slim model against the teacher ---
        ft_epochs = int(pack["recipe"].get("finetune", {}).get("epochs", 5))
        print(f"\nStarting fine tuning for {ft_epochs} epochs...")
        
        ft_cfg = FinetuneConfig(
            epochs=ft_epochs,
            lr=float(pack["recipe"].get("finetune", {}).get("lr", 3e-4)),
            kd=KDConfig(**pack["recipe"].get("trainer", {}).get("kd", {})),
            amp=bool(pack["recipe"].get("trainer", {}).get("amp", True)),
            device=pack["device"],
            log_every=50,
        )
        slim = finetune_student(
            slim,
            teacher,
            pack["train_loader"],
            get_student_logits=lambda m, x: ViTAdapter.get_logits(m, x, head=pack["student_head"]),
            get_teacher_logits=lambda m, x: ViTAdapter.get_logits(m, x, head=pack["teacher_head"]),
            cfg=ft_cfg,
            val_loader=pack["val_loader"],
        )
        torch.save(slim, os.path.join(args.outdir, "vit_slim_finetune.pth"))
        
    else:
        print("Skipping fine-tuning. [To run fine-tuning, set '--finetuning True' or omit this argument]")

    print(f"Starting benchmarking with batch size = {B}...")
        
    mean_keep, p95_keep = measure_latency_ms(ViTAdapter.export_keepall(student), (B, 3, img_size, img_size), device=pack["device"])
    mean_slim, p95_slim = measure_latency_ms(slim, (B, 3, img_size, img_size), device=pack["device"])

    print(f"Keep-all: mean={mean_keep:.3f}ms p95={p95_keep:.3f}ms | Slim: mean={mean_slim:.3f}ms p95={p95_slim:.3f}ms | \n"
          f"Speedup={(mean_keep-mean_slim)/max(1e-6,mean_keep)*100:.2f}%")    
    

if __name__ == "__main__":
    main()
