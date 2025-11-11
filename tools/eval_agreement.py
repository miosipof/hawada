# tools/eval_agreement.py
from __future__ import annotations
import argparse, yaml
from types import SimpleNamespace as NS

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# data + adapters
from data.vision import VisionDataConfig, build_imagenet_like_loaders, _images_from_batch
from adapters.huggingface.vit import ViTAdapter
from adapters.torchvision.resnet import ResNetAdapter

# heads / KD
from core.distill import KDConfig, kd_loss


def _ns_from_recipe(recipe_path: str):
    with open(recipe_path, "r") as f:
        y = yaml.safe_load(f)

    d = y["data"]
    dcfg = VisionDataConfig(
        train_root=d.get("train_root"),
        val_root=d.get("val_root"),
        img_size=d.get("img_size", 224),
        batch_size=d.get("batch_size", 32),
        val_batch_size=d.get("val_batch_size"),
        # num_workers=d.get("num_workers", 8),
        limit_train=d.get("limit_train"),
        limit_val=d.get("limit_val"),
        # drop_last=True,
        # pin_memory=True,
    )
    return NS(
        data=dcfg,
        teacher=y.get("teacher", {}),
        student=y.get("student", {}),
        model=y.get("model", "vit"),
    )


@torch.inference_mode()
def _eval_loop(get_s, get_t, loader, batches: int, T: float, device: str):
    kd_cfg = KDConfig(temperature=T, alpha=1.0)
    kl_sum, n_batches = 0.0, 0
    agree_top1, n_samples = 0, 0

    for i, batch in enumerate(loader, 1):
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch
        x = x.to(device, non_blocking=True)

        s = get_s(x)
        t = get_t(x)

        # KL
        kl = kd_loss(s, t, kd_cfg).item()
        kl_sum += kl
        n_batches += 1

        # argmax agreement
        agree_top1 += (s.argmax(-1) == t.argmax(-1)).sum().item()
        n_samples += x.size(0)

        if i >= batches:
            break

    return {
        "kl_mean": kl_sum / max(1, n_batches),
        "top1_agreement": agree_top1 / max(1, n_samples),
        "batches": n_batches,
        "samples": n_samples,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["vit", "resnet"], required=True)
    ap.add_argument("--recipe", type=str, required=True)
    ap.add_argument("--slim", type=str, required=True)
    ap.add_argument("--batches", type=int, default=200)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--T", type=float, default=1.0)
    args = ap.parse_args()

    cfg = _ns_from_recipe(args.recipe)
    dev = args.device

    # loaders
    dcfg = cfg.data
    # override BS for faster eval if provided
    dcfg = VisionDataConfig(**{**dcfg.__dict__, "val_batch_size": args.bs})
    _, val_loader = build_imagenet_like_loaders(dcfg)

    if args.model == "vit":
        from transformers import ViTModel, ViTForImageClassification


        class _LocalHead(nn.Module):
            """Replicates ViTForImageClassification head: LayerNorm + Linear."""
            def __init__(self, hidden_size: int, num_classes: int, base_fc: nn.Linear):
                super().__init__()
                self.norm = nn.LayerNorm(hidden_size)
                self.fc = nn.Linear(hidden_size, num_classes, bias=(base_fc.bias is not None))
                with torch.no_grad():
                    self.fc.weight.copy_(base_fc.weight)
                    if base_fc.bias is not None:
                        self.fc.bias.copy_(base_fc.bias)
        
            def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
                return self.fc(self.norm(cls_token))
        
    
        teacher_name = cfg.model
        student_name = cfg.model
    
        teacher = ViTForImageClassification.from_pretrained(teacher_name).to(dev).eval()
    
        # base_head = ViTForImageClassification.from_pretrained(student_name)
        # hidden = base_head.config.hidden_size
        # num_labels = base_head.config.num_labels

        student = torch.load(args.slim, map_location=dev, weights_only=False).to(dev).eval()
        hidden = int(student.config.hidden_size)
        num_classes = int(student.config.num_labels)
        teacher_head = None #_LocalHead(hidden, num_labels, base_head.classifier).to(dev).eval()
        student_head = None # _LocalHead(hidden, num_labels, base_head.classifier).to(dev).eval()
    
        # --- Adapter-specific logits providers
        get_t = lambda batch: ViTAdapter.get_logits(
            teacher, _images_from_batch(batch).to(next(teacher.parameters()).device, non_blocking=True), head=teacher_head
        ).detach()
        get_s = lambda batch: ViTAdapter.get_logits(
            student, _images_from_batch(batch).to(next(student.parameters()).device, non_blocking=True), head=student_head
        )


    else:  # resnet
        import torchvision
        # teacher: torchvision resnet50 (same as the example)
        teacher = torchvision.models.resnet50(weights="IMAGENET1K_V2").to(dev).eval()
        student = torch.load(args.slim, map_location=dev, weights_only=False).to(dev).eval()

        get_s = lambda x: ResNetAdapter.get_logits(student, x)
        get_t = lambda x: ResNetAdapter.get_logits(teacher, x)

    stats = _eval_loop(get_s, get_t, val_loader, args.batches, args.T, dev)
    print(f"KL@T={args.T}: {stats['kl_mean']:.6f}")
    print(f"Top-1 agreement: {100.0*stats['top1_agreement']:.2f}% "
          f"on {stats['samples']} samples / {stats['batches']} batches")


if __name__ == "__main__":
    main()
