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

# ---- NEW: Llama adapter (you already have this) ----
from adapters.huggingface.llama import LlamaAdapter

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
    """
    Generic eval loop for single-logit-per-sample models (ViT / ResNet).

    get_s/get_t: callable(batch) -> logits of shape (B, C)
    """
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


@torch.inference_mode()
def _eval_loop_llama(get_s, get_t, loader, batches: int, T: float, device: str):
    """
    Eval loop for autoregressive LMs (Llama).

    Assumes each batch is a dict with:
        - "input_ids": (B, T)
        - "attention_mask": (B, T) with 1 for real tokens, 0 for padding

    get_s/get_t: callable(batch_dict) -> logits of shape (B, T, V)
    We compute:
      - token-level KL over masked tokens
      - token-level top-1 agreement over masked tokens
    """
    kd_cfg = KDConfig(temperature=T, alpha=1.0)
    kl_sum, n_batches = 0.0, 0
    agree_top1, n_tokens = 0, 0

    for i, batch in enumerate(loader, 1):
        # Move everything to device
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        attn_mask = batch.get("attention_mask", None)

        s = get_s(batch)  # (B, T, V)
        t = get_t(batch)  # (B, T, V)

        # KL: assume kd_loss supports a `mask` argument (same as in your LM training)
        kl = kd_loss(s, t, kd_cfg, mask=attn_mask).item()
        kl_sum += kl
        n_batches += 1

        # Token-level argmax agreement
        s_arg = s.argmax(-1)  # (B, T)
        t_arg = t.argmax(-1)  # (B, T)

        if attn_mask is not None:
            mask = attn_mask.bool()
            agree_top1 += (s_arg[mask] == t_arg[mask]).sum().item()
            n_tokens += mask.sum().item()
        else:
            agree_top1 += (s_arg == t_arg).sum().item()
            n_tokens += s_arg.numel()

        if i >= batches:
            break

    return {
        "kl_mean": kl_sum / max(1, n_batches),
        "top1_agreement": agree_top1 / max(1, n_tokens),
        "batches": n_batches,
        "samples": n_tokens,  # here: tokens, not sequences
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["vit", "resnet", "llama"], required=True)
    ap.add_argument("--recipe", type=str, required=True)
    ap.add_argument("--slim", type=str, required=True)
    ap.add_argument("--batches", type=int, default=200)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--T", type=float, default=1.0)
    args = ap.parse_args()

    cfg = _ns_from_recipe(args.recipe)
    dev = args.device

    # -----------------------
    # Vision models (ViT/ResNet)
    # -----------------------
    if args.model in ("vit", "resnet"):
        # loaders
        dcfg = cfg.data
        # override BS for faster eval if provided
        dcfg = VisionDataConfig(**{**dcfg.__dict__, "val_batch_size": args.bs})
        _, val_loader = build_imagenet_like_loaders(dcfg)

    # -----------------------
    # Llama: text data loader
    # -----------------------
    if args.model == "llama":
        # Adjust these imports to your actual LM data pipeline.
        # Example assumes you have something like:
        #   from data.text import LMDataConfig, build_lm_loader
        #
        # The recipe should contain the text-data config under `data` similarly
        # to how vision recipes work.
        from data.text import LMDataConfig, build_lm_loader  # TODO: fix imports

        d = cfg.data  # reusing same NS; if needed, create LMDataConfig directly from yaml
        lm_cfg = LMDataConfig(
            dataset=d.__dict__.get("dataset", None),
            data_root=d.__dict__.get("data_root", None),
            seq_len=d.__dict__.get("seq_len", 2048),
            batch_size=args.bs,
            limit=d.__dict__.get("limit_val", None),
        )
        val_loader = build_lm_loader(lm_cfg, split="val")

    # ==========================================================
    # MODEL-SPECIFIC SETUP
    # ==========================================================
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

        # For now, we just use cfg.model as HF id
        teacher_name = cfg.model
        student_name = cfg.model

        teacher = ViTForImageClassification.from_pretrained(teacher_name).to(dev).eval()
        student = torch.load(args.slim, map_location=dev, weights_only=False).to(dev).eval()

        hidden = int(student.config.hidden_size)
        num_classes = int(student.config.num_labels)
        teacher_head = None
        student_head = None

        get_t = lambda batch: ViTAdapter.get_logits(
            teacher,
            _images_from_batch(batch).to(next(teacher.parameters()).device, non_blocking=True),
            head=teacher_head,
        ).detach()
        get_s = lambda batch: ViTAdapter.get_logits(
            student,
            _images_from_batch(batch).to(next(student.parameters()).device, non_blocking=True),
            head=student_head,
        )

        stats = _eval_loop(get_s, get_t, val_loader, args.batches, args.T, dev)

    elif args.model == "resnet":
        import torchvision

        # teacher: torchvision resnet50 (same as the example)
        teacher = torchvision.models.resnet50(weights="IMAGENET1K_V2").to(dev).eval()
        student = torch.load(args.slim, map_location=dev, weights_only=False).to(dev).eval()

        get_s = lambda x: ResNetAdapter.get_logits(student, x)
        get_t = lambda x: ResNetAdapter.get_logits(teacher, x)

        stats = _eval_loop(get_s, get_t, val_loader, args.batches, args.T, dev)

    else:  # ---- LLAMA ----
        from transformers import AutoModelForCausalLM

        # Try to get teacher HF id from the recipe; fall back to cfg.model
        teacher_name = (
            cfg.teacher.get("id")
            or cfg.teacher.get("name")
            or cfg.model
        )

        teacher = AutoModelForCausalLM.from_pretrained(
            teacher_name,
            torch_dtype=torch.float16 if "cuda" in dev else torch.float32,
        ).to(dev).eval()

        # Student: your slim checkpoint (SlimLlamaForCausalLM or similar)
        student = torch.load(args.slim, map_location=dev, weights_only=False).to(dev).eval()

        def _get_t(batch):
            # Expect batch: {"input_ids", "attention_mask", ...}
            return LlamaAdapter.get_logits(
                teacher,
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask", None),
            ).detach()

        def _get_s(batch):
            return LlamaAdapter.get_logits(
                student,
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask", None),
            )

        stats = _eval_loop_llama(_get_s, _get_t, val_loader, args.batches, args.T, dev)

    # ----------------- Print results -----------------
    print(f"KL@T={args.T}: {stats['kl_mean']:.6f}")
    if args.model == "llama":
        print(
            f"Top-1 token agreement: {100.0*stats['top1_agreement']:.2f}% "
            f"on {stats['samples']} tokens / {stats['batches']} batches"
        )
    else:
        print(
            f"Top-1 agreement: {100.0*stats['top1_agreement']:.2f}% "
            f"on {stats['samples']} samples / {stats['batches']} batches"
        )


if __name__ == "__main__":
    main()
