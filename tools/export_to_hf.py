#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Export/push models to Hugging Face Hub:
  (a) student with trained gates (custom code allowed)
  (b) pruned slim model after fine-tuning

Supports:
  - LLaMA-like (HF AutoModelForCausalLM)
  - ResNet (torchvision/timm-style) as custom-code repos

Usage examples:

# LLaMA — gated student & slim
python -m tools.export_to_hf \
  --task llama \
  --base_id meta-llama/Llama-3.2-1B \
  --student_ckpt runs/llama3p2_1b/llama_student_gated.pt \
  --slim_ckpt runs/llama3p2_1b/slim.pt \
  --repo_gated yourname/llama3p2-1b-gated \
  --repo_slim yourname/llama3p2-1b-slim \
  --token $HF_TOKEN \
  --include_code adapters/huggingface,core,gates

# ResNet — gated student & slim
python -m tools.export_to_hf \
  --task resnet \
  --base_id torchvision/resnet18 \
  --student_ckpt runs/resnet18/resnet18_student_gated.pth \
  --slim_ckpt runs/resnet18/resnet18_slim.pth \
  --repo_gated yourname/resnet18-gated \
  --repo_slim yourname/resnet18-slim \
  --token $HF_TOKEN \
  --include_code adapters/torchvision,core,gates
"""

from __future__ import annotations
import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from typing import List, Optional

import torch

# Optional but recommended
from huggingface_hub import HfApi, create_repo, upload_folder, metadata_update
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

# --- tiny utils --------------------------------------------------------------

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _write_readme(dst: Path, title: str, task: str, extra: dict):
    md = dst / "README.md"
    lines = ["```yaml",
        "---",
        "library_name: pytorch",
        "tags:",
        "  - resnet",
        "  - pruning",
        "  - knowledge-distillation",
        "  - speedup",
        "license: apache-2.0",
        "dataset: imagenet-1k",
        "pipeline_tag: image-classification",
        "---",
        "```"]
    
    lines += [f"# {title}\n"]
    lines += [
        "This repository contains two variants:",
        "- **Gated student** (with learned pruning gates) – requires custom code.",
        "- **Slim student** (post-prune/export) – loads with standard code (LLM) or bundled code (ResNet).",
        "",
        "## Inference (LLM, slim)",
        "```python",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        f"tok = AutoTokenizer.from_pretrained('{extra.get('repo_slim','')}')",
        f"mdl = AutoModelForCausalLM.from_pretrained('{extra.get('repo_slim','')}', torch_dtype='auto').eval()",
        "x = tok('Hello', return_tensors='pt')",
        "print(tok.decode(mdl.generate(**x, max_new_tokens=16)[0]))",
        "```",
        "",
        "## Notes",
        "- The **gated** repo includes lightweight custom code (adapters/…, core/…) needed to attach/load gates.",
        "- The **slim** LLM is exported to standard HF architecture for out-of-the-box loading.",
        "- For ResNet, both repos include minimal custom code to define the module.",
        "",
        "## Training metadata",
        "```json",
        json.dumps(extra, indent=2),
        "```",
        ""
    ]
    md.write_text("\n".join(lines), encoding="utf-8")

def _copy_code_tree(dst: Path, roots: List[str]):
    for r in roots:
        src = Path(r)
        if not src.exists():
            print(f"[warn] include_code path not found: {src}")
            continue
        tgt = dst / src.name
        if tgt.exists():
            shutil.rmtree(tgt)
        shutil.copytree(src, tgt)

def _save_model_card_and_meta(dst: Path, title: str, task: str, meta: dict, include_code: List[str], repo_slim: str):
    _write_readme(dst, title, task, {**meta, "repo_slim": repo_slim})
    # Add a lightweight model index (optional)
    (dst / "model_index.json").write_text(json.dumps({"task": task, **meta}, indent=2), encoding="utf-8")

# --- LLaMA export ------------------------------------------------------------

def export_llama_variants(
    base_id: str,
    student_ckpt: Path,
    slim_ckpt: Path,
    repo_gated: str,
    repo_slim: str,
    token: Optional[str],
    private: bool,
    include_code: List[str],
    push: bool = True,
):
    api = HfApi(token=token)

    # Load base config & tokenizer
    cfg = AutoConfig.from_pretrained(base_id, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(base_id, use_fast=True)

    # ---------- (a) Gated student (custom-code) ----------
    gated_dir = Path("hf_export_gated_llama")
    if gated_dir.exists():
        shutil.rmtree(gated_dir)
    _ensure_dir(gated_dir)

    # Save tokenizer/config
    tok.save_pretrained(gated_dir)
    cfg.save_pretrained(gated_dir)

    # Save checkpoint (just state dict); custom code will define class at load time
    # Expect your loader to call LlamaAdapter(student).attach_gates(...) then load_state_dict.
    sd = torch.load(student_ckpt, map_location="cpu", weights_only=False)
    torch.save(sd, gated_dir / "pytorch_model.bin")

    # Bundle custom code
    if include_code:
        _copy_code_tree(gated_dir, include_code)
        # HF flag so code is trusted
        (gated_dir / "custom_code.py").write_text(
            "# Marker file so Hub shows 'custom code' banner.\n", encoding="utf-8"
        )

    _save_model_card_and_meta(
        gated_dir, title=f"{repo_gated}", task="causal-lm",
        meta={"base_id": base_id, "variant": "gated-student"},
        include_code=include_code, repo_slim=repo_slim
    )

    if push:
        create_repo(repo_gated, token=token, private=private, exist_ok=True)
        upload_folder(repo_id=repo_gated, folder_path=str(gated_dir), token=token)
        print(f"[ok] Pushed gated student → {repo_gated}")

    # ---------- (b) Slim (standard HF) ----------
    slim_dir = Path("hf_export_slim_llama")
    if slim_dir.exists():
        shutil.rmtree(slim_dir)
    _ensure_dir(slim_dir)

    tok.save_pretrained(slim_dir)
    cfg.save_pretrained(slim_dir)

    # Build a base model then load slim state dict (exported to standard arch)
    mdl = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype="auto", trust_remote_code=True)
    sd_slim = torch.load(slim_ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = mdl.load_state_dict(sd_slim, strict=False)
    print(f"[slim] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")

    mdl.save_pretrained(slim_dir, safe_serialization=True)

    _save_model_card_and_meta(
        slim_dir, title=f"{repo_slim}", task="causal-lm",
        meta={"base_id": base_id, "variant": "slim-export"},
        include_code=[], repo_slim=repo_slim
    )

    if push:
        create_repo(repo_slim, token=token, private=private, exist_ok=True)
        upload_folder(repo_id=repo_slim, folder_path=str(slim_dir), token=token)
        print(f"[ok] Pushed slim student → {repo_slim}")

# --- ResNet export (custom code both variants) -------------------------------

RESNET_MIN_CODE = """\
# minimal_resnet_loader.py
# Minimal loader expected by this repo: defines StudentResNet + SlimResNet and
# helper functions to load checkpoints.

import torch
import torch.nn as nn
import torchvision.models as tv

def build_base_resnet18(num_classes=1000):
    m = tv.resnet18(weights=None, num_classes=num_classes)
    return m

class StudentResNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # Start from resnet18 skeleton; gated or slim weights will be loaded
        self.model = build_base_resnet18(num_classes=num_classes)
    def forward(self, x):
        return self.model(x)

# Convenience
def load_student(checkpoint_path, device='cpu', num_classes=1000):
    m = StudentResNet(num_classes=num_classes)
    sd = torch.load(checkpoint_path, map_location=device)
    m.load_state_dict(sd, strict=False)
    return m
"""

def export_resnet_variants(
    base_id: str,
    student_ckpt: Path,
    slim_ckpt: Path,
    repo_gated: str,
    repo_slim: str,
    token: Optional[str],
    private: bool,
    include_code: List[str],
    push: bool = True,
):
    api = HfApi(token=token)

    # ---------- (a) Gated student (custom code) ----------
    gated_dir = Path("hf_export_gated_resnet")
    if gated_dir.exists():
        shutil.rmtree(gated_dir)
    _ensure_dir(gated_dir)

    # Save raw SD
    torch.save(torch.load(student_ckpt, map_location="cpu", weights_only=False), gated_dir / "pytorch_model.bin")

    # Bundle minimal loader + user code (if any)
    (gated_dir / "minimal_resnet_loader.py").write_text(RESNET_MIN_CODE, encoding="utf-8")
    if include_code:
        _copy_code_tree(gated_dir, include_code)
    _save_model_card_and_meta(
        gated_dir, title=f"{repo_gated}", task="image-classification",
        meta={"base_id": base_id, "variant": "gated-student"},
        include_code=include_code, repo_slim=repo_slim
    )
    if push:
        create_repo(repo_gated, token=token, private=private, exist_ok=True)
        upload_folder(repo_id=repo_gated, folder_path=str(gated_dir), token=token)
        print(f"[ok] Pushed gated student → {repo_gated}")

    # ---------- (b) Slim student (custom code, but same loader works) ----------
    slim_dir = Path("hf_export_slim_resnet")
    if slim_dir.exists():
        shutil.rmtree(slim_dir)
    _ensure_dir(slim_dir)

    torch.save(torch.load(slim_ckpt, map_location="cpu", weights_only=False), slim_dir / "pytorch_model.bin")
    (slim_dir / "minimal_resnet_loader.py").write_text(RESNET_MIN_CODE, encoding="utf-8")
    if include_code:
        _copy_code_tree(slim_dir, include_code)
    _save_model_card_and_meta(
        slim_dir, title=f"{repo_slim}", task="image-classification",
        meta={"base_id": base_id, "variant": "slim-export"},
        include_code=include_code, repo_slim=repo_slim
    )
    if push:
        create_repo(repo_slim, token=token, private=private, exist_ok=True)
        upload_folder(repo_id=repo_slim, folder_path=str(slim_dir), token=token)
        print(f"[ok] Pushed slim student → {repo_slim}")

# --- CLI ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser("Export and push student/slim models to Hugging Face")
    ap.add_argument("--task", choices=["llama", "resnet"], required=True,
                    help="Model family: 'llama' (HF causal LM) or 'resnet' (vision)")
    ap.add_argument("--base_id", type=str, required=True,
                    help="Base HF id used for config/tokenizer (llama) or as metadata (resnet)")
    ap.add_argument("--student_ckpt", type=str, required=True, help="Path to student gated checkpoint")
    ap.add_argument("--slim_ckpt", type=str, required=True, help="Path to slim exported checkpoint")
    ap.add_argument("--repo_gated", type=str, required=True, help="Destination repo for gated student")
    ap.add_argument("--repo_slim", type=str, required=True, help="Destination repo for slim model")
    ap.add_argument("--token", type=str, default=os.getenv("HF_TOKEN"), help="HF token (or set HF_TOKEN env)")
    ap.add_argument("--private", action="store_true", help="Create private repos")
    ap.add_argument("--no_push", action="store_true", help="Prepare folders only, do not push")
    ap.add_argument("--include_code", type=str, default="",
                    help="Comma-separated local code roots to bundle (e.g., 'adapters/huggingface,core')")
    args = ap.parse_args()

    include_code = [s.strip() for s in args.include_code.split(",") if s.strip()]
    push = not args.no_push

    if args.task == "llama":
        export_llama_variants(
            base_id=args.base_id,
            student_ckpt=Path(args.student_ckpt),
            slim_ckpt=Path(args.slim_ckpt),
            repo_gated=args.repo_gated,
            repo_slim=args.repo_slim,
            token=args.token,
            private=args.private,
            include_code=include_code,
            push=push,
        )
    else:
        export_resnet_variants(
            base_id=args.base_id,
            student_ckpt=Path(args.student_ckpt),
            slim_ckpt=Path(args.slim_ckpt),
            repo_gated=args.repo_gated,
            repo_slim=args.repo_slim,
            token=args.token,
            private=args.private,
            include_code=include_code,
            push=push,
        )

if __name__ == "__main__":
    main()
