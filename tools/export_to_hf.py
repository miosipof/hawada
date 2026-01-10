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
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers import ViTModel, ViTForImageClassification

from pathlib import Path
from typing import List, Optional, Tuple

import shutil
import torch.nn as nn
from transformers import AutoConfig, ViTForImageClassification
from huggingface_hub import HfApi, create_repo, upload_folder
from copy import deepcopy

from adapters.huggingface.llama import SlimLlamaForCausalLM
from adapters.huggingface.llama import load_slim_llama

# --- tiny utils --------------------------------------------------------------

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _write_readme(dst: Path, title: str, task: str, extra: dict):
    md = dst / "README.md"

    if task == "causal-lm":
        tags = [
            "llama",
            "causal-lm",
            "text-generation",
            "pruning",
            "knowledge-distillation",
            "speedup",
        ]
        pipeline_tag = "text-generation"
        dataset = extra.get("dataset", "slimpajama-test")
    else:
        tags = [
            "resnet",
            "pruning",
            "knowledge-distillation",
            "speedup",
        ]
        pipeline_tag = "image-classification"
        dataset = extra.get("dataset", "imagenet-1k")

    lines = [
        "```yaml",
        "---",
        "library_name: pytorch",
        "tags:",
    ]
    for t in tags:
        lines.append(f"  - {t}")
    lines += [
        "license: apache-2.0",
        f"dataset: {dataset}",
        f"pipeline_tag: {pipeline_tag}",
        "---",
        "```",
        "",
        f"# {title}",
        "",
        "This repository contains two variants:",
        "- **Gated student** (with learned pruning gates) – requires custom code.",
        "- **Slim student** (post-prune/export) – loads with standard HF APIs plus this repo’s custom code.",
        "",
    ]

    if task == "causal-lm":
        lines += [
            "## Inference (LLaMA slim)",
            "```python",
            "from transformers import AutoModelForCausalLM, AutoTokenizer",
            f"tok = AutoTokenizer.from_pretrained('{extra.get('repo_slim','')}')",
            f"mdl = AutoModelForCausalLM.from_pretrained('{extra.get('repo_slim','')}', torch_dtype='auto').eval()",
            "x = tok('Hello', return_tensors='pt')",
            "print(tok.decode(mdl.generate(**x, max_new_tokens=16)[0]))",
            "```",
            "",
        ]
    else:
        lines += [
            "## Inference (Vision slim)",
            "```python",
            "import torch",
            "from torchvision import transforms",
            "from PIL import Image",
            "",
            f"repo = '{extra.get('repo_slim','')}'",
            "img = Image.open('some_image.jpg')",
            "# ... your preprocessing here ...",
            "```",
            "",
        ]

    lines += [
        "## Notes",
        "- The **gated** repo includes lightweight custom code (adapters/…, core/…) needed to attach/load gates.",
        "- The **slim** model is exported for efficient inference.",
        "",
        "## Training metadata",
        "```json",
        json.dumps(extra, indent=2),
        "```",
        "",
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



# --- ViT export ------------------------------------------------------------

from copy import deepcopy
from transformers import AutoConfig

SLIM_VIT_SOURCE = r"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import ViTConfig
from transformers.activations import ACT2FN
from transformers.modeling_outputs import ImageClassifierOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.vit.modeling_vit import ViTEmbeddings


@dataclass
class SlimLayout:
    num_heads: List[int]
    intermediate_sizes: List[int]
    head_dim: int


class SlimViTSelfAttention(nn.Module):
    def __init__(self, config: ViTConfig, layer_idx: int, layout: SlimLayout):
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.num_attention_heads = int(layout.num_heads[layer_idx])
        self.attention_head_size = int(layout.head_dim)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if self.all_head_size <= 0:
            raise ValueError(f"Invalid all_head_size at layer {layer_idx}: {self.all_head_size}")

        self.query = nn.Linear(self.hidden_size, self.all_head_size, bias=True)
        self.key   = nn.Linear(self.hidden_size, self.all_head_size, bias=True)
        self.value = nn.Linear(self.hidden_size, self.all_head_size, bias=True)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def _shape(self, x: torch.Tensor, bsz: int, seq_len: int) -> torch.Tensor:
        return x.view(bsz, seq_len, self.num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.size()
        q = self._shape(self.query(hidden_states), bsz, seq_len)
        k = self._shape(self.key(hidden_states),   bsz, seq_len)
        v = self._shape(self.value(hidden_states), bsz, seq_len)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / (self.attention_head_size ** 0.5)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, v)  # [B, H, S, Dh]
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(bsz, seq_len, self.all_head_size)  # [B, S, H*Dh]
        return attn_output


class SlimViTAttention(nn.Module):
    def __init__(self, config: ViTConfig, layer_idx: int, layout: SlimLayout):
        super().__init__()
        self.attention = SlimViTSelfAttention(config, layer_idx, layout)
        self.output = nn.Module()
        self.output.dense = nn.Linear(layout.head_dim * layout.num_heads[layer_idx], config.hidden_size, bias=True)
        self.output.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.attention(hidden_states)
        x = self.output.dense(x)
        x = self.output.dropout(x)
        return x


class SlimViTIntermediate(nn.Module):
    def __init__(self, config: ViTConfig, layer_idx: int, layout: SlimLayout):
        super().__init__()
        dim = int(layout.intermediate_sizes[layer_idx])
        self.dense = nn.Linear(config.hidden_size, dim)
        self.intermediate_act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.intermediate_act_fn(self.dense(hidden_states))


class SlimViTOutput(nn.Module):
    def __init__(self, config: ViTConfig, layer_idx: int, layout: SlimLayout):
        super().__init__()
        dim = int(layout.intermediate_sizes[layer_idx])
        self.dense = nn.Linear(dim, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.dense(hidden_states)
        x = self.dropout(x)
        return x


class SlimViTLayer(nn.Module):
    def __init__(self, config: ViTConfig, layer_idx: int, layout: SlimLayout):
        super().__init__()
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = SlimViTAttention(config, layer_idx, layout)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.intermediate = SlimViTIntermediate(config, layer_idx, layout)
        self.output = SlimViTOutput(config, layer_idx, layout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # MHA block
        residual = hidden_states
        hidden_states = self.layernorm_before(hidden_states)
        hidden_states = self.attention(hidden_states)
        hidden_states = residual + hidden_states

        # FFN block
        residual = hidden_states
        hidden_states = self.layernorm_after(hidden_states)
        hidden_states = self.output(self.intermediate(hidden_states))
        hidden_states = residual + hidden_states
        return hidden_states


class SlimViTEncoder(nn.Module):
    def __init__(self, config: ViTConfig, layout: SlimLayout):
        super().__init__()
        self.layer = nn.ModuleList(
            SlimViTLayer(config, i, layout) for i in range(config.num_hidden_layers)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for blk in self.layer:
            hidden_states = blk(hidden_states)
        return hidden_states


class SlimViTModel(nn.Module):
    def __init__(self, config: ViTConfig, layout: SlimLayout):
        super().__init__()
        self.config = config
        self.embeddings = ViTEmbeddings(config)
        self.encoder = SlimViTEncoder(config, layout)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.encoder(hidden_states)
        hidden_states = self.layernorm(hidden_states)
        return hidden_states


class SlimViTForImageClassification(PreTrainedModel):
    config_class = ViTConfig
    base_model_prefix = "vit"

    def __init__(self, config: ViTConfig):
        super().__init__(config)

        slim_num_heads = getattr(config, "slim_num_heads", None)
        slim_inter_sizes = getattr(config, "slim_intermediate_sizes", None)
        if slim_num_heads is None or slim_inter_sizes is None:
            raise ValueError("Config must define slim_num_heads and slim_intermediate_sizes.")

        if hasattr(config, "slim_head_dim"):
            head_dim = int(config.slim_head_dim)
        else:
            if config.hidden_size % config.num_attention_heads != 0:
                raise ValueError(
                    "Cannot infer head_dim: hidden_size must be divisible by num_attention_heads "
                    "if slim_head_dim is not provided."
                )
            head_dim = config.hidden_size // config.num_attention_heads

        layout = SlimLayout(
            num_heads=[int(h) for h in slim_num_heads],
            intermediate_sizes=[int(d) for d in slim_inter_sizes],
            head_dim=int(head_dim),
        )

        self.num_labels = config.num_labels
        self.vit = SlimViTModel(config, layout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.post_init()

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> ImageClassifierOutput:

        hidden_states = self.vit(pixel_values)
        cls = hidden_states[:, 0]  # CLS token
        logits = self.classifier(cls)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss = F.mse_loss(logits.view(-1), labels.view(-1))
            else:
                loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            out = (logits, hidden_states)
            return ((loss,) + out) if loss is not None else out

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
"""


def export_vit_variants(
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

    # -------------------------------------------------------------------------
    # (a) Gated student: full ViT + attached gates (state_dict only)
    # -------------------------------------------------------------------------
    cfg = AutoConfig.from_pretrained(base_id, trust_remote_code=True)

    gated_dir = Path("hf_export_gated_vit")
    if gated_dir.exists():
        shutil.rmtree(gated_dir)
    _ensure_dir(gated_dir)

    # Save config as-is (no structural change; user will reattach gates)
    cfg.save_pretrained(gated_dir)

    # Save checkpoint (state dict)
    sd_gated = torch.load(student_ckpt, map_location="cpu", weights_only=False)
    torch.save(sd_gated, gated_dir / "pytorch_model.bin")
    
    # Bundle custom code for the gating adapter
    if include_code:
        _copy_code_tree(gated_dir, include_code)
        (gated_dir / "custom_code.py").write_text(
            "# Marker file so Hub shows 'custom code' banner.\n", encoding="utf-8"
        )

    _save_model_card_and_meta(
        gated_dir,
        title=f"{repo_gated}",
        task="image-classification",
        meta={
            "base_id": base_id,
            "variant": "gated-student",
            "description": "Full ViT backbone with learned soft gates for heads/FFN.",
        },
        include_code=include_code,
        repo_slim=repo_slim,
    )

    if push:
        create_repo(repo_gated, token=token, private=private, exist_ok=True)
        upload_folder(repo_id=repo_gated, folder_path=str(gated_dir), token=token)
        print(f"[ok] Pushed gated student → {repo_gated}")

    # -------------------------------------------------------------------------
    # (b) Slim student: custom SlimViTForImageClassification
    # -------------------------------------------------------------------------
    slim_dir = Path("hf_export_slim_vit")
    if slim_dir.exists():
        shutil.rmtree(slim_dir)
    _ensure_dir(slim_dir)

    # Load base config and slim state_dict
    base_cfg = AutoConfig.from_pretrained(base_id, trust_remote_code=True)
    sd_slim = torch.load(slim_ckpt, map_location="cpu", weights_only=False)

    num_layers = int(base_cfg.num_hidden_layers)
    orig_heads = int(base_cfg.num_attention_heads)
    hidden_size = int(base_cfg.hidden_size)
    
    # Original ViT head dim (e.g. 64 for ViT-B/16)
    if orig_heads <= 0 or hidden_size % orig_heads != 0:
        raise ValueError(
            f"Base config has hidden_size={hidden_size}, num_attention_heads={orig_heads}, "
            "cannot infer head_dim."
        )
    head_dim = hidden_size // orig_heads

    slim_num_heads: List[int] = []
    slim_inter_sizes: List[int] = []

    for i in range(num_layers):
        q_key = f"vit.encoder.layer.{i}.attention.attention.query.weight"
        inter_key = f"vit.encoder.layer.{i}.intermediate.dense.weight"

        if q_key not in sd_slim:
            raise KeyError(f"Missing key in slim state_dict: {q_key}")
        if inter_key not in sd_slim:
            raise KeyError(f"Missing key in slim state_dict: {inter_key}")

        q_w = sd_slim[q_key]          # [H_i * head_dim, hidden_size]
        inter_w = sd_slim[inter_key]  # [d_ff_i, hidden_size]

        out_dim = q_w.shape[0]
        if out_dim % head_dim != 0:
            raise ValueError(
                f"Layer {i}: query out_features={out_dim} not divisible by head_dim={head_dim}"
            )
        h_i = out_dim // head_dim
        slim_num_heads.append(int(h_i))
        slim_inter_sizes.append(int(inter_w.shape[0]))

    # Build a config specialized for SlimViT
    cfg_slim = base_cfg
    cfg_slim.slim_num_heads = slim_num_heads
    cfg_slim.slim_intermediate_sizes = slim_inter_sizes
    cfg_slim.slim_head_dim = head_dim

    # Infer num_classes from checkpoint
    candidate_keys = [
        "classifier.bias",
        "classifier.weight",
        "vit.classifier.bias",
        "vit.classifier.weight",
        "head.bias",
        "head.weight",
    ]

    num_labels = 1000
    for key in candidate_keys:
        if key in sd_slim:
            tensor = sd_slim[key]
            # For Linear, weight is [out_features, in_features], bias is [out_features]
            num_labels = int(tensor.shape[0]) 
            break

    if num_labels is not None:
        old = getattr(cfg_slim, "num_labels", None)
        if old is not None and old != num_labels:
            print(f"[slim] Overriding config.num_labels: {old} -> {num_labels} (inferred from checkpoint)")
        cfg_slim.num_labels = int(num_labels)
    
        # Keep id2label/label2id consistent with num_labels
        if not getattr(cfg_slim, "id2label", None) or len(cfg_slim.id2label) != num_labels:
            cfg_slim.id2label = {str(i): f"LABEL_{i}" for i in range(num_labels)}
        if not getattr(cfg_slim, "label2id", None) or len(cfg_slim.label2id) != num_labels:
            cfg_slim.label2id = {v: k for k, v in cfg_slim.id2label.items()}
    else:
        print(f"[slim][warn] Could not infer num_labels from checkpoint; leaving config.num_labels={cfg_slim.num_labels}")
     

    # Tell HF which class to use when trust_remote_code=True
    cfg_slim.architectures = ["SlimViTForImageClassification"]
    cfg_slim.auto_map = {
        "AutoModel": "slim_vit.SlimViTForImageClassification",
        "AutoModelForImageClassification": "slim_vit.SlimViTForImageClassification",
    }

    cfg_slim.save_pretrained(slim_dir)

    # Save slim weights as a plain state_dict
    torch.save(sd_slim, slim_dir / "pytorch_model.bin")

    # Drop our custom SlimViT implementation into the repo
    (slim_dir / "slim_vit.py").write_text(SLIM_VIT_SOURCE, encoding="utf-8")
    (slim_dir / "custom_code.py").write_text(
        "# Marker file so Hub shows 'custom code' banner.\n", encoding="utf-8"
    )

    _save_model_card_and_meta(
        slim_dir,
        title=f"{repo_slim}",
        task="image-classification",
        meta={
            "base_id": base_id,
            "variant": "slim-export",
            "description": "Pruned ViT with per-layer head/FFN widths baked in.",
            "slim_num_heads": slim_num_heads,
            "slim_intermediate_sizes": slim_inter_sizes,
        },
        include_code=["slim_vit.py"],
        repo_slim=repo_slim,
    )

    if push:
        create_repo(repo_slim, token=token, private=private, exist_ok=True)
        upload_folder(repo_id=repo_slim, folder_path=str(slim_dir), token=token)
        print(f"[ok] Pushed slim student → {repo_slim}")


# def _infer_vit_config_from_state_dict(sd: dict, base_cfg: "AutoConfig") -> "AutoConfig":
#     """Infer a ViT config (num_heads, intermediate_size, num_labels, maybe hidden_size)
#     from a pruned state_dict, starting from base_cfg as a template.
#     """
#     cfg = deepcopy(base_cfg)

#     # ---- 1) attention heads / hidden size ----
#     # We look at layer 0 query weight: [all_head_size, hidden_size]
#     q_key = None
#     for k in sd.keys():
#         if "encoder.layer.0.attention.attention.query.weight" in k:
#             q_key = k
#             break
#     if q_key is not None:
#         q_w = sd[q_key]  # [all_head_size, hidden_size]
#         all_head_size, hidden_size = q_w.shape

#         # base head dim (usually hidden_size_base / num_heads_base)
#         base_head_dim = base_cfg.hidden_size // base_cfg.num_attention_heads
#         if all_head_size % base_head_dim == 0:
#             num_heads = all_head_size // base_head_dim
#         else:
#             # fallback: keep base, but loggable if you want
#             num_heads = base_cfg.num_attention_heads

#         cfg.hidden_size = int(hidden_size)
#         cfg.num_attention_heads = int(num_heads)

#     # ---- 2) FFN intermediate size ----
#     inter_key = None
#     for k in sd.keys():
#         if "encoder.layer.0.intermediate.dense.weight" in k:
#             inter_key = k
#             break
#     if inter_key is not None:
#         inter_w = sd[inter_key]  # [intermediate_size, hidden_size]
#         intermediate_size, _ = inter_w.shape
#         cfg.intermediate_size = int(intermediate_size)

#     # ---- 3) classifier output size (num_labels) ----
#     cls_key = None
#     for k in sd.keys():
#         if k.endswith("classifier.weight"):
#             cls_key = k
#             break
#     if cls_key is not None:
#         cls_w = sd[cls_key]  # [num_labels, hidden_size]
#         num_labels, _ = cls_w.shape
#         cfg.num_labels = int(num_labels)

#         # keep label2id/id2label in sync if it exists
#         if hasattr(cfg, "label2id") and isinstance(cfg.label2id, dict):
#             # shrink or rebuild mapping
#             labels = list(cfg.label2id.keys())
#             labels = labels[:num_labels] or [str(i) for i in range(num_labels)]
#             cfg.label2id = {lbl: i for i, lbl in enumerate(labels)}
#             cfg.id2label = {i: lbl for lbl, i in cfg.label2id.items()}

#     return cfg

# def _load_vit_state_dict_and_config(
#     ckpt_path: Path,
#     base_id: str,
# ) -> Tuple[dict, "AutoConfig"]:
#     """
#     Normalize a ViT checkpoint into (state_dict, config).

#     - If ckpt is a full nn.Module: use its .state_dict() and .config if present.
#     - If ckpt is already a state_dict: infer config (num_heads, FFN width, num_labels)
#       from shapes, starting from the base config.
#     """
#     obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)

#     base_cfg = AutoConfig.from_pretrained(base_id, trust_remote_code=True)

#     # Full module case
#     if isinstance(obj, nn.Module):
#         sd = obj.state_dict()
#         cfg = getattr(obj, "config", None)
#         if cfg is None:
#             # not HF-style, just fall back to base
#             cfg = base_cfg
#         return sd, cfg

#     # Raw state_dict case
#     if isinstance(obj, dict):
#         # Heuristic: pure param dict -> infer pruned config
#         if all(isinstance(v, torch.Tensor) for v in obj.values()):
#             sd = obj
#             cfg = _infer_vit_config_from_state_dict(sd, base_cfg)
#             return sd, cfg

#     raise TypeError(f"Unexpected checkpoint object type at {ckpt_path}: {type(obj)}")


# def export_vit_variants(
#     base_id: str,
#     student_ckpt: Path,
#     slim_ckpt: Path,
#     repo_gated: str,
#     repo_slim: str,
#     token: Optional[str],
#     private: bool,
#     include_code: List[str],
#     push: bool = True,
# ):
#     api = HfApi(token=token)

#     base_cfg = AutoConfig.from_pretrained(base_id, trust_remote_code=True)

#     # ---------- (a) Gated student: state_dict + custom code ----------
#     gated_dir = Path("hf_export_gated_vit")
#     if gated_dir.exists():
#         shutil.rmtree(gated_dir)
#     _ensure_dir(gated_dir)

#     base_cfg.save_pretrained(gated_dir)

#     sd_gated = torch.load(student_ckpt, map_location="cpu", weights_only=False)
#     torch.save(sd_gated, gated_dir / "pytorch_model.bin")

#     if include_code:
#         _copy_code_tree(gated_dir, include_code)
#         (gated_dir / "custom_code.py").write_text(
#             "# Marker file so Hub shows 'custom code' banner.\n",
#             encoding="utf-8",
#         )

#     _save_model_card_and_meta(
#         gated_dir,
#         title=repo_gated,
#         task="image-classification",
#         meta={"base_id": base_id, "variant": "gated-student"},
#         include_code=include_code,
#         repo_slim=repo_slim,
#     )

#     if push:
#         create_repo(repo_gated, token=token, private=private, exist_ok=True)
#         upload_folder(repo_id=repo_gated, folder_path=str(gated_dir), token=token)
#         print(f"[ok] Pushed gated student → {repo_gated}")

#     # ---------- (b) Slim model: upload full module as custom model ----------
#     slim_dir = Path("hf_export_slim_vit")
#     if slim_dir.exists():
#         shutil.rmtree(slim_dir)
#     _ensure_dir(slim_dir)

#     obj = torch.load(slim_ckpt, map_location="cpu", weights_only=False)

#     if isinstance(obj, nn.Module):
#         mdl = obj
#         cfg_slim = getattr(mdl, "config", None) or base_cfg
#         cfg_slim.save_pretrained(slim_dir)

#         # Save as HF-style model; this will create config + model weights
#         mdl.save_pretrained(slim_dir, safe_serialization=True)

#         # Optionally also drop raw torch model (not strictly needed)
#         # torch.save(mdl, slim_dir / "pytorch_model.bin")

#         # We *do not* need additional custom code if everything is pure HF;
#         # but if your pruned model still refers to adapter classes, copy them:
#         if include_code:
#             _copy_code_tree(slim_dir, include_code)
#             (slim_dir / "custom_code.py").write_text(
#                 "# Marker file so Hub shows 'custom code' banner for slim model.\n",
#                 encoding="utf-8",
#             )

#         _save_model_card_and_meta(
#             slim_dir,
#             title=repo_slim,
#             task="image-classification",
#             meta={"base_id": base_id, "variant": "slim-export"},
#             include_code=include_code,
#             repo_slim=repo_slim,
#         )

#         if push:
#             create_repo(repo_slim, token=token, private=private, exist_ok=True)
#             upload_folder(repo_id=repo_slim, folder_path=str(slim_dir), token=token)
#             print(f"[ok] Pushed slim student → {repo_slim}")

#     else:
#         # Fallback: pure state_dict -> we *can't* reconstruct a standard ViT,
#         # so treat it as a custom model that user has to re-wrap manually.
#         slim_dir = Path("hf_export_slim_vit_state_dict")
#         if slim_dir.exists():
#             shutil.rmtree(slim_dir)
#         _ensure_dir(slim_dir)

#         base_cfg.save_pretrained(slim_dir)
#         torch.save(obj, slim_dir / "pytorch_model.bin")

#         if include_code:
#             _copy_code_tree(slim_dir, include_code)
#             (slim_dir / "custom_code.py").write_text(
#                 "# Custom code required to reconstruct the pruned ViT from this state_dict.\n",
#                 encoding="utf-8",
#             )

#         _save_model_card_and_meta(
#             slim_dir,
#             title=repo_slim,
#             task="image-classification",
#             meta={
#                 "base_id": base_id,
#                 "variant": "slim-state-dict",
#                 "note": "state_dict only; use HAda ViTAdapter.export_pruned-style loader.",
#             },
#             include_code=include_code,
#             repo_slim=repo_slim,
#         )

#         if push:
#             create_repo(repo_slim, token=token, private=private, exist_ok=True)
#             upload_folder(repo_id=repo_slim, folder_path=str(slim_dir), token=token)
#             print(f"[ok] Pushed slim state_dict → {repo_slim}")

        

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

    # Load base config & tokenizer (used as template)
    cfg = AutoConfig.from_pretrained(base_id, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(base_id, use_fast=True)

    # -------------------------------------------------------------------------
    # (a) Gated student (custom code, state_dict only)
    # -------------------------------------------------------------------------
    gated_dir = Path("hf_export_gated_llama")
    if gated_dir.exists():
        shutil.rmtree(gated_dir)
    _ensure_dir(gated_dir)

    tok.save_pretrained(gated_dir)
    cfg.save_pretrained(gated_dir)

    obj_student = torch.load(student_ckpt, map_location="cpu", weights_only=False)

    # Normalize to pure state_dict for robustness
    if isinstance(obj_student, torch.nn.Module):
        sd_student = obj_student.state_dict()
    elif isinstance(obj_student, dict):
        # assume it's already a state_dict-like mapping
        sd_student = obj_student
    else:
        raise TypeError(
            f"[gated] Unexpected object type in {student_ckpt}: {type(obj_student)}. "
            "Expected nn.Module or state_dict-like dict."
        )

    torch.save(sd_student, gated_dir / "pytorch_model.bin")

    if include_code:
        _copy_code_tree(gated_dir, include_code)
        (gated_dir / "custom_code.py").write_text(
            "# Marker file so Hub shows 'custom code' banner for gated LLaMA.\n",
            encoding="utf-8",
        )

    _save_model_card_and_meta(
        gated_dir,
        title=repo_gated,
        task="causal-lm",
        meta={"base_id": base_id, "variant": "gated-student"},
        include_code=include_code,
        repo_slim=repo_slim,
    )

    if push:
        create_repo(repo_gated, token=token, private=private, exist_ok=True)
        upload_folder(repo_id=repo_gated, folder_path=str(gated_dir), token=token)
        print(f"[ok] Pushed gated LLaMA student → {repo_gated}")

    # -------------------------------------------------------------------------
    # (b) Slim model (HF-compatible SlimLlamaForCausalLM + custom code)
    # -------------------------------------------------------------------------
    slim_dir = Path("hf_export_slim_llama")
    if slim_dir.exists():
        shutil.rmtree(slim_dir)
    _ensure_dir(slim_dir)

    tok.save_pretrained(slim_dir)
    cfg.save_pretrained(slim_dir)

    # Load slim checkpoint: accept either full model or pure state_dict
    mdl = load_slim_llama(slim_ckpt, base_id, device="cpu")

    # # Build a base model then load the pruned weights
    # mdl = AutoModelForCausalLM.from_pretrained(
    #     base_id, torch_dtype="auto", trust_remote_code=True
    # )
    # missing, unexpected = mdl.load_state_dict(sd_slim, strict=False)
    # print(f"[slim] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")

    mdl.save_pretrained(slim_dir, safe_serialization=False)

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
    sd = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
    ap.add_argument("--task", choices=["llama", "vit", "resnet"], required=True,
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
    elif args.task == "vit":
        export_vit_variants(
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
