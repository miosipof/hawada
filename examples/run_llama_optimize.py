#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_llama_optimize.py

End-to-end “gate + KD + latency” optimization for HF LLaMA/Mistral-style models,
using the family-agnostic core (export, train, proxy) and your LLaMA adapter.
"""
from __future__ import annotations

import sys
import math
import json
import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import yaml

# Make repo root importable
sys.path.append(str(Path(__file__).resolve().parent))




# ------------------------
# Adapter & Data
# ------------------------
from adapters.huggingface.llama import (
    LlamaAdapter,
    LlamaGatingConfig,
    LlamaExportPolicy,
    LatencyProxyLLM,
    load_slim_for_finetune
)
from data.llms import build_llm_dataloaders_from_cfg

# ------------------------
# Core (family-agnostic)
# ------------------------
from core.train import LagrangeTrainer, TrainerConfig, DualConfig
from core.distill import KDConfig
from core.gates import PenaltyWeights, Constraints
from core.export import Rounding as CoreRounding
from core.profiler import measure_latency_text_ms  # (B,S,T)-aware timing

# HF
from transformers import AutoModelForCausalLM, AutoTokenizer



# ------------------------
# Small utilities
# ------------------------
def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_json(obj, path: Path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def _pick_id(model_cfg: Dict[str, Any], fallback_key: str = "hf_id") -> str:
    # Robust HF id retrieval (mirrors ViT pattern)
    return (
        model_cfg.get("student_name_or_path")
        or model_cfg.get("name_or_path")
        or model_cfg.get("name")
        or model_cfg.get(fallback_key)
    )

# ------------------------
# Batch extractors (dict/tuple/ids)
# ------------------------
def _ids_mask(batch) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(batch, dict):
        ids = batch["input_ids"]
        mask = batch.get("attention_mask", None)
    elif isinstance(batch, (list, tuple)):
        ids = batch[0]
        mask = batch[1] if len(batch) > 1 else None
    else:
        ids = batch
        mask = None
    return ids, mask


# ------------------------
# A tiny bridge: generic trainer expects proxy.predict(student, batch),
# but LLM proxy wants (B,S,T)
# ------------------------
class _ProxyBridge:
    def __init__(self, inner_llm_proxy: LatencyProxyLLM, decode_T: int):
        self.inner = inner_llm_proxy
        self.decode_T = int(decode_T)
        self.scale_ms = inner_llm_proxy.scale_ms

    def predict(self, model, batch):
        ids, _ = _ids_mask(batch)
        B, S = int(ids.size(0)), int(ids.size(1))
        return self.inner.predict(model, B=B, S=S, T=self.decode_T)


# ------------------------
# Argparse
# ------------------------
def build_argparser():
    ap = argparse.ArgumentParser(description="Optimize LLaMA (gating + KD + latency proxy).")
    ap.add_argument("--recipe", type=str, required=True, help="YAML recipe path (e.g., recipes/llama_3_2_1b.yaml)")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--epochs", type=int, default=None, help="Override epochs in recipe")
    ap.add_argument("--outdir", type=str, default="runs/llama_opt")
    ap.add_argument("--save", type=str, default=None, help="Optional checkpoint file name (default auto)")
    ap.add_argument("--real-every", type=int, default=None, help="Override frequency of real latency checks")
    ap.add_argument("--finetune", type=bool, default=True)        
    ap.add_argument("--slim", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", type=str, default="bf16", choices=["off", "fp16", "bf16"])
    ap.add_argument("--calibrate-proxy", action="store_true", help="Calibrate proxy.scale_ms to keep-all latency")
    ap.add_argument("--export-only", action="store_true", help="Skip training; just export with policy")
    return ap


# ------------------------
# Main
# ------------------------
def main():
    args = build_argparser().parse_args()
    torch.manual_seed(args.seed)
    torch.autograd.set_detect_anomaly(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")    

    from torch.backends.cuda import sdp_kernel    
    sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)


    cfg = load_yaml(args.recipe)
    device = args.device

    # -------- model ids / config --------
    model_cfg = cfg.get("model", {})
    student_id = _pick_id(model_cfg)  # student + teacher same by default
    teacher_id = model_cfg.get("teacher_name_or_path") or student_id

    # -------- data loaders --------
    data_cfg = cfg.get("data", {})
    train_loader, val_loader, tok_s, tok_t, st_vocab_map = build_llm_dataloaders_from_cfg(data_cfg)

    if args.slim is None:
        # -------- build models --------
        print(f"[load] teacher: {teacher_id}")
        teacher = AutoModelForCausalLM.from_pretrained(teacher_id)
        print(f"[load] student: {student_id}")
        student = AutoModelForCausalLM.from_pretrained(student_id)
    
        # kill attention & residual dropouts on the student
        if hasattr(student, "config"):
            if hasattr(student.config, "attention_dropout"):
                student.config.attention_dropout = 0.0
            if hasattr(student.config, "hidden_dropout_prob"):
                student.config.hidden_dropout_prob = 0.0
        for m in student.modules():
            if hasattr(m, "attention_dropout"):
                try: m.attention_dropout = 0.0
                except: pass
            if isinstance(m, torch.nn.Dropout):
                m.p = 0.0
                
        if hasattr(student, "config"):
            student.config.use_cache = False
        try:
            student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except Exception:
            try: student.gradient_checkpointing_enable()
            except: pass
    
        
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        teacher.eval().to(device)
        for p in teacher.parameters():
            p.requires_grad_(False)
    
        # -------- attach gates to STUDENT via adapter --------
        adapter = LlamaAdapter(student)
        gate_cfg = LlamaGatingConfig(
            tau=float(model_cfg.get("gating", {}).get("tau", 1.5)),
            init_logit=float(model_cfg.get("gating", {}).get("init_logit", 3.0)),
            head_gating=bool(model_cfg.get("gating", {}).get("head_gating", True)),
            gate_kv=bool(model_cfg.get("gating", {}).get("gate_kv", False)),
            ffn_group=int(model_cfg.get("gating", {}).get("ffn_group", 128)),
            ffn_gating=bool(model_cfg.get("gating", {}).get("ffn_gating", True)),
            hard_eval=bool(model_cfg.get("gating", {}).get("hard_eval", True)),
        )
        student = adapter.attach_gates(gate_cfg).train().to(device)
    
        # -------- latency proxy (bridge to generic interface) --------
        lat_cfg = cfg.get("latency", {})
        decode_T = int(lat_cfg.get("decode_T_tokens", 128))
    
        llm_proxy = LatencyProxyLLM(gate_kv_in_proxy=bool(lat_cfg.get("proxy_gate_kv", False)))
        proxy = _ProxyBridge(llm_proxy, decode_T=decode_T)
    
        # -------- export policies --------
        export_cfg = cfg.get("export", {})
    
        # Eval policy used during training/probes (permissive)
        policy_eval = LlamaExportPolicy(
            warmup_steps=0,
            head_rounding=CoreRounding(
                floor_groups=1,
                multiple_groups=1,
                min_keep_ratio=0.5,
            ),
            q_rounding=CoreRounding(
                floor_groups=1,
                multiple_groups=1,
                min_keep_ratio=0.5,
            ),          
            ffn_rounding=CoreRounding(
                floor_groups=1,
                multiple_groups=1,
                min_keep_ratio=0.5,
            ),
        )
    
        # Final export policy after training (stricter + kernel-friendly)
        export_policy_final = LlamaExportPolicy(
            warmup_steps=int(export_cfg.get("warmup_steps", 5)),
            head_rounding=CoreRounding(
                floor_groups=int(export_cfg.get("heads",{}).get("floor", 4)),
                multiple_groups=int(export_cfg.get("heads",{}).get("multiple", 8)),
                min_keep_ratio=float(export_cfg.get("heads",{}).get("min_keep_ratio", 0.5)),
            ),
            q_rounding=CoreRounding(
                floor_groups=int(export_cfg.get("q",{}).get("floor", 4)),
                multiple_groups=int(export_cfg.get("q",{}).get("multiple", 8)),
                min_keep_ratio=float(export_cfg.get("q",{}).get("min_keep_ratio", 0.5)),
            ),        
            ffn_rounding=CoreRounding(
                floor_groups=int(export_cfg.get("ffn",{}).get("floor", 1)),
                multiple_groups=int(export_cfg.get("ffn",{}).get("multiple", 128)),
                min_keep_ratio=float(export_cfg.get("ffn",{}).get("min_keep_ratio", 0.5)),
            ),
        )
    
        # Proxy calibration
        batch = next(iter(train_loader))
        ids, _ = _ids_mask(batch)
        B, S = int(ids.size(0)), int(ids.size(1))
    
        # measure real keep-all
        slim_keepall = adapter.export_keepall(student).to(device).eval()
        real_ms, _, _ = measure_latency_text_ms(slim_keepall, B=B, S=S, T=decode_T, device=device)
        del slim_keepall
    
        # proxy's raw keep-all prediction (pre-scale)
        raw_pred = llm_proxy.predict(student, B=B, S=S, T=decode_T)
        raw_pred = float(raw_pred.detach().item() if hasattr(raw_pred, "detach") else raw_pred)
        raw_pred = max(raw_pred, 1e-9)
    
        llm_proxy.scale_ms = max(1e-9, float(real_ms) / raw_pred)
        print(f"[calib] keep-all measured ≈ {real_ms:.3f} ms; proxy.scale_ms set to {llm_proxy.scale_ms:.6e}")
    
    
    
        # -------- training knobs --------
        tr_cfg = cfg.get("trainer", {})
        epochs = args.epochs if args.epochs is not None else int(tr_cfg.get("epochs", 1))
        amp_choice = args.amp
        use_amp = (amp_choice != "off")
        real_every = float(tr_cfg.get("lagrange",{}).get("real_every", 50))
    
        tau_target_scale = float(tr_cfg.get("trainer",{}).get("lagrange",{}).get("tau_target_scale", 0.7))
        target_ms = real_ms * tau_target_scale
    
        kd_cfg = KDConfig(
            temperature=float(tr_cfg.get("kd",{}).get("temperature", 4.0)),
            alpha=float(tr_cfg.get("kd",{}).get("alpha", 2.0)),
        )
        penalties = PenaltyWeights(
            l0=float(tr_cfg.get("penalties",{}).get("l0", 1e-4)),
            keep_floor_ratio=float(tr_cfg.get("penalties",{}).get("keep_floor_ratio", 0.25)),
            bimodality=float(tr_cfg.get("penalties",{}).get("bimodality", 1e-6)),
        )
        constraints = Constraints(
            min_keep_ratio=float(tr_cfg.get("constraints",{}).get("min_keep_ratio", 0.25)),
            min_groups=float(tr_cfg.get("constraints",{}).get("min_groups", 0.25)),
            max_groups_drop=None,
        )
        trainer_cfg = TrainerConfig(
            kd=kd_cfg,
            penalties=penalties,
            constraints=constraints,
            latency_target_ms=target_ms,
            real_probe_every=real_every,
            probe_batch_override=tr_cfg.get("probe_batch_override", None),
            amp=use_amp,
            device=device,
            mse_weight=float(tr_cfg.get("mse_weight", 0.1)),
            lr_gate=float(tr_cfg.get("lagrange",{}).get("gate_lr", 5e-2)),
            lr_linear=float(tr_cfg.get("lagrange",{}).get("lr_linear", 1e-4)),
            lr_affine=float(tr_cfg.get("lagrange",{}).get("lr_affine", 3e-4)),
            wd_linear=float(tr_cfg.get("lagrange",{}).get("wd_linear", 1e-4)),
            use_grad_scaler=False,
            dual=DualConfig(lr=float(tr_cfg.get("lagrange",{}).get("lambda_lr", 0.05)), ema_beta=0.5, clip=10.0),
        )
    
        # -------- trainer (family-agnostic, with adapter callables) --------
        get_s = lambda model, batch: LlamaAdapter.get_logits(
            model,
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask", None),
            last_only=True,
        )
        
        get_t = lambda model, batch: LlamaAdapter.get_logits(
            model,
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask", None),
            last_only=True,
        )
    
        trainer = LagrangeTrainer(
            student=student,
            teacher=teacher,
            proxy=proxy,
            adapter_get_student_logits=get_s,
            adapter_get_teacher_logits=get_t,
            adapter_export_keepall=adapter.export_keepall, #LlamaAdapter.export_keepall,
            adapter_export_pruned=adapter.export_pruned,   #LlamaAdapter.export_pruned,
            export_policy=policy_eval,   # warmup/permissive during training & probes
            cfg=trainer_cfg,             # <<< single source of optimizer/loss/dual params
        )
    
        # -----------------------
        # Optional: export-only
        # -----------------------
        outdir = Path(args.outdir)
        ensure_dir(outdir)
        ckpt_path = outdir / (args.save or "llama_student_gated.pt")
    
        if args.export_only:
            print("[export-only] Skipping training; exporting directly with provided export policy.")
            slim = adapter.export_pruned(student, policy=export_policy_final, step=int(export_cfg.get("step", 0)))
            torch.save(slim.state_dict(), outdir / "slim_state_dict.pt")
            print(f"[export-only] Saved slim state_dict → {outdir/'slim_state_dict.pt'}")
            return
    
        # -----------------------
        # Train epochs
        # -----------------------
        print(f"[train] epochs={epochs} | target_ms={target_ms} | decode_T={decode_T}")
        for ep in range(1, epochs + 1):
            trainer.train_epoch(train_loader, verbose_every=int(tr_cfg.get("log_every", 50)))
            torch.save(student.state_dict(), ckpt_path)
            print(f"[ckpt] saved → {ckpt_path}")
    
        # -----------------------
        # Export pruned model
        # -----------------------
        slim = adapter.export_pruned(student, policy=export_policy_final, step=int(export_cfg.get("step", epochs * 100)))
        slim_path = outdir / "slim.pt"
        torch.save(slim.state_dict(), slim_path)
        print(f"[export] saved slim state_dict → {slim_path}")

    # args.slim is not None ==> load pre-trained pruned slim model
    else:
        
        tokenizer = AutoTokenizer.from_pretrained(student_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 2) Load slim model for training
        slim = load_slim_for_finetune(
            dense_id=student_id,
            slim_dir=args.slim,
            device=device,
            dtype=torch.bfloat16,
        )
        
        # print(slim.__class__.__name__)
        # print(slim.config)

    # -----------------------
    # Optional: fine tuning
    # -----------------------
    if args.finetune:

        # -------- build models --------
        print(f"[load] teacher: {teacher_id}")
        teacher = AutoModelForCausalLM.from_pretrained(teacher_id)
        adapter = LlamaAdapter(teacher)

        
        slim.train()
        optimizer = torch.optim.AdamW(slim.parameters(), lr=1e-5)
        
        # texts = ["Hello, this is a test.", "Another example prompt."]
        # batch = tokenizer(
        #     texts,
        #     padding=True,
        #     truncation=True,
        #     return_tensors="pt",
        # ).to(device)
        
        # # standard LM loss: labels = input_ids shifted inside HF
        # outputs = slim(**batch, labels=batch["input_ids"])
        # loss = outputs.loss
        # loss.backward()
        # optimizer.step()
        # optimizer.zero_grad()
        
        # print("Loss:", float(loss))

        teacher.to(device)
        slim.to(device)

        from core.train import _move_batch_to_device
        import torch.nn.functional as F


        # accum_steps = 32
        # optimizer.zero_grad()

        # max_steps = 100000

        # for idx, batch in enumerate(train_loader):

        #     batch = _move_batch_to_device(batch, device)
            
        #     with torch.no_grad():
        #         teacher_logits = adapter.get_logits(teacher, batch["input_ids"])

        #     student_outputs = slim(**batch, labels=batch["input_ids"], return_dict=True)
        #     student_logits = student_outputs.logits
        #     lm_loss = student_outputs.loss                

        #     T = 4.0 - idx/max_steps*3.0
        #     alpha = 0.1 # + idx/max_steps * 1.0
        #     beta  = 0.1 + idx/max_steps * 0.4
        #     kl_per_token = F.kl_div(
        #         F.log_softmax(student_logits / T, dim=-1),
        #         F.softmax(teacher_logits / T, dim=-1),
        #         reduction="none",
        #     ).sum(-1)  # (B, S)
            
        #     # mask padding
        #     mask = batch["attention_mask"].to(device)  # (B, S), 1 for real tokens
        #     kl_masked = kl_per_token * mask

        #     # mean over non-pad tokens
        #     kl_mean = kl_masked.sum() / mask.sum()
            
        #     # standard temperature scaling for distillation
        #     kl_mean = kl_mean * (T * T)     
        #     kl_mean = torch.clamp(kl_mean, max=10.0)

        #     # Add last-token agreement
        #     s_last = student_logits[:, -1, :]   # (B, V)
        #     t_last = teacher_logits[:, -1, :]   # (B, V)  # if adapter returns (B, S, V)
        #     kl_last = F.kl_div(
        #         F.log_softmax(s_last / T, dim=-1),
        #         F.softmax(t_last / T, dim=-1),
        #         reduction="batchmean",
        #     ) * (T * T)    
        #     kl_last = torch.clamp(kl_last, max=10.0)
            
        #     loss = lm_loss + alpha*kl_mean + beta*kl_last
            
        #     (loss / accum_steps).backward()
        #     if (idx + 1) % accum_steps == 0:
        #         optimizer.step()
        #         optimizer.zero_grad()

            
        #     optimizer.step()
        #     optimizer.zero_grad()

        #     if idx % 1000 == 0:
        #         print(f"[Batch {idx}, alpha={alpha:.2f}, beta={beta:.2f}, T={T:.1f}]: loss = {loss.item():.4f} = {lm_loss.item():.4f}[LM] "
        #               f"+ {alpha:.1f}*{kl_mean.item():.4f}[KL] + {beta:.1f}*{kl_last.item():.4f}[KL last]")

        #     if (idx+1) % 5000 == 0:
        #         torch.save(slim, f"runs/llama3p2_1b/slim_finetune_step_{idx+1}.pt")
            
        #     if idx == max_steps:
        #         break

        
        # torch.save(slim, "runs/llama3p2_1b/slim_finetune.pt")

        del slim
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        slim = torch.load("runs/llama3p2_1b/slim_finetune_step_30000.pt", map_location=device, weights_only=False)
        slim.eval()

        max_steps = 500
        kl_sum, n_batches = 0.0, 0
        agree_top1, n_tokens = 0, 0        
        for idx, batch in enumerate(val_loader):
            
            batch = _move_batch_to_device(batch, device)
                        
            with torch.no_grad():
                t = adapter.get_logits(teacher, batch["input_ids"])  # see shape below
            
            with torch.no_grad():
                student_outputs = slim(**batch, labels=batch["input_ids"], return_dict=True)
                s = student_outputs.logits   # (B, S, V)
            
            attention_mask = batch["attention_mask"].to(device)  # (B, S)
            last_mask = attention_mask[:, -1].bool()  # (B,)
            
            # --- KL on last token only (recommended for agreement metric) ---
            
            T = 1.0
            
            # student last token logits: (B, V)
            s_last = s[:, -1, :]  # (B, V)
            
            # teacher last token logits:
            if t.dim() == 3:        # (B, S, V)
                t_last = t[:, -1, :]
            elif t.dim() == 2:      # (B, V), adapter already returns last token
                t_last = t
            else:
                raise ValueError(f"Unexpected teacher logits shape: {t.shape}")
            
            kl = F.kl_div(
                F.log_softmax(s_last / T, dim=-1),
                F.softmax(t_last / T, dim=-1),
                reduction="batchmean",
            ) * (T * T)
            
            kl_sum += kl.item()
            n_batches += 1
            
            # --- Top-1 argmax agreement on last token only ---
            
            s_arg_last = s_last.argmax(-1)  # (B,)
            t_arg_last = t_last.argmax(-1)  # (B,)
            
            valid = last_mask  # (B,)
            
            agree_top1 += (s_arg_last[valid] == t_arg_last[valid]).sum().item()
            n_tokens  += valid.sum().item()
    
            if idx == max_steps:
                break

        
        print("kl_mean", kl_sum / max(1, n_batches))
        print("top1_agreement", agree_top1 / max(1, n_tokens))
                
            
def linear_shape(mod):
    if mod is None:
        return None
    w = mod.weight
    return tuple(w.shape)


def print_llama_layer_info(model, label="MODEL"):
    """Print Q/K/V/O and FFN sizes for each layer."""
    core = getattr(model, "model", model)
    layers = core.layers

    print("=" * 60)
    print(f" MODEL SHAPE REPORT: {label}")
    print("=" * 60)

    for i, layer in enumerate(layers):
        attn = layer.self_attn
        mlp  = layer.mlp

        print(f"\n---- Layer {i} ----")

        # Attention projections
        print(f"  Q_proj: {linear_shape(attn.q_proj)}")
        print(f"  K_proj: {linear_shape(attn.k_proj)}")
        print(f"  V_proj: {linear_shape(attn.v_proj)}")

        # Depending on HF version, o_proj is either "o_proj" or "out_proj"
        o_proj = getattr(attn, "o_proj", None)
        if o_proj is None:
            o_proj = getattr(attn, "out_proj", None)
        print(f"  O_proj: {linear_shape(o_proj)}")

        # Metadata
        print(f"  num_heads:           {attn.num_heads}")
        if hasattr(attn, "num_key_value_heads"):
            print(f"  num_kv_heads:        {attn.num_key_value_heads}")
        if hasattr(attn, "head_dim"):
            print(f"  head_dim:            {attn.head_dim}")

        # FFN projections (SwiGLU)
        print(f"  up_proj:   {linear_shape(mlp.up_proj)}")
        print(f"  gate_proj: {linear_shape(mlp.gate_proj)}")
        print(f"  down_proj: {linear_shape(mlp.down_proj)}")

    print("\nDone.\n")





    # optional: measure latency on a real batch
    try:
        batch = next(iter(val_loader))
        ids, _ = _ids_mask(batch)
        B, S = int(ids.size(0)), int(ids.size(1))
        mean_ms, p95_ms, std = measure_latency_text_ms(slim.to(device).eval(), B=B, S=S, T=decode_T, device=device)
        print(f"[latency/slim] mean={mean_ms:.3f}ms | p95={p95_ms:.3f}ms | std={std:.3f} | (B={B}, S={S}, T={decode_T})")
    except Exception as e:
        print(f"[latency] skipping measure: {e}")


    print_llama_layer_info(slim, label="SLIM MODEL")

    keep = adapter.export_keepall(student)
    print_llama_layer_info(keep, label="KEEP MODEL")





if __name__ == "__main__":
    main()
