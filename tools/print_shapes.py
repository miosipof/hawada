import argparse
from typing import Tuple, Optional

import torch
from transformers import AutoModelForCausalLM


def guess_layer_prefix(sd_keys) -> str:
    """
    Guess whether keys look like:
      - 'model.layers.0.self_attn.q_proj.weight'
      - 'layers.0.self_attn.q_proj.weight'
    Returns the prefix before 'layers', e.g. 'model' or ''.
    """
    patterns = [
        "layers.0.self_attn.q_proj.weight",
        "layers.0.self_attn.k_proj.weight",
        "layers.0.mlp.up_proj.weight",
    ]

    for k in sd_keys:
        for pattern in patterns:
            idx = k.find(pattern)
            if idx != -1:
                prefix = k[:idx].rstrip(".")
                return prefix
    # Fallback: assume 'model'
    return "model"


def get_sd_tensor(sd, prefix: str, layer_idx: int, suffix: str) -> Optional[torch.Tensor]:
    """
    Build a key like:
      '{prefix}.layers.{i}.{suffix}'
    or
      'layers.{i}.{suffix}' if prefix == ''.
    """
    if prefix:
        key = f"{prefix}.layers.{layer_idx}.{suffix}"
    else:
        key = f"layers.{layer_idx}.{suffix}"

    return sd.get(key, None)


def shape(t: Optional[torch.Tensor]) -> Optional[Tuple[int, ...]]:
    return tuple(t.shape) if t is not None else None


def ratio_str(slim_dim: int, dense_dim: int) -> str:
    if dense_dim == 0:
        return "n/a"
    return f"{slim_dim}/{dense_dim} = {slim_dim / dense_dim:.1%}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--slim_state",
        type=str,
        required=True,
        help="Path to slim state_dict file (e.g. runs/llama3p2_1b/slim.pt)",
    )
    parser.add_argument(
        "--dense_id",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="HF model ID for vanilla LLaMA (dense reference)",
    )
    args = parser.parse_args()

    print(f"[load] dense model: {args.dense_id}")
    dense = AutoModelForCausalLM.from_pretrained(args.dense_id, torch_dtype=torch.float32)
    core = getattr(dense, "model", dense)
    layers = core.layers
    num_layers = len(layers)

    head_dim = layers[0].self_attn.head_dim
    hidden_size = dense.config.hidden_size

    print(f"[info] hidden_size = {hidden_size}, head_dim = {head_dim}, num_layers = {num_layers}")

    print(f"[load] slim state_dict: {args.slim_state}")
    slim_sd = torch.load(args.slim_state, map_location="cpu")
    prefix = guess_layer_prefix(list(slim_sd.keys()))
    print(f"[info] guessed slim prefix before 'layers': '{prefix}'")

    print("\n================ LAYER-BY-LAYER COMPARISON ================\n")

    for i, layer in enumerate(layers):
        attn = layer.self_attn
        mlp = layer.mlp

        print(f"---------------- Layer {i} ----------------")

        # ----- Attention tensors -----
        dense_q = attn.q_proj.weight
        dense_k = attn.k_proj.weight
        dense_v = attn.v_proj.weight
        dense_o_mod = getattr(attn, "o_proj", None)
        if dense_o_mod is None:
            dense_o_mod = getattr(attn, "out_proj", None)
        dense_o_shape = tuple(dense_o_mod.weight.shape) if dense_o_mod is not None else None

        slim_q = get_sd_tensor(slim_sd, prefix, i, "self_attn.q_proj.weight")
        slim_k = get_sd_tensor(slim_sd, prefix, i, "self_attn.k_proj.weight")
        slim_v = get_sd_tensor(slim_sd, prefix, i, "self_attn.v_proj.weight")

        slim_o = get_sd_tensor(slim_sd, prefix, i, "self_attn.o_proj.weight")
        if slim_o is None:
            slim_o = get_sd_tensor(slim_sd, prefix, i, "self_attn.out_proj.weight")

        print(f"Q_proj: dense {tuple(dense_q.shape)} | slim {shape(slim_q)}")
        print(f"K_proj: dense {tuple(dense_k.shape)} | slim {shape(slim_k)}")
        print(f"V_proj: dense {tuple(dense_v.shape)} | slim {shape(slim_v)}")
        print(f"O_proj: dense {dense_o_shape} | slim {shape(slim_o)}")

        dense_Hq = attn.num_heads
        dense_Hkv = attn.num_key_value_heads
        dense_head_dim = attn.head_dim

        print(f"meta (dense): num_heads={dense_Hq}, num_kv_heads={dense_Hkv}, head_dim={dense_head_dim}")

        if slim_q is not None:
            slim_Hq = slim_q.shape[0] // dense_head_dim
            print(f"  → inferred slim num_heads ≈ {slim_Hq} "
                  f"({ratio_str(slim_Hq, dense_Hq)})")
        else:
            print("  → slim Q_proj missing, cannot infer heads")

        # ----- FFN tensors -----
        dense_up = mlp.up_proj.weight
        dense_gate = mlp.gate_proj.weight
        dense_down = mlp.down_proj.weight

        slim_up = get_sd_tensor(slim_sd, prefix, i, "mlp.up_proj.weight")
        slim_gate = get_sd_tensor(slim_sd, prefix, i, "mlp.gate_proj.weight")
        slim_down = get_sd_tensor(slim_sd, prefix, i, "mlp.down_proj.weight")

        print(f"up_proj:   dense {tuple(dense_up.shape)}   | slim {shape(slim_up)}")
        print(f"gate_proj: dense {tuple(dense_gate.shape)} | slim {shape(slim_gate)}")
        print(f"down_proj: dense {tuple(dense_down.shape)} | slim {shape(slim_down)}")

        if slim_up is not None:
            dense_exp = dense_up.shape[0]
            slim_exp = slim_up.shape[0]
            print(f"  → FFN expansion keep ratio: {ratio_str(slim_exp, dense_exp)}")

        print()

    print("============================================================")
    print("Done.")


if __name__ == "__main__":
    main()
