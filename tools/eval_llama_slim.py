# eval_llama_slim.py

import torch
from transformers import AutoTokenizer

from lm_eval import api
from lm_eval.models.huggingface import HFLM  # we can reuse their HF wrapper

from adapters.huggingface.llama import SlimLlamaForCausalLM, load_slim_llama
import torch
from transformers import AutoTokenizer, AutoConfig


def load_llama_slim(slim_dir: str, base_id: str) -> api.LM:
    """
    Entry point for lm-eval 'python' loader.
    `pretrained` will be the base model id, e.g. 'meta-llama/Llama-3.2-1B'.
    Extra kwargs can include paths to slim weights / meta.
    """
    device = kwargs.pop("device", "cuda")


    save_dir = f"{slim_dir}/slim_hf"
    
    config = AutoConfig.from_pretrained(base_id)
    config.architectures = ["SlimLlamaForCausalLM"]
    config.auto_map = {
        "AutoModelForCausalLM": "adapters.huggingface.llama.SlimLlamaForCausalLM"
    }
    

    tok = AutoTokenizer.from_pretrained(base_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    slim = load_slim_llama(slim_dir, pretrained, device=device)
    slim.to(device)
    slim.eval()
    
    slim.save_pretrained(save_dir)
    tok.save_pretrained(save_dir)

    lm = HFLM(pretrained=pretrained, tokenizer=tok, model=model, device=device)

    return lm




