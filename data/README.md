# Overview

`data/` contains lightweight dataset/dataloader helpers used by the notebooks and example scripts.

# Files

`vision.py` — Vision datasets / transforms helpers (e.g., ImageNet-style 224 pipelines)

`llms.py` — Text/LLM data utilities (tokenization/batching helpers for causal LM distillation setups)

The goal is to keep core/ independent from datasets, while still providing reusable loaders for the supported model families.