# Overview

`examples/` contains runnable scripts that demonstrate end-to-end optimization for each supported family.

# Scripts

`run_resnet_optimize.py` — ResNet hardware-aware gate training + export flow

`run_vit_optimize.py` — ViT hardware-aware gate training + export flow

`run_llama_optimize.py` — LLaMA hardware-aware gate training + export flow

These scripts typically:

* load a base model
* attach the appropriate adapter
* train gates using core/train.py
* optionally run export search via core/search_export.py
* export a slim model and evaluate