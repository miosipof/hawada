# Overview

`recipes/` contains GPU-specific YAML recipes that define optimization targets and hyperparameters for each model family.

# Layout
```
recipes/
├── RTX4090/
│   ├── resnet18_imagenet224.yaml
│   ├── vit_base_patch16_224.yaml
│   └── llama_3_2_1b.yaml
└── H100/
    ├── resnet18_imagenet224.yaml
    ├── vit_base_patch16_224.yaml
    └── llama_3_2_1b.yaml
```

Each recipe typically specifies:

* latency target
* gate/penalty settings
* KD temperature / MSE weight
* export rounding constraints (e.g., head multiples)
* batch shapes and measurement settings