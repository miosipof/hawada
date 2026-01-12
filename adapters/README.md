# `adapters/` — HawAda Gating & Model Injection

`adapters/` contains model-family integrations that connect HugAda’s family-agnostic core/ to concrete model implementations (Hugging Face LLaMA/ViT, Torchvision ResNet).

Adapters are responsible for:

* Attaching gates to the right submodules (heads / FFN groups / channels)
* Providing get_student_logits / get_teacher_logits for KD
* Implementing export functions:
    * keep-all: remove wrappers, restore clean modules
    * pruned: slice weights and update model metadata

## Layout

```
adapters/
├── huggingface/
│   ├── llama.py   # HF causal LM (LLaMA/Mistral-like)
│   └── vit.py     # HF ViT
└── torchvision/
    └── resnet.py  # Torchvision ResNet
```

Hugging Face adapters
---------------------

### huggingface/llama.py

Bridges core/\* to HF causal LMs (e.g. LlamaForCausalLM).

Implements:

*   gating for attention **Q heads** (optional KV gating) and grouped MLP (SwiGLU)
    
*   logits getters for student/teacher
    
*   exporters that:
    
    *   slice q\_proj, o\_proj
        
    *   slice SwiGLU up/gate/down
        
    *   update HF config/metadata appropriately
        
*   optional grid-search wrapper via core.search\_export.grid\_search\_latency
    

### huggingface/vit.py

Implements gating/export for HF ViT models (e.g. patch16/224 variants), including kernel-aligned rounding when exporting.

Torchvision adapters
--------------------

### torchvision/resnet.py

Implements gating/export logic for torchvision ResNet variants.

Adapter contract expected by core/train.py
------------------------------------------

core/train.LagrangeTrainer expects the adapter to supply callables:

*   adapter\_get\_student\_logits(model, batch) -> logits
    
*   adapter\_get\_teacher\_logits(model, batch) -> logits
    
*   adapter\_export\_keepall(model\_with\_gates) -> clean\_model
    
*   adapter\_export\_pruned(model\_with\_gates, policy, step) -> pruned\_model