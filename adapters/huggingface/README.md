Overview
--------

The adapters/huggingface module provides **HawAda adapters for Hugging Face Transformer models**, currently supporting:

*   Vision Transformers (ViT)
    
*   Causal language models (LLaMA-style)
    

These adapters enable **fine-grained structural slimming** (heads, FFN width) while maintaining compatibility with the Hugging Face ecosystem.

Supported models
----------------

*   **ViT-Base / Patch16**
    
*   **LLaMA-3.2-1B** (and compatible architectures)
    

Shared design principles
------------------------

All Hugging Face adapters:

*   Preserve HF model APIs (forward, generate, etc.)
    
*   Avoid modifying transformers internals
    
*   Ensure exported models load via from\_pretrained
    
*   Handle config and metadata updates correctly
    

llama.py — LLaMA adapter
------------------------

### Gating strategy

The LLaMA adapter introduces **structured gates** at:

*   **Attention heads**
    
    *   Gates applied to Q-projection heads
        
*   **MLP blocks**
    
    *   SwiGLU up/gate/down projections grouped and gated consistently
        

This allows HawAda to learn:

*   Which heads contribute meaningfully
    
*   Which FFN dimensions can be removed
    

### Training interface

Implements:

*   get\_student\_logits(model, batch)
    
*   get\_teacher\_logits(model, batch)
    

These return logits compatible with:

*   KL divergence
    
*   Optional MSE regularization
    

### Export logic

#### Keep-all export

*   Removes adapters
    
*   Restores clean HF LLaMA modules
    

#### Pruned export

*   Selects head and FFN indices using trained gates
    
*   Uses core/export.py utilities:
    
    *   slice\_linear
        
    *   slice\_embedding
        
*   Updates:
    
    *   num\_attention\_heads
        
    *   hidden\_size
        
    *   intermediate sizes
        
*   Preserves:
    
    *   tied embeddings
        
    *   weight sharing semantics
        

### GPU-aware rounding

Export optionally integrates with:

*   core/search\_export.py
    

This ensures:

*   Head counts align with GPU-friendly multiples
    
*   FFN dimensions are kernel-efficient
    

vit.py — Vision Transformer adapter
-----------------------------------

### Gating strategy

ViT adapters gate:

*   **Attention heads**
    
*   **MLP hidden dimensions**
    

This enables structured slimming while preserving:

*   Patch embeddings
    
*   Positional encodings
    
*   Class tokens
    

### Export logic

*   Slices attention and MLP projections
    
*   Updates ViT config fields
    
*   Preserves HF model compatibility
    

Interaction with Hugging Face
-----------------------------

Exported models:

*   Load via AutoModel.from\_pretrained
    
*   Support standard HF trainers
    
*   Can be uploaded to the Hub without custom code
    

Extension guide
---------------

### Supporting a new HF architecture

To add a new HF model:

1.  Identify prunable dimensions (heads, FFN)
    
2.  Insert structured gates
    
3.  Implement:
    
    *   logits getters
        
    *   keep-all exporter
        
    *   pruned exporter
        
4.  Update HF config fields
    
5.  Add a recipe and notebook
    

Assumptions & invariants
------------------------

*   HF model configs correctly describe structure
    
*   Attention heads are independently removable
    
*   No adapter state remains post-export
    
*   Tokenizers are unchanged
    

Summary
-------

The adapters/huggingface module enables **production-ready, hardware-aware slimming** of Transformer models while remaining fully compatible with the Hugging Face ecosystem.

It is the key enabler for:

*   LLaMA head slimming
    
*   ViT structural pruning
    
*   Hub-ready slim model distribution