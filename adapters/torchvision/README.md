Overview
--------

The `adapters/torchvision` module provides **HawAda adapters for Torchvision models**, currently focused on **ResNet-style CNNs**.

These adapters bridge:

*   Torchvision model implementations
    
*   HawAda’s **family-agnostic training engine** (core/)
    
*   Gate-based pruning and export utilities
    

The design ensures that:

*   Base model weights remain frozen
    
*   Only HawAda gates are trained
    
*   Exported models are clean torchvision modules with no adapter residue
    

Supported models
----------------

*   **ResNet-18** (reference implementation)
    
*   The design generalizes to other ResNet variants (ResNet-34/50) with minimal changes

Adapter responsibilities
------------------------

resnet.py implements the following responsibilities:

### 1\. Adapter attachment

*   Traverses the ResNet module hierarchy
    
*   Wraps target convolutions / blocks with gated adapters
    
*   Preserves original module names for state-dict compatibility
    

### 2\. Logits interface

Provides standardized functions used by core/train.py:

*   get\_student\_logits(model, batch)
    
*   get\_teacher\_logits(model, batch)
    

This allows core/ to remain model-agnostic.

### 3\. Export logic

Implements two export paths:

#### Keep-all export

*   Removes adapters
    
*   Restores a clean ResNet with original structure
    
*   Useful for debugging and comparison
    

#### Pruned export

*   Reads trained gate values
    
*   Selects surviving channels
    
*   Uses core/export.py slicing utilities:
    
    *   slice\_conv2d
        
    *   slice\_linear
        
*   Rebuilds convolutional layers with reduced width
    

Interaction with core/
----------------------

During training:

*   Gates participate in forward/backward passes
    
*   Penalties and latency proxies operate on gate activations
    

During export:

*   Gate values are thresholded
    
*   Adapter modules are removed
    
*   Structural pruning is materialized
    

Extension guide
---------------

### Supporting another Torchvision model

To add a new Torchvision model:

1.  Identify prunable structure (channels, blocks)
    
2.  Attach gates in forward pass
    
3.  Implement:
    
    *   logits getters
        
    *   keep-all export
        
    *   pruned export using core/export.py
        
4.  Add a recipe in recipes/
    

Assumptions & invariants
------------------------

*   Torchvision model APIs remain stable
    
*   Channel pruning preserves residual compatibility
    
*   No dynamic control flow in forward pass
    
*   Adapters must be fully removable
    

Summary
-------

adapters/torchvision provides a **clean and minimal integration layer** between Torchvision CNNs and HawAda’s hardware-aware optimization pipeline, enabling:

*   Channel-level slimming
    
*   Real-GPU latency optimization
    
*   Safe export to standard Torchvision models