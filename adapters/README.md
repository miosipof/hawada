# `adapters/` — HawAda Gating & Model Injection

The `adapters/` module defines HawAda adapters: lightweight, trainable components that are injected into a base model to enable hardware-aware structural optimization.

Adapters are responsible for:

* Attaching trainable gates to existing model layers
* Modulating layer outputs during training
* Exposing gate values for pruning and export
* Remaining fully architecture-agnostic

The base model’s weights remain frozen; only adapter parameters are trained.

## Design goals

* Non-invasive: no changes to the original model code
* Composable: adapters can be attached or removed cleanly
* Differentiable: gates support gradient-based optimization
* Export-friendly: gate states map directly to pruning decisions

## Directory structure
```
adapters/
├── base.py        # Abstract adapter interfaces
├── gates.py       # Trainable sigmoid gates
├── attach.py      # Utilities to inject adapters into models
└── __init__.py
```

## Concept: HawAda gates

A gate is a scalar or vector-valued parameter `g ∈ [0, 1]` applied to an intermediate activation:

```python
y = g * f(x)
g = sigmoid(theta)
```

Where:

* theta is a trainable parameter
* f(x) is the original layer output
* g determines whether a component is kept, attenuated, or removed

Gates can correspond to:

* Channels
* Attention heads
* MLP blocks
* Entire layers



# gates.py — Trainable Gates

Defines the fundamental gating primitives used throughout HawAda.

### Core components

*   **SigmoidGate**
    
    *   Learnable parameter(s)
        
    *   Outputs values in \[0, 1\]
        
    *   Supports broadcasting to match tensor shapes
        
*   **Structured gates**
    
    *   Head-level gates
        
    *   Channel-wise gates
        
    *   Block-level gates
        

### Key properties

*   Initialized to near-1 to preserve teacher behavior
    
*   Compatible with mixed precision
    
*   Expose raw and sigmoid-activated values


```
gate = SigmoidGate(shape=(num_heads,))
y = gate(x)
```

base.py — Adapter Interfaces
----------------------------

Defines abstract base classes for all HawAda adapters.

### Responsibilities

*   Wrap a target module
    
*   Apply one or more gates to its output
    
*   Register gates for training and export
    
*   Provide introspection utilities
    

### Typical adapter lifecycle

1.  Receive a reference to a base model module
    
2.  Insert gates in the forward pass
    
3.  Expose gate tensors via a standardized interface
    

Adapters **do not**:

*   Own base model weights
    
*   Perform pruning
    
*   Handle training logic
    

attach.py — Adapter Injection
-----------------------------

Contains utilities to **attach adapters to arbitrary models**.

### Responsibilities

*   Traverse model graphs
    
*   Match layers by type or name
    
*   Replace or wrap modules with adapters
    
*   Maintain parameter naming consistency
    

### Design highlights

*   Works with nn.Module hierarchies
    
*   Preserves state dict compatibility
    
*   Supports selective attachment (e.g. only attention layers)
    

### Example

```
attach_adapters(
    model,
    adapter_cls=AttentionHeadAdapter,
    filter_fn=is_attention_layer
)
```

Training interaction
--------------------

During training:

*   Gates are active in the forward pass
    
*   Gradients flow only through gate parameters
    
*   Base model weights are frozen
    

The trainer queries adapters to:

*   Collect all gates
    
*   Apply penalties
    
*   Compute latency proxies
    

Export interaction
------------------

During export:

*   Gate values are read and thresholded
    
*   Adapters are removed
    
*   Pruned structure is materialized in a slim model
    

Adapters are **not** part of the final (slim) exported model.

Extension guide
---------------

### Adding a new adapter type

1.  Subclass the base adapter in base.py
    
2.  Define where and how gates are applied
    
3.  Register gates for export
    
4.  Provide a matching prune/export implementation
    

### Adding a new gate type

1.  Implement it in gates.py
    
2.  Ensure it exposes:
    
    *   Raw parameters
        
    *   Sigmoid output
        
3.  Make it compatible with penalties and latency proxy
    

Invariants & assumptions
------------------------

*   Adapters must be removable without side effects
    
*   Gate shapes must align with prunable structure
    
*   No adapter should alter numerical semantics beyond scaling
    
*   Adapter logic must remain differentiable