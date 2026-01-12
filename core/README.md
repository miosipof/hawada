`core/` — Training & Optimization Engine
======================================

The core/ module contains the **heart of HawAda**: the Lagrangian training loop, loss definitions, latency integration, and regularization logic used to train hardware-aware gates.

This module is **model-agnostic** and does not assume any specific architecture (CNN, ViT, LLM).All model-specific behavior is injected via adapters, latency proxies, and configuration objects.

Responsibilities
----------------

The core/ module is responsible for:

*   Training HawAda gates using a **Lagrangian formulation**
    
*   Combining accuracy and hardware latency objectives
    
*   Integrating **real GPU latency measurements** into training
    
*   Applying structured penalties and regularization
    
*   Providing a clean abstraction for student–teacher optimization


# Directory structure

```
core/
├── train.py        # Main Lagrangian training loop
├── losses.py       # KL, MSE, and auxiliary losses
├── penalties.py    # Gate regularization and penalties
├── latency.py      # Latency proxy and calibration
└── __init__.py
```

# Training formulation

HawAda optimizes gate parameters using the following objective:

```
L = L_KD(student, teacher)
  + λ · max(0, Latency_proxy − Latency_target)
  + Regularization(gates)
```

# `train.py` — Lagrangian Trainer

## Key class

```
class LagrangeTrainer:
    ...
```

This class orchestrates **gate optimization** under hardware constraints.

### Core responsibilities

*   Runs forward passes for **student and teacher**
*   Computes KD loss (KL + optional MSE)
*   Queries latency proxy and real hardware measurements
*   Updates the Lagrange multiplier (lambda\_)
*   Applies structured gate penalties
*   Supports early stopping and logging

### Key methods

*   `train\_epoch(loader, ...)` Executes a single optimization epoch and updates lambda\_.
    
*   `step\_lambda(latency\_gap)` Updates the Lagrange multiplier based on constraint violation.
    
*   `measure\_latency(model)` Periodically measures real GPU latency to calibrate the proxy.
    

### Design notes

*   The trainer **never modifies model structure directly**
    
*   Only gate parameters are trainable
    
*   Teacher can be:
    
    *   The original base model
        
    *   A frozen copy
        
    *   A different higher-capacity model


## `losses.py` — Accuracy Preservation

Defines losses used to preserve teacher behavior.

### Implemented losses:

* KL divergence
    * Temperature-scaled
    * Token-wise or output-wise

* MSE loss (optional)

    * Stabilizes early training
    * Especially useful for large language models

### Usage

Losses are combined and weighted via config:

```yaml
kd:
  temperature: 2.0
  mse_weight: 0.1

```

# penalties.py — Gate Regularization

Contains regularization terms applied to gate values.

### Supported penalties

*   **L1 sparsity** Encourages aggressive pruning
*   **Group / structured penalties** Enforces consistent pruning across:
    
    *   Attention heads
    *   Channels
    *   Blocks
        
*   **Floor / ceiling constraints** Prevents degenerate all-off or all-on solutions
    

### Design principle

Penalties operate **only on gates**, never on base model weights.


# `latency.py` — Hardware Awareness

Defines the interface between training and hardware latency.

## Components

* Latency proxy
    * Differentiable
    * Predicts latency from gate activations / structure

* Real latency measurement
    * Periodically executed on the target GPU
    * Used to recalibrate the proxy

* Why both?
    * Proxies allow gradient-based optimization
    * Real measurements ensure hardware fidelity

# Configuration-driven behavior

All behavior in `core/` is controlled via configuration:

```yaml
latency_target_ms: 12.0
lambda:
  init: 1.0
  lr: 0.01
penalties:
  l1: 1e-4
  structured: true
```

This allows:

*   Reproducibility
    
*   Easy GPU retargeting
    
*   Model-agnostic training logic
    

# Extension points

To extend `core/`, you can:

### Add a new loss

*   Implement it in `losses.py`
*   Register it in the trainer
    

### Add a new penalty

*   Implement in `penalties.py`
*   Reference it via config
    

### Add a new latency proxy

*   Implement the proxy interface in `latency.py`
*   Plug it into the trainer
    

No changes to model code are required.


# Invariants & assumptions

* Base model weights are frozen

* Only gate parameters are trainable

* Latency constraints are soft via Lagrangian optimization

* Export / pruning happens outside core/