`core/` — Training & Optimization Engine
======================================

Overview
--------

core/ contains HawAda’s **family-agnostic optimization engine**: gate training (Lagrangian KD + latency constraint), gate regularization/constraints, latency proxy integration, real latency measurement utilities, and generic export helpers (slicing + rounding).

This folder is intentionally **model-agnostic**: it does not hardcode ResNet/ViT/LLaMA details. Model specifics live in adapters/.

What’s inside
-------------

*   train.py — Generic Lagrangian trainer for gated students
    
    *   Two-phase step:
        
        1.  **Weights update** using KD (and optional MSE)
            
        2.  **Gates update** using KD + penalties + λ \* latency\_gap
            
    *   Optional real latency probing via profiler.measure\_latency\_ms
        
    *   Gate projection after updates via project\_gates\_into\_constraints
        
    *   Adapter contract (passed in as callables): get logits, export keep-all, export pruned
        
*   distill.py — Distillation utilities
    
    *   KDConfig, kd\_loss, and optional mse\_reg used by the trainer
        
*   gates.py — Gate modules + penalties + constraints
    
    *   Gate types used by adapters (e.g. head gates, grouped gates)
        
    *   Penalty composition (combined\_penalty)
        
    *   Constraint projection (project\_gates\_into\_constraints)
        
    *   Parameter grouping helper used to build separate optimizers
        
*   proxy\_cost.py — Differentiable latency proxy interface
    
    *   Provides LatencyProxy used during training to produce a differentiable latency estimate
        
*   profiler.py — Real latency measurement
    
    *   measure\_latency\_ms(model, batch\_shape, ...) utility used for periodic hardware probes and export search
        
*   export.py — Family-agnostic pruning/export primitives
    
    *   ExportPolicy, Rounding
        
    *   Keep-index selection from gate probabilities (keep\_group\_indices\_from\_gate, keep\_element\_indices\_from\_gate)
        
    *   Generic weight slicers:
        
        *   slice\_linear
            
        *   slice\_conv2d
            
        *   slice\_embedding
            
    *   Adapters use these primitives to rebuild _family-specific_ pruned modules
        
*   search\_export.py — Hardware-aware export parameter search
    
    *   grid\_search\_latency(...) runs a small grid over export rounding knobs (e.g., head multiples, FFN snaps)
        
    *   Picks the best configuration by **measured** latency using profiler
        
*   finetune.py — Fine-tuning utilities for exported slim models
    
*   utils.py — Small shared utilities (e.g., safe deepcopy helpers)
    

How core/ connects to the rest
------------------------------

*   adapters/ supplies:
    
    *   where gates are attached
        
    *   how to get logits for KD
        
    *   how to export keep-all / pruned variants
        
*   recipes/ supplies configs (GPU-specific)
    
*   tools/ includes helper CLIs (export to HF, eval, etc.)