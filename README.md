# HawAda: Hardware-aware Adaptation

__HawAda__ is an open-source universal framework for *hardware-aware optimization* of machine learning models, enabling models to be *automatically adapted* to a specific *target GPU*.

Inspired by hardware-aware pruning and efficiency methods, __HawAda__ provides a simple, user-friendly, and model-agnostic interface for optimizing neural networks with respect to real hardware latency, while preserving model quality.

High-level overview
-------------------

HawAda follows a teacher–student, gate-based optimization pipeline:

1.  **Base model**: Start from an existing pretrained model (the _teacher_).
    
2.  **HawAda adapter**: Trainable _gates_ are attached to selected layers or components of the base model.Each gate modulates its corresponding output via a learnable sigmoid, enabling fine-grained structural control.
    
3.  **Training objectives**: The gates are optimized using two complementary objectives:
    
    *   **Hardware-aware latency objective**A differentiable proxy of model latency, periodically calibrated with _real measured latency_ on the target GPU and incorporated via a Lagrange multiplier.
        
    *   **Teacher agreement objective**KL divergence (and optionally MSE) between the student and the teacher model outputs, ensuring minimal accuracy degradation.
        
4.  **Pruning / export**: After training (or by loading a pretrained HawAda checkpoint, e.g. from Hugging Face), the learned gate values are used to **prune the base model**.Only components that are critical for maintaining agreement with the teacher are retained.
    
5.  **GPU-aware slimming**: During export, HawAda searches for the **fastest valid architecture** by iterating over GPU-friendly structural constraints (e.g. head multiples), ensuring optimal performance on the target hardware.
    
6.  **Final fine-tuning**: The resulting _slim model_ can be fine-tuned on **any GPU** (not necessarily the one used for hardware-aware optimization) to recover or improve final accuracy.Alternatively, pretrained slim models can be downloaded from the HawAda model zoo.
    
Current support
---------------

HawAda currently includes reference implementations for:

*   **ResNet-18** optimized for **RTX 4090**
    
*   **ViT-Base** optimized for **RTX 4090**
    
*   **LLaMA-3.2-1B** optimized for **RTX 4090** and **H100**

Notebooks
---------

The /notebooks directory contains end-to-end, executable examples demonstrating the full HawAda workflow — from loading pretrained models to exporting and fine-tuning hardware-optimized slim models.

Each notebook covers:

* Loading pretrained models from Hugging Face
* Measuring real GPU latency
* Training HawAda gates with hardware-aware objectives
* Exporting and pruning slim models
* Optional fine-tuning of the exported models

## Available notebooks

* ResNet-18: `/notebooks/resnet.ipynb`

* ViT-Base/16 `/notebooks/vit.ipynb`

* LLaMA-3.2-1B `/notebooks/llama.ipynb`

Repository overview
-------------------

The HawAda repository is organized around a modular, model-agnostic hardware-aware optimization pipeline, separating model definitions, adapters, training logic, latency modeling, and export utilities.

```
hawada/
├── core/                   # Core training and optimization logic
│   ├── train.py            # Lagrangian trainer, optimization loop
│   ├── penalties.py        # Regularization and structured penalties
│   ├── latency.py          # Latency proxy models and calibration
│   └── losses.py           # KL, MSE, and auxiliary losses
│
├── adapters/               # HawAda adapters and gating modules
│   ├── base.py             # Abstract adapter interfaces
│   ├── gates.py            # Trainable sigmoid gates
│   └── attach.py           # Utilities to inject adapters into models
│
├── models/                 # Model-specific integration
│   ├── resnet/             # ResNet adapters and export logic
│   ├── vit/                # Vision Transformer support
│   └── llama/              # LLaMA adapters, head slimming, export
│
├── export/                 # Pruning and slim model export
│   ├── prune.py            # Gate-based pruning logic
│   ├── search.py           # GPU-aware architecture search
│   └── hf_export.py        # Hugging Face–compatible export
│
├── proxy/                  # Hardware latency proxy models
│   ├── base.py             # Proxy interface
│   ├── trt.py              # TensorRT-based proxies
│   └── torch.py            # PyTorch runtime proxies
│
├── benchmarks/             # Hardware measurement utilities
│   ├── measure.py          # Real GPU latency measurement
│   └── configs/            # GPU-specific benchmark configs
│
├── configs/                # Training and optimization configs
│   ├── resnet18_rtx4090.yaml
│   ├── vit_base_rtx4090.yaml
│   └── llama32_1b_h100.yaml
│
├── scripts/                # End-to-end runnable scripts
│   ├── train.py            # Launch HawAda optimization
│   ├── export.py           # Export slim model
│   └── benchmark.py        # Measure final latency
│
├── examples/               # Minimal working examples
│   ├── resnet18/
│   ├── vit_base/
│   └── llama/
│
├── utils/                  # Shared utilities
│   ├── logging.py
│   ├── checkpointing.py
│   └── distributed.py
│
└── README.md
```

Design principles
-----------------

*   **Model-agnostic**: Adapters and gates can be attached to arbitrary architectures.
    
*   **Hardware-aware**: Real GPU latency is periodically measured and integrated into training.
    
*   **Separation of concerns**: Training, latency modeling, pruning, and export are cleanly decoupled.
    
*   **Reproducible**: GPU- and model-specific configs fully define an experiment.
    
*   **HF-compatible**: Exported slim models can be pushed directly to Hugging Face.
    

Typical workflow
----------------

1.  Choose a base model and target GPU
    
2.  Attach HawAda adapters
    
3.  Train gates using the Lagrangian optimizer
    
4.  Export and prune the model using learned gates
    
5.  Fine-tune or deploy the slim model


