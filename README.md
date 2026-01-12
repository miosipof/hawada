# HawAda: Hardware-aware Adaptation

__HawAda__ is an open-source universal framework for *hardware-aware optimization* of machine learning models, enabling models to be *automatically adapted* to a specific *target GPU*.

Inspired by hardware-aware pruning and efficiency methods [[1](https://openaccess.thecvf.com/content_ECCV_2018/papers/Tien-Ju_Yang_NetAdapt_Platform-Aware_Neural_ECCV_2018_paper.pdf), [2](https://arxiv.org/abs/1908.09791), [3](https://arxiv.org/abs/1807.11626), [4](https://arxiv.org/pdf/2110.10811) ], __HawAda__ provides a simple, user-friendly, and model-agnostic interface for optimizing neural networks with respect to real hardware latency, while preserving model quality.

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
├── .gitignore
├── README.md
├── __init__.py
├── requirements.txt
│
├── adapters/
│   ├── __init__.py
│   ├── huggingface/
│   │   ├── __init__.py
│   │   ├── llama.py
│   │   └── vit.py
│   └── torchvision/
│       └── resnet.py
│
├── core/
│   ├── __init__.py
│   ├── distill.py
│   ├── export.py
│   ├── finetune.py
│   ├── gates.py
│   ├── profiler.py
│   ├── proxy_cost.py
│   ├── search_export.py
│   ├── train.py
│   └── utils.py
│
├── data/
│   ├── __init__.py
│   ├── llms.py
│   └── vision.py
│
├── examples/
│   ├── __init__.py
│   ├── run_llama_optimize.py
│   ├── run_resnet_optimize.py
│   └── run_vit_optimize.py
│
├── notebooks/
│   ├── llama.ipynb
│   ├── resnet.ipynb
│   ├── vit.ipynb
│   └── utils/
│       ├── ViT_pretraining.ipynb
│       └── imagenet224.ipynb
│
├── recipes/
│   ├── H100/
│   │   ├── llama_3_2_1b.yaml
│   │   ├── resnet18_imagenet224.yaml
│   │   └── vit_base_patch16_224.yaml
│   └── RTX4090/
│       ├── llama_3_2_1b.yaml
│       ├── resnet18_imagenet224.yaml
│       └── vit_base_patch16_224.yaml
│
└── tools/
    ├── eval_agreement.py
    ├── eval_llama_slim.py
    ├── export_to_hf.py
    └── print_shapes.py
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



References
----------------

[1] [NetAdapt: Platform-Aware Neural Network Adaptation for Mobile Applications](https://openaccess.thecvf.com/content_ECCV_2018/papers/Tien-Ju_Yang_NetAdapt_Platform-Aware_Neural_ECCV_2018_paper.pdf)

[2] [Once-for-All: Train One Network and Specialize it for Efficient Deployment](https://arxiv.org/abs/1908.09791)

[3] [MnasNet: Platform-Aware Neural Architecture Search for Mobile](https://arxiv.org/abs/1807.11626)

[4] [HALP: Hardware-Aware Latency Pruning](https://arxiv.org/pdf/2110.10811)