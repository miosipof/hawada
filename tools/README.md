`tools/` contains small utilities for evaluation, inspection, and packaging exported models.

# Utilities

* `eval_agreement.py` — evaluate student/teacher agreement (KD-style metrics)
* `eval_llama_slim.py` — evaluate exported slim LLaMA models
* `export_to_hf.py` — export a trained/pruned model into a Hugging Face–loadable format
* `print_shapes.py` — debug helper to print module/tensor shapes (useful during slicing/export)

These tools are intended for quick iteration and sanity checks outside notebooks.