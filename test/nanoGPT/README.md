# LLM Optimizer Benchmark

A minimal, clean codebase for benchmarking optimizers on language model training.
Adapted from nanoGPT with added gradient diagnostics for optimizer research.

## Quick Start

### 1. Install dependencies

```bash
pip install torch numpy requests
# For OpenWebText or TinyStories:
pip install datasets tiktoken tqdm
```

### 2. Prepare data

**Option A: Shakespeare (fastest, good for debugging)**
```bash
python prepare_data.py --dataset shakespeare
```
- ~1M characters, character-level
- Trains in minutes on GPU, ~1 hour on CPU
- Good for: quick iteration, debugging optimizer code

**Option B: TinyStories (medium scale)**
```bash
python prepare_data.py --dataset tinystories
```
- ~2.5M stories, BPE tokenized
- Trains in hours
- Good for: testing optimizer on realistic dynamics without huge compute

**Option C: OpenWebText (full scale)**
```bash
python prepare_data.py --dataset openwebtext
```
- ~9B tokens, BPE tokenized
- Trains in days (8xA100) to weeks (single GPU)
- Good for: final validation, reproducing GPT-2 results

### 3. Train

```bash
# Shakespeare (fast iteration)
python train.py --dataset shakespeare --n_layer 6 --n_head 6 --n_embd 384

# TinyStories (medium scale)
python train.py --dataset tinystories --n_layer 12 --n_head 12 --n_embd 768 --batch_size 32

# OpenWebText GPT-2 124M
python train.py --dataset openwebtext --n_layer 12 --n_head 12 --n_embd 768 --batch_size 12
```

## Swapping in Your Optimizer

Look for the **OPTIMIZER SECTION** in `train.py` (around line 200). Replace the AdamW call:

```python
# BEFORE (AdamW baseline)
optimizer = torch.optim.AdamW(
    optim_groups,
    lr=learning_rate,
    betas=betas,
    eps=1e-8,
    **extra_args
)

# AFTER (your optimizer)
from your_optimizer import RelativisticAdam  # or whatever

optimizer = RelativisticAdam(
    optim_groups,
    lr=learning_rate,
    betas=betas,
    # your custom params
    metric_tensor=...,
)
```

## Gradient Diagnostics

The training script logs several metrics useful for optimizer research:

- **`global_grad_norm`**: Overall gradient magnitude
- **`max_grad`**: Maximum gradient value (spikes here!)
- **`grad_sparsity`**: Fraction of near-zero gradients
- **Gradient spike detection**: Warns when grad_norm jumps >10x

These appear in the training logs. For more detailed analysis, the `compute_gradient_stats()` function can be extended.

## Common Gradient Pathologies in Transformers

1. **Attention logit explosion**: QK^T grows unboundedly → entropy collapse → huge gradients in attention
2. **LayerNorm instabilities**: Especially in early layers, gradients through LayerNorm can spike
3. **Embedding layer oscillations**: wte/wpe can have unstable gradients early in training
4. **Loss spikes**: Often precede gradient spikes by 1-2 steps

## Suggested Experiments

### Baseline comparison
```bash
# AdamW baseline
python train.py --dataset shakespeare --out_dir out_adamw

# Your optimizer
# (modify train.py to use your optimizer)
python train.py --dataset shakespeare --out_dir out_yours
```

### Sensitivity analysis
```bash
# Vary learning rate
for lr in 1e-4 5e-4 1e-3 5e-3 1e-2; do
    python train.py --learning_rate $lr --out_dir out_lr_$lr
done
```

### Scale test
```bash
# Small model
python train.py --n_layer 4 --n_head 4 --n_embd 128 --out_dir out_small

# Medium model  
python train.py --n_layer 8 --n_head 8 --n_embd 512 --out_dir out_medium

# Large model (needs more GPU memory)
python train.py --n_layer 12 --n_head 12 --n_embd 768 --out_dir out_large
```

## Model Configurations

| Config | Layers | Heads | Embed | Params | Notes |
|--------|--------|-------|-------|--------|-------|
| Tiny | 4 | 4 | 128 | ~1M | Debug |
| Small | 6 | 6 | 384 | ~10M | Shakespeare default |
| Medium | 8 | 8 | 512 | ~40M | Good for TinyStories |
| GPT-2 Small | 12 | 12 | 768 | ~124M | OpenWebText default |
| GPT-2 Medium | 24 | 16 | 1024 | ~350M | Needs multi-GPU |

## Key References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original transformer
- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - Training details
- [μP (Maximal Update Parameterization)](https://arxiv.org/abs/2203.03466) - Hyperparameter transfer across scales
- [On the Stability of Transformers](https://arxiv.org/abs/2002.04745) - Gradient flow analysis
