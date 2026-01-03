"""
LLM Optimizer Benchmark Training Script
========================================
Clean, minimal GPT training for optimizer research.

Key features for optimizer research:
- Gradient norm logging (per-layer and global)
- Gradient spike detection
- Loss landscape smoothness metrics

Usage:
    # Shakespeare (fast iteration, minutes)
    python train.py --dataset shakespeare --n_layer 6 --n_head 6 --n_embd 384
    
    # TinyStories (medium scale, hours)
    python train.py --dataset tinystories --n_layer 12 --n_head 12 --n_embd 768
    
    # OpenWebText GPT-2 124M (full scale, days)
    python train.py --dataset openwebtext --n_layer 12 --n_head 12 --n_embd 768

Adapted from Karpathy's nanoGPT for optimizer research
"""

import math
import time
import pickle
import argparse
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from optimizers import *

# ============================================================================
# MODEL DEFINITION
# ============================================================================

@dataclass
class GPTConfig:
    block_size: int = 1024      # max sequence length
    vocab_size: int = 50304     # GPT-2 vocab size (padded for efficiency)
    n_layer: int = 12           # number of transformer layers
    n_head: int = 12            # number of attention heads
    n_embd: int = 768           # embedding dimension
    dropout: float = 0.0        # dropout rate
    bias: bool = True           # use bias in linear layers


class LayerNorm(nn.Module):
    """LayerNorm with optional bias (PyTorch's doesn't support bias=False)"""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention"""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # Flash attention (PyTorch 2.0+)
        self.flash = hasattr(F, 'scaled_dot_product_attention')

    def forward(self, x):
        B, T, C = x.size()
        
        # Compute Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Attention
        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            # Manual attention (fallback)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            att = att.masked_fill(mask, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Feed-forward network"""
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer block"""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """GPT Language Model"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        # Special scaled init for residual projections (GPT-2 paper)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 /
                    math.sqrt(2 * config.n_layer)
                )

        # Report number of parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model parameters: {n_params/1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Sequence length {t} > block size {self.config.block_size}"

        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        # Forward through transformer
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        
        return logits, loss

    def estimate_mfu(self, batch_size, dt):
        """Estimate model flops utilization (MFU)"""
        N = sum(p.numel() for p in self.parameters())
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_iter = flops_per_token * T * batch_size
        flops_achieved = flops_per_iter / dt
        # A100 GPU bfloat16 peak: 312 TFLOPS
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised
        return mfu


# ============================================================================
# DATA LOADING
# ============================================================================

def get_batch(data, batch_size, block_size, device):
    """Get a random batch of data"""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


def load_data(data_dir):
    """Load train/val data from binary files"""
    train_data = np.memmap(data_dir / 'train.bin', dtype=np.uint16, mode='r')
    val_data = np.memmap(data_dir / 'val.bin', dtype=np.uint16, mode='r')
    
    # Load vocab size from meta
    meta_path = data_dir / 'meta.pkl'
    if meta_path.exists():
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        vocab_size = meta.get('vocab_size', 50304)
    else:
        vocab_size = 50304  # default GPT-2
    
    return train_data, val_data, vocab_size


# ============================================================================
# OPTIMIZER SECTION
# ============================================================================

def configure_optimizer(model, learning_rate, weight_decay, betas, device_type):
    """
    Configure the optimizer.

    Currently uses AdamW. To test your optimizer:
    1. Import your optimizer
    2. Replace the AdamW call below
    3. Adjust any optimizer-specific parameters
    
    Example for a custom optimizer:
        from my_optimizer import RelativisticAdam
        optimizer = RelativisticAdam(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            # your custom params here
        )
    """
    # Separate parameters that should/shouldn't have weight decay
    decay_params = []
    nodecay_params = []
    
    for pn, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Don't decay biases, LayerNorm, or embeddings
        if p.dim() < 2 or 'ln' in pn or 'bias' in pn or 'wpe' in pn or 'wte' in pn:
            nodecay_params.append(p)
        else:
            decay_params.append(p)
    
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    
    num_decay = sum(p.numel() for p in decay_params)
    num_nodecay = sum(p.numel() for p in nodecay_params)
    print(f"Decay params: {num_decay:,}, No-decay params: {num_nodecay:,}")
    
    # Use fused AdamW if available (faster on CUDA)
    fused_available = 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()

    # =========================================
    # OPTIMIZER INSTANTIATION - SWAP HERE
    # =========================================

    # optimizer = torch.optim.AdamW(
    #     optim_groups,
    #     lr=learning_rate,
    #     betas=betas,
    #     eps=1e-8,
    #     **extra_args,
    # )

    # optimizer = RelativisticLion(
    #     optim_groups,
    #     lr=learning_rate,
    #     betas=betas,
    #     mass=args.mass,
    # )

    optimizer = RelativisticSignum(
        optim_groups,
        lr=learning_rate,
        momentum=betas[0],
        mass=args.mass,
    )

    # optimizer = SignSGD(
    #     optim_groups,
    #     lr=learning_rate,
    #     momentum=betas[0],
    #     dampening=0.0,  # 1 - betas[0]
    #     # mass=args.mass,
    # )

    # =========================================
    print(optimizer)
    # print(f"Using {'fused ' if use_fused else ''}AdamW optimizer")
    return optimizer


# ============================================================================
# GRADIENT DIAGNOSTICS - USEFUL FOR OPTIMIZER RESEARCH
# ============================================================================

def compute_gradient_stats(model) -> Dict[str, float]:
    """
    Compute detailed gradient statistics for optimizer research.
    
    Returns:
        - global_grad_norm: overall gradient norm
        - per_layer_norms: gradient norms by layer
        - max_grad: maximum gradient magnitude
        - grad_sparsity: fraction of near-zero gradients
    """
    stats = {}
    
    all_grads = []
    layer_norms = {}
    
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad = p.grad.detach()
            all_grads.append(grad.view(-1))
            
            # Group by layer
            layer_name = name.split('.')[0]
            if layer_name not in layer_norms:
                layer_norms[layer_name] = []
            layer_norms[layer_name].append(grad.norm().item())
    
    if all_grads:
        all_grads = torch.cat(all_grads)
        stats['global_grad_norm'] = all_grads.norm().item()
        stats['max_grad'] = all_grads.abs().max().item()
        stats['grad_sparsity'] = (all_grads.abs() < 1e-8).float().mean().item()
        
        # Per-layer norms (averaged)
        for layer_name, norms in layer_norms.items():
            stats[f'grad_norm/{layer_name}'] = sum(norms) / len(norms)
    
    return stats


def detect_gradient_spike(grad_norm, prev_grad_norm, threshold=10.0):
    """Detect if gradient norm spiked (common in transformers)"""
    if prev_grad_norm is None or prev_grad_norm == 0:
        return False
    return grad_norm / prev_grad_norm > threshold


# ============================================================================
# LEARNING RATE SCHEDULE
# ============================================================================

def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    """Cosine learning rate schedule with warmup"""
    # Linear warmup
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters
    # Constant after decay
    if it > lr_decay_iters:
        return min_lr
    # Cosine decay
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# ============================================================================
# EVALUATION
# ============================================================================

@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, device, eval_iters=200):
    """Estimate loss on train and val splits"""
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = []
        for _ in range(eval_iters):
            X, Y = get_batch(data, batch_size, block_size, device)
            _, loss = model(X, Y)
            losses.append(loss.item())
        out[split] = np.mean(losses)
    model.train()
    return out


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train(args):
    # Setup
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    ctx = torch.amp.autocast(device_type='cuda', dtype=dtype) if 'cuda' in args.device else torch.amp.autocast(device_type='cpu', dtype=torch.bfloat16)
    
    # Load data
    data_dir = Path(args.data_dir)
    train_data, val_data, vocab_size = load_data(data_dir)
    print(f"Train tokens: {len(train_data):,}, Val tokens: {len(val_data):,}")
    print(f"Vocab size: {vocab_size}")
    
    # Model
    model_config = GPTConfig(
        block_size=args.block_size,
        vocab_size=vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        bias=args.bias,
    )
    model = GPT(model_config).to(device)
    
    # Compile model (PyTorch 2.0+)
    if args.compile and hasattr(torch, 'compile'):
        print("Compiling model...")
        model = torch.compile(model)
    
    # Optimizer
    optimizer = configure_optimizer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        device_type='cuda' if 'cuda' in args.device else 'cpu'
    )
    
    # Gradient scaler for mixed precision
    scaler = torch.amp.GradScaler(enabled=('cuda' in args.device))
    
    # Training state
    prev_grad_norm = None
    best_val_loss = float('inf')
    spike_count = 0
    
    # Output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print(f"\nStarting training for {args.max_iters} iterations...")
    print("=" * 60)
    
    t0 = time.time()
    for iter_num in range(args.max_iters):
        # Learning rate schedule
        lr = get_lr(iter_num, args.warmup_iters, args.lr_decay_iters, 
                    args.learning_rate, args.min_lr)
        # lr = args.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Evaluation
        if iter_num % args.eval_interval == 0 or iter_num == args.max_iters:
            if iter_num == 0:
                continue
            losses = estimate_loss(model, train_data, val_data, 
                                   args.batch_size, args.block_size, device)

            # print optim and val info
            opt_name = optimizer.__class__.__name__
            pg0 = optimizer.param_groups[0]

            opt_label = (
                f"{opt_name}(mass={pg0['mass']})"
                if 'mass' in pg0 else opt_name
            )
            print(
                f"iter {iter_num:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | "
                f"opt {opt_label} | current lr = {lr:.2e} (/ {args.learning_rate:.2e})"
            )

            # Save best checkpoint
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': model_config,
                }
                torch.save(checkpoint, out_dir / 'ckpt_best.pt')
        
        # Get batch
        X, Y = get_batch(train_data, args.batch_size, args.block_size, device)
        
        # Forward pass
        with ctx:
            logits, loss = model(X, Y)
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient clipping
        if args.grad_clip is not None and args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip)
        
        # Gradient diagnostics (for optimizer research)
        if iter_num % args.log_interval == 0:
            grad_stats = compute_gradient_stats(model)
            grad_norm = grad_stats.get('global_grad_norm', 0)
            
            # Detect spikes
            if detect_gradient_spike(grad_norm, prev_grad_norm):
                spike_count += 1
                print(f"  ⚠️  Gradient spike detected! {prev_grad_norm:.2f} -> {grad_norm:.2f}")
            
            prev_grad_norm = grad_norm
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        # Logging
        if iter_num % args.log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            tokens_per_sec = args.batch_size * args.block_size / dt
            print(
                f"iter {iter_num:5d} | loss {loss.item():.4f} | lr {lr:.2e} | "
                f"grad_norm {grad_stats.get('global_grad_norm', 0):.2f} | "
                f"{tokens_per_sec:.0f} tok/s")
            t0 = t1
    
    # Final save
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iter_num': args.max_iters,
        'best_val_loss': best_val_loss,
        'config': model_config,
    }
    torch.save(checkpoint, out_dir / 'ckpt_final.pt')
    
    print("=" * 60)
    print(f"Training complete!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Gradient spikes detected: {spike_count}")
    print(f"Checkpoints saved to: {out_dir}")


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Optimizer Benchmark")

    # Data
    parser.add_argument("--dataset", type=str, default="shakespeare",
                        choices=["shakespeare", "openwebtext", "tinystories"])
    parser.add_argument("--data_dir", type=str, default=None)

    # Model
    parser.add_argument("--n_layer", type=int, default=8)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_embd", type=int, default=512)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--bias", type=bool, default=True)

    # Training
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_iters", type=int, default=100_000)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_iters", type=int, default=200)

    # Optimizer
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.95)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--mass", type=float, default=10.0)

    # LR schedule
    parser.add_argument("--warmup_iters", type=int, default=10_000)
    parser.add_argument("--lr_decay_iters", type=int, default=90_000)
    
    # System
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--compile", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out_dir", type=str, default="out")
    
    args = parser.parse_args()
    
    # Set data directory based on dataset
    if args.data_dir is None:
        args.data_dir = f"data/{args.dataset}"
    
    train(args)
