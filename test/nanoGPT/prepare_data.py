"""
Data Preparation for LLM Optimizer Benchmarks
==============================================
Two options:
  1. Shakespeare (char-level) - Fast iteration, ~1M tokens, trains in minutes
  2. OpenWebText (BPE) - More realistic, ~9B tokens, trains in hours/days

Usage:
    python prepare_data.py --dataset shakespeare
    python prepare_data.py --dataset openwebtext
"""

import os
import argparse
import numpy as np
import requests
from pathlib import Path


def prepare_shakespeare(data_dir: Path):
    """
    Character-level Shakespeare dataset.
    Fast iteration for optimizer debugging (~1M chars, trains in minutes on GPU).
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download Shakespeare
    input_file = data_dir / "input.txt"
    if not input_file.exists():
        print("Downloading Shakespeare...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        response = requests.get(url)
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write(response.text)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.read()
    
    print(f"Dataset size: {len(data):,} characters")
    
    # Build character vocabulary
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print(f"Vocabulary size: {vocab_size} unique characters")
    
    # Create mappings
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    # Encode
    def encode(s):
        return [stoi[c] for c in s]
    
    # Train/val split (90/10)
    n = len(data)
    train_data = data[:int(n * 0.9)]
    val_data = data[int(n * 0.9):]
    
    train_ids = np.array(encode(train_data), dtype=np.uint16)
    val_ids = np.array(encode(val_data), dtype=np.uint16)
    
    print(f"Train tokens: {len(train_ids):,}")
    print(f"Val tokens: {len(val_ids):,}")
    
    # Save
    train_ids.tofile(data_dir / "train.bin")
    val_ids.tofile(data_dir / "val.bin")
    
    # Save meta info
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    import pickle
    with open(data_dir / "meta.pkl", 'wb') as f:
        pickle.dump(meta, f)
    
    print(f"Data saved to {data_dir}")
    return vocab_size


def prepare_openwebtext(data_dir: Path):
    """
    BPE-tokenized OpenWebText dataset.
    More realistic LLM training dynamics (~9B tokens).
    Requires: pip install datasets tiktoken
    """
    try:
        import tiktoken
        from datasets import load_dataset
        from tqdm import tqdm
    except ImportError:
        print("Please install: pip install datasets tiktoken tqdm")
        return
    
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # GPT-2 BPE tokenizer
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab  # 50257
    
    print("Loading OpenWebText dataset (this may take a while)...")
    dataset = load_dataset("openwebtext", num_proc=8)
    
    # Split into train/val (0.05% for validation)
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=42)
    split_dataset['val'] = split_dataset.pop('test')
    
    def process(example):
        ids = enc.encode_ordinary(example['text'])
        ids.append(enc.eot_token)  # end of text token
        return {'ids': ids, 'len': len(ids)}
    
    # Tokenize
    print("Tokenizing...")
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        num_proc=8,
        desc="Tokenizing"
    )
    
    # Write to binary files
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'])
        filename = data_dir / f'{split}.bin'
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        
        print(f"Writing {split}.bin...")
        idx = 0
        for example in tqdm(dset, desc=f"Writing {split}"):
            arr[idx:idx + example['len']] = example['ids']
            idx += example['len']
        arr.flush()
        print(f"{split} has {arr_len:,} tokens")
    
    # Save meta
    meta = {'vocab_size': vocab_size}
    import pickle
    with open(data_dir / "meta.pkl", 'wb') as f:
        pickle.dump(meta, f)
    
    print(f"Data saved to {data_dir}")
    return vocab_size


def prepare_tinystories(data_dir: Path):
    """
    TinyStories dataset - good middle ground.
    ~2.5M examples, small models show emergent capabilities.
    """
    try:
        import tiktoken
        from datasets import load_dataset
        from tqdm import tqdm
    except ImportError:
        print("Please install: pip install datasets tiktoken tqdm")
        return
    
    data_dir.mkdir(parents=True, exist_ok=True)
    
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    
    print("Loading TinyStories dataset...")
    dataset = load_dataset("roneneldan/TinyStories")
    
    def process(example):
        ids = enc.encode_ordinary(example['text'])
        ids.append(enc.eot_token)
        return {'ids': ids, 'len': len(ids)}
    
    print("Tokenizing...")
    tokenized = dataset.map(
        process,
        remove_columns=['text'],
        num_proc=8,
        desc="Tokenizing"
    )
    
    for split in ['train', 'validation']:
        dset = tokenized[split]
        arr_len = np.sum(dset['len'])
        out_split = 'val' if split == 'validation' else split
        filename = data_dir / f'{out_split}.bin'
        arr = np.memmap(filename, dtype=np.uint16, mode='w+', shape=(arr_len,))
        
        print(f"Writing {out_split}.bin...")
        idx = 0
        for example in tqdm(dset, desc=f"Writing {out_split}"):
            arr[idx:idx + example['len']] = example['ids']
            idx += example['len']
        arr.flush()
        print(f"{out_split} has {arr_len:,} tokens")
    
    meta = {'vocab_size': vocab_size}
    import pickle
    with open(data_dir / "meta.pkl", 'wb') as f:
        pickle.dump(meta, f)
    
    print(f"Data saved to {data_dir}")
    return vocab_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="shakespeare",
                        choices=["shakespeare", "openwebtext", "tinystories"])
    parser.add_argument("--data_dir", type=str, default=None)
    args = parser.parse_args()
    
    if args.data_dir is None:
        args.data_dir = Path(f"data/{args.dataset}")
    else:
        args.data_dir = Path(args.data_dir)
    
    if args.dataset == "shakespeare":
        prepare_shakespeare(args.data_dir)
    elif args.dataset == "openwebtext":
        prepare_openwebtext(args.data_dir)
    elif args.dataset == "tinystories":
        prepare_tinystories(args.data_dir)
