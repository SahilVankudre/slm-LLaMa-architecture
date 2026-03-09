"""
scripts/train.py

Training Entry Point
Trains the SLM model on TinyStories dataset.

Usage:
    python scripts/train.py [--config CONFIG_PATH]

This script:
  1. Loads configuration
  2. Initializes model, tokenizer, datasets
  3. Sets up optimizer and scheduler
  4. Creates trainer
  5. Runs training loop
  6. Saves final model

Configuration can be customized via command-line args or config file.
"""

import os
import sys
import argparse
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.slm import SLM
from src.model.config import ModelConfig, estimate_params
from src.tokenizer.tokenizer import Tokenizer
from src.training.dataset import TinyStoriesDataset, TinyStoriesDataLoader, SimpleTextDataset
from src.training.trainer import Trainer
from src.training.scheduler import SimpleCosineScheduler


def parse_args():
    parser = argparse.ArgumentParser(description="Train SLM on TinyStories")
    
    # Model config
    parser.add_argument("--d_model", type=int, default=384, help="Hidden dimension")
    parser.add_argument("--n_layers", type=int, default=8, help="Number of transformer layers")
    parser.add_argument("--n_heads", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--n_kv_heads", type=int, default=2, help="Number of KV heads (GQA)")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Maximum sequence length")

    # Training config
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation")
    parser.add_argument("--max_steps", type=int, default=20000, help="Maximum training steps")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--min_lr", type=float, default=3e-5, help="Minimum learning rate")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping")
    
    # Data config
    parser.add_argument("--max_samples", type=int, default=None, help="Limit dataset size (for testing)")
    parser.add_argument("--streaming", action="store_true", help="Use streaming dataset")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    
    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--save_interval", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--eval_interval", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--log_interval", type=int, default=100, help="Log every N steps")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint")
    
    # System
    parser.add_argument("--device", type=str, default="auto", help="Device: 'cuda', 'cpu', or 'auto'")
    parser.add_argument("--use_amp", action="store_true", help="Use mixed precision training")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"], 
                        help="Mixed precision dtype")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Quick test mode
    parser.add_argument("--test", action="store_true", help="Quick test mode (small dataset, few steps)")
    
    return parser.parse_args()


def setup_device(device_arg: str) -> str:

    if device_arg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg
    
    if device == "cuda" and not torch.cuda.is_available():
        print("  CUDA requested but not available, falling back to CPU")
        device = "cpu"
    
    return device


def create_model(args) -> SLM:

    config = ModelConfig(
        vocab_size=50265, 
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        max_seq_len=args.max_seq_len,
        dropout=0.1,
        bias=False,
        tie_weights=True
    )
    
    config.validate()
    
    model = SLM(config)
    
    return model


def create_datasets(args, tokenizer):

    if args.test:
        print(" Test mode: using small in-memory dataset")
        
        texts = [
            "Once upon a time, there was a little girl who loved to play.",
            "She had a magic toy that could talk and sing songs.",
            "Every day, she would go to the park with her toy.",
            "One day, they met a friendly dog who wanted to play too.",
            "They all became best friends and had many adventures."
        ] * 20 
        
        train_dataset = SimpleTextDataset(tokenizer, texts, max_seq_len=args.max_seq_len)
        val_dataset = SimpleTextDataset(tokenizer, texts[:10], max_seq_len=args.max_seq_len)
        
    else:
        print("📚 Loading TinyStories dataset...")
        
        train_dataset = TinyStoriesDataset(
            tokenizer,
            split="train",
            max_seq_len=args.max_seq_len,
            max_samples=args.max_samples,
            streaming=args.streaming
        )
        
        val_dataset = TinyStoriesDataset(
            tokenizer,
            split="validation",
            max_seq_len=args.max_seq_len,
            max_samples=min(1000, args.max_samples) if args.max_samples else 1000,
            streaming=False  
        )
    
    train_loader = TinyStoriesDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=not args.streaming,
        num_workers=args.num_workers if not args.test else 0
    )
    
    val_loader = TinyStoriesDataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0  
    )
    
    return train_loader, val_loader


def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)
    
    if args.test:
        args.max_steps = 100
        args.eval_interval = 50
        args.log_interval = 20
        args.save_interval = 50
        args.batch_size = 4
        args.gradient_accumulation_steps = 2
        print(" Running in TEST mode (small dataset, 100 steps)")
    
    device = setup_device(args.device)
    
    print("=" * 60)
    print("          SLM Training")
    print("=" * 60)
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Mixed precision: {args.use_amp} ({args.dtype if args.use_amp else 'float32'})")
    print(f"Random seed: {args.seed}")
    print("-" * 60)
    
    print("🔤 Creating tokenizer...")
    tokenizer = Tokenizer()
    print(f"   Vocabulary size: {tokenizer.vocab_size}")
    
    print("🏗️  Creating model...")
    model = create_model(args)
    num_params = model.get_num_params()
    num_params_no_embed = model.get_num_params(non_embedding=True)
    
    print(f"   Total parameters: {num_params / 1e6:.2f}M")
    print(f"   Non-embedding: {num_params_no_embed / 1e6:.2f}M")
    print(f"   Architecture: {args.n_layers} layers, {args.n_heads} heads, {args.d_model} dim")
    print(f"   GQA ratio: {args.n_heads // args.n_kv_heads}:1")
    
    train_loader, val_loader = create_datasets(args, tokenizer)
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    print("⚙️  Setting up optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay
    )
    
    scheduler = SimpleCosineScheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        max_lr=args.learning_rate,
        min_lr=args.min_lr
    )
    
    train_config = {
        'max_steps': args.max_steps,
        'eval_interval': args.eval_interval,
        'log_interval': args.log_interval,
        'save_interval': args.save_interval,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'max_grad_norm': args.max_grad_norm,
        'checkpoint_dir': args.checkpoint_dir,
        'use_amp': args.use_amp and device == "cuda",
        'dtype': args.dtype,
        'batch_size': args.batch_size,
        'max_seq_len': args.max_seq_len
    }
    
    print("🚂 Creating trainer...")
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=train_config,
        device=device
    )
    
    if args.resume_from:
        print(f" Resuming from checkpoint: {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)
    
    print("-" * 60)
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Total steps: {args.max_steps}")
    print(f"Warmup steps: {args.warmup_steps}")
    print(f"Peak LR: {args.learning_rate:.2e}")
    print(f"Min LR: {args.min_lr:.2e}")
    print("=" * 60)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\n  Training interrupted by user")
        print("Saving checkpoint...")
        trainer.save_checkpoint()
    
    print("\n" + "=" * 60)
    print(" Training complete!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
