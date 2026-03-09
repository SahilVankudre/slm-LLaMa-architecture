"""
src/training/trainer.py

Trainer for SLM
Handles the full training loop with modern best practices.

Features:
  - Gradient accumulation (simulate larger batches)
  - Mixed precision training (fp16/bf16)
  - Gradient clipping (prevent exploding gradients)
  - Checkpointing (save/resume training)
  - Evaluation during training
  - Progress logging

Training loop:
  1. Load batch from dataloader
  2. Forward pass (compute loss)
  3. Backward pass (accumulate gradients)
  4. Clip gradients
  5. Optimizer step
  6. Scheduler step
  7. Periodic evaluation and checkpointing
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from tqdm.auto import tqdm
import json
import time
from typing import Optional, Dict


class Trainer:
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader,
        val_dataloader,
        optimizer,
        scheduler,
        config: Dict,
        device: str = 'cuda'
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        
        self.model.to(device)
        
        self.max_steps = config.get('max_steps', 10000)
        self.eval_interval = config.get('eval_interval', 500)
        self.log_interval = config.get('log_interval', 100)
        self.save_interval = config.get('save_interval', 1000)
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        self.use_amp = config.get('use_amp', True) and device == 'cuda'
        self.dtype = config.get('dtype', 'bfloat16')  
        
        if self.use_amp:
            self.scaler = GradScaler(enabled=(self.dtype == 'float16'))
        else:
            self.scaler = None
        
        self.current_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_stats = {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': []
        }
    
    def train(self):
        print("=" * 60)
        print("Starting training...")
        print(f"  Max steps: {self.max_steps}")
        print(f"  Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print(f"  Effective batch size: {self.config.get('batch_size', 32) * self.gradient_accumulation_steps}")
        print(f"  Device: {self.device}")
        print(f"  Mixed precision: {self.use_amp} ({self.dtype if self.use_amp else 'float32'})")
        print("=" * 60)
        
        self.model.train()
        
        pbar = tqdm(total=self.max_steps, desc="Training")
        pbar.update(self.current_step)
        
        running_loss = 0.0
        start_time = time.time()
        
        while self.current_step < self.max_steps:
            for batch in self.train_dataloader:
                if self.current_step >= self.max_steps:
                    break
                
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                with autocast(enabled=self.use_amp, dtype=torch.bfloat16 if self.dtype == 'bfloat16' else torch.float16):
                    logits, _ = self.model(input_ids)
                    loss = self._compute_loss(logits, labels)
                    
                    loss = loss / self.gradient_accumulation_steps
                
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                running_loss += loss.item()
                
                if (self.current_step + 1) % self.gradient_accumulation_steps == 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                    
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad(set_to_none=True)
                    
                    self.scheduler.step()
                
                self.current_step += 1
                pbar.update(1)
                
                if self.current_step % self.log_interval == 0:
                    avg_loss = running_loss / self.log_interval * self.gradient_accumulation_steps
                    lr = self.scheduler.get_last_lr()[0]
                    elapsed = time.time() - start_time
                    tokens_per_sec = (self.log_interval * self.config.get('batch_size', 32) * 
                                    self.config.get('max_seq_len', 512)) / elapsed
                    
                    pbar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{lr:.2e}',
                        'tok/s': f'{tokens_per_sec:.0f}'
                    })
                    
                    self.training_stats['train_losses'].append(avg_loss)
                    self.training_stats['learning_rates'].append(lr)
                    
                    running_loss = 0.0
                    start_time = time.time()
                
                if self.current_step % self.eval_interval == 0:
                    val_loss = self.evaluate()
                    self.training_stats['val_losses'].append(val_loss)
                    
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint(is_best=True)
                    
                    self.model.train()
                
                if self.current_step % self.save_interval == 0:
                    self.save_checkpoint()
        
        pbar.close()
        
        print("\nTraining complete!")
        val_loss = self.evaluate()
        self.save_checkpoint(is_best=False)
        
        print(f"Final validation loss: {val_loss:.4f}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
    
    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:

        logits = logits.reshape(-1, logits.size(-1))
        labels = labels.reshape(-1)
        
        loss = nn.functional.cross_entropy(logits, labels, ignore_index=-100)
        
        return loss
    
    @torch.no_grad()
    def evaluate(self) -> float:

        if self.val_dataloader is None:
            return 0.0
        
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_dataloader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            with torch.amp.autocast(enabled=self.use_amp, dtype=torch.bfloat16 if self.dtype == 'bfloat16' else torch.float16):
                logits, _ = self.model(input_ids)
                loss = self._compute_loss(logits, labels)
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        
        print(f"\nValidation loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def save_checkpoint(self, is_best: bool = False):

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'current_step': self.current_step,
            'current_epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'training_stats': self.training_stats,
            'config': self.config
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{self.current_step}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")
        
        stats_path = self.checkpoint_dir / "training_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_step = checkpoint['current_step']
        self.current_epoch = checkpoint['current_epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_stats = checkpoint['training_stats']
        
        print(f"Loaded checkpoint from step {self.current_step}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

if __name__ == "__main__":
    import sys
    sys.path.append('/mnt/user-data/outputs')
    
    from src.model.slm import SLM
    from src.model.config import ModelConfig
    from src.tokenizer.tokenizer import Tokenizer
    from src.training.dataset import SimpleTextDataset, TinyStoriesDataLoader
    from src.training.scheduler import SimpleCosineScheduler
    
    print("=" * 60)
    print("          Trainer Unit Test")
    print("=" * 60)
    
    config = ModelConfig(
        d_model=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=50265, 
        max_seq_len=64
    )
    
    model = SLM(config)
    print(f"Model parameters: {model.get_num_params() / 1e6:.2f}M")
    
    tokenizer = Tokenizer()
    texts = [
        "Once upon a time, there was a little girl who loved to play.",
        "She had a magic toy that could talk and sing.",
        "Every day, she would go to the park with her toy."
    ] * 10  
    
    train_dataset = SimpleTextDataset(tokenizer, texts, max_seq_len=64)
    train_loader = TinyStoriesDataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = TinyStoriesDataLoader(train_dataset, batch_size=2, shuffle=False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95))
    scheduler = SimpleCosineScheduler(
        optimizer,
        warmup_steps=10,
        max_steps=100,
        max_lr=1e-3,
        min_lr=1e-4
    )
    
    train_config = {
        'max_steps': 100,
        'eval_interval': 50,
        'log_interval': 20,
        'save_interval': 50,
        'gradient_accumulation_steps': 2,
        'max_grad_norm': 1.0,
        'checkpoint_dir': '/tmp/slm_test_checkpoints',
        'use_amp': False,  
        'batch_size': 2,
        'max_seq_len': 64
    }
    
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=train_config,
        device='cpu'  
    )
    
    print("Running mini training loop (100 steps)...")
    print("-" * 60)
    
    trainer.train()
    
    print("-" * 60)
    print("✅ Trainer test completed")
    print(f"Final step: {trainer.current_step}")
    print(f"Best val loss: {trainer.best_val_loss:.4f}")
    print("=" * 60)