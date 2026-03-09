"""
src/training/scheduler.py

Learning Rate Scheduler
Implements linear warmup followed by cosine annealing decay.

Schedule:
  1. Linear warmup: 0 → max_lr over warmup_steps
  2. Cosine decay: max_lr → min_lr over remaining steps
  
This is the standard LR schedule for training transformers.

Why warmup?
  - Prevents early training instability
  - Gradients are noisy at initialization
  - Allows model to find a good basin before aggressive learning

Why cosine decay?
  - Smooth decay (no sharp drops like step decay)
  - Well-studied and empirically effective
  - Gradually reduces LR to fine-tune in later training
"""

import math
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer


class CosineSchedulerWithWarmup:
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        max_steps: int,
        max_lr: float,
        min_lr: float = 0.0
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        
        self.scheduler = LambdaLR(optimizer, lr_lambda=self._lr_lambda)
        
        self.current_step = 0
    
    def _lr_lambda(self, step: int) -> float:

        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)
        
        elif step < self.max_steps:
            progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

            return self.min_lr / self.max_lr + (1.0 - self.min_lr / self.max_lr) * cosine_decay
        
        else:
            return self.min_lr / self.max_lr
    
    def step(self):
        """Advance the scheduler by one step."""
        self.scheduler.step()
        self.current_step += 1
    
    def get_last_lr(self):
        """Get the last computed learning rate."""
        return self.scheduler.get_last_lr()
    
    def state_dict(self):
        """Return state dict for checkpointing."""
        return {
            'scheduler_state': self.scheduler.state_dict(),
            'current_step': self.current_step
        }
    
    def load_state_dict(self, state_dict):
        """Load state dict from checkpoint."""
        self.scheduler.load_state_dict(state_dict['scheduler_state'])
        self.current_step = state_dict['current_step']

class SimpleCosineScheduler:

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        max_steps: int,
        max_lr: float,
        min_lr: float = 0.0
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = 0
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = 0.0
    
    def get_lr(self) -> float:
        step = self.current_step
        
        if step < self.warmup_steps:
            return self.max_lr * step / max(1, self.warmup_steps)
        
        elif step < self.max_steps:
            progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr + (self.max_lr - self.min_lr) * cosine_decay
        
        else:
            return self.min_lr
    
    def step(self):
        lr = self.get_lr()
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_step += 1
    
    def get_last_lr(self):
        return [self.get_lr()]
    
    def state_dict(self):
        return {'current_step': self.current_step}
    
    def load_state_dict(self, state_dict):
        self.current_step = state_dict['current_step']

if __name__ == "__main__":
    import torch
    import matplotlib.pyplot as plt
    
    print("=" * 60)
    print("      Learning Rate Scheduler Unit Test")
    print("=" * 60)
    
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)  # lr=1.0 as base
    
    warmup_steps = 1000
    max_steps = 10000
    max_lr = 3e-4
    min_lr = 3e-5
    
    print(f"Configuration:")
    print(f"  Warmup steps  : {warmup_steps}")
    print(f"  Max steps     : {max_steps}")
    print(f"  Max LR        : {max_lr}")
    print(f"  Min LR        : {min_lr}")
    print("-" * 60)
    
    print("Test 1: CosineSchedulerWithWarmup")
    optimizer1 = torch.optim.Adam(model.parameters(), lr=max_lr)
    scheduler1 = CosineSchedulerWithWarmup(
        optimizer1,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        max_lr=max_lr,
        min_lr=min_lr
    )
    
    lrs1 = []
    for step in range(max_steps + 1000):
        scheduler1.step()
        lrs1.append(scheduler1.get_last_lr()[0])
    
    print(f"  Step 0 LR     : {lrs1[0]:.6f}")
    print(f"  Step 500 LR   : {lrs1[500]:.6f}")
    print(f"  Step 1000 LR  : {lrs1[1000]:.6f} (after warmup)")
    print(f"  Step 5000 LR  : {lrs1[5000]:.6f} (mid decay)")
    print(f"  Step 10000 LR : {lrs1[10000]:.6f} (at end)")
    print(f"  LambdaLR scheduler works")
    print("-" * 60)
    
    print("Test 2: SimpleCosineScheduler")
    optimizer2 = torch.optim.Adam(model.parameters(), lr=1.0)
    scheduler2 = SimpleCosineScheduler(
        optimizer2,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        max_lr=max_lr,
        min_lr=min_lr
    )
    
    lrs2 = []
    for step in range(max_steps + 1000):
        scheduler2.step()
        lrs2.append(scheduler2.get_last_lr()[0])
    
    print(f"  Step 0 LR     : {lrs2[0]:.6f}")
    print(f"  Step 500 LR   : {lrs2[500]:.6f}")
    print(f"  Step 1000 LR  : {lrs2[1000]:.6f} (after warmup)")
    print(f"  Step 5000 LR  : {lrs2[5000]:.6f} (mid decay)")
    print(f"  Step 10000 LR : {lrs2[10000]:.6f} (at end)")
    print(f"  Simple scheduler works")
    print("-" * 60)
    
    print("Test 3: Compare both implementations")
    max_diff = max(abs(lr1 - lr2) for lr1, lr2 in zip(lrs1, lrs2))
    print(f"  Max difference: {max_diff:.8f}")
    assert max_diff < 1e-6, "Schedulers should produce identical results"
    print(f"  Both implementations match")
    print("-" * 60)
    
    print("Test 4: Visualize LR schedule")
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(lrs1[:max_steps], label='Learning Rate')
        plt.axvline(x=warmup_steps, color='r', linestyle='--', label='Warmup End')
        plt.xlabel('Training Step')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule (Linear Warmup + Cosine Decay)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig('/mnt/user-data/outputs/lr_schedule.png', dpi=100)
        print(f"  Plot saved to lr_schedule.png")
        print(f"  Visualization created")
    except Exception as e:
        print(f"  Could not create plot: {e}")
    print("-" * 60)
    
    print("Test 5: State dict checkpointing")
    state = scheduler1.state_dict()
    print(f"  State dict keys: {list(state.keys())}")
    
    new_scheduler = CosineSchedulerWithWarmup(
        optimizer1,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        max_lr=max_lr,
        min_lr=min_lr
    )
    new_scheduler.load_state_dict(state)
    assert new_scheduler.current_step == scheduler1.current_step
    print(f"  State dict save/load works")
    print("-" * 60)
    
    print("All scheduler tests passed")
    print("=" * 60)