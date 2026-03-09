"""
src/model/rope.py

RoPE — Rotary Positional Embedding
Introduced in "RoFormer: Enhanced Transformer with Rotary Position Embedding"

Key idea:
  Instead of adding positional embeddings to tokens, we rotate the Q and K
  vectors in attention based on their position. This encodes relative position
  implicitly in the dot product.

Advantages over learned positional embeddings:
  - Better length generalization (can handle sequences longer than training)
  - No extra parameters
  - Relative position encoding naturally emerges from the rotation

Math:
  For each position m and dimension pair (2i, 2i+1):
    θ_i = 10000^(-2i/d)           # base frequency for dimension pair i
    
  Rotation matrix for position m:
    R_m = [[cos(mθ), -sin(mθ)],
           [sin(mθ),  cos(mθ)]]
  
  Applied to Q and K:
    Q_rotated = R_m @ Q
    K_rotated = R_m @ K
"""

import torch
import torch.nn as nn
import math


class RoPE(nn.Module):

    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):

        dim = torch.arange(0, self.head_dim, 2, dtype=torch.float32)

        freqs = 1.0 / (self.base ** (dim / self.head_dim))

        positions = torch.arange(seq_len, dtype=torch.float32)

        freqs = torch.outer(positions, freqs)

        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)
    
    def _apply_rotary_emb(
        self, 
        x: torch.Tensor, 
        cos: torch.Tensor, 
        sin: torch.Tensor
    ) -> torch.Tensor:

        x = x.reshape(*x.shape[:-1], -1, 2)
        
        x1, x2 = x.unbind(dim=-1) 
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos

        x_rot = torch.stack([x1_rot, x2_rot], dim=-1)

        return x_rot.flatten(-2)
    
    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor,
        start_pos: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:

        seq_len = q.shape[1]

        cos = self.cos_cached[start_pos:start_pos + seq_len]
        sin = self.sin_cached[start_pos:start_pos + seq_len]

        q_rot = self._apply_rotary_emb(q, cos, sin)
        k_rot = self._apply_rotary_emb(k, cos, sin)
        
        return q_rot, k_rot

if __name__ == "__main__":
    print("=" * 50)
    print("            RoPE Unit Test")
    print("=" * 50)
    
    batch_size = 2
    seq_len = 8
    n_heads = 6
    n_kv_heads = 2
    head_dim = 64
    max_seq_len = 512
    
    rope = RoPE(head_dim=head_dim, max_seq_len=max_seq_len)
    
    q = torch.randn(batch_size, seq_len, n_heads, head_dim)
    k = torch.randn(batch_size, seq_len, n_kv_heads, head_dim)
    
    print(f"Input Q shape  : {q.shape}")
    print(f"Input K shape  : {k.shape}")
    print(f"Head dimension : {head_dim}")
    print(f"Max seq length : {max_seq_len}")
    print(f"Cache shapes   : cos={rope.cos_cached.shape}, sin={rope.sin_cached.shape}")
    print("-" * 50)
    
    q_rot, k_rot = rope(q, k)
    
    assert q_rot.shape == q.shape, f"Q shape mismatch: {q_rot.shape} vs {q.shape}"
    assert k_rot.shape == k.shape, f"K shape mismatch: {k_rot.shape} vs {k.shape}"
    
    q_norm_before = torch.norm(q, dim=-1).mean()
    q_norm_after = torch.norm(q_rot, dim=-1).mean()
    norm_diff = abs(q_norm_before - q_norm_after)
    
    print(f"Q norm before  : {q_norm_before:.6f}")
    print(f"Q norm after   : {q_norm_after:.6f}")
    print(f"Norm difference: {norm_diff:.6f} (should be ~0)")
    
    print("-" * 50)
    print("Testing incremental decoding...")
    start_pos = 10
    q_inc = torch.randn(batch_size, 1, n_heads, head_dim) 
    k_inc = torch.randn(batch_size, 1, n_kv_heads, head_dim)
    
    q_rot_inc, k_rot_inc = rope(q_inc, k_inc, start_pos=start_pos)
    assert q_rot_inc.shape == q_inc.shape
    print(f"Incremental decode works: start_pos={start_pos}, seq_len=1")
    
    print("-" * 50)
    print("RoPE working correctly")
    print("=" * 50)