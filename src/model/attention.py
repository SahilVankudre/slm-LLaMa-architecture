"""
src/model/attention.py

Grouped Query Attention (GQA) with KV Cache
Used in LLaMA 2/3, Mistral, and other modern transformers.

Key concepts:
  1. Multi-Head Attention (MHA): n_heads for Q, K, V
  2. Multi-Query Attention (MQA): n_heads for Q, 1 head for K, V
  3. Grouped Query Attention (GQA): n_heads for Q, n_kv_heads for K, V
     - Each KV head is shared by (n_heads // n_kv_heads) query heads
     - Reduces KV cache memory while maintaining quality

KV Cache:
  - During inference, cache K and V from previous tokens
  - Only compute K, V for new tokens
  - Reduces computation from O(T²) to O(T) per decode step
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from .config import ModelConfig
from .rope import RoPE


class GroupedQueryAttention(nn.Module):
  
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model
        
        self.n_rep = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=config.bias)
        
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=config.bias)
        
        self.o_proj = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=config.bias)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.rope = RoPE(head_dim=config.head_dim, max_seq_len=config.max_seq_len)
        
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        
        if not self.flash:
            mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
            self.register_buffer("causal_mask", mask.view(1, 1, config.max_seq_len, config.max_seq_len))
    
    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:

        batch, seq_len, n_kv_heads, head_dim = x.shape
        
        if self.n_rep == 1:
            return x

        x = x[:, :, :, None, :].expand(batch, seq_len, n_kv_heads, self.n_rep, head_dim)
        
        return x.reshape(batch, seq_len, n_kv_heads * self.n_rep, head_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with optional KV caching.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            start_pos: Starting position (for KV cache during incremental decoding)
            kv_cache: Tuple of (k_cache, v_cache) from previous step
            use_cache: Whether to return updated KV cache
        
        Returns:
            (output, new_kv_cache):
                output: [batch, seq_len, d_model]
                new_kv_cache: Updated (k_cache, v_cache) if use_cache=True, else None
        """
        batch, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = v.view(batch, seq_len, self.n_kv_heads, self.head_dim)
        
        q, k = self.rope(q, k, start_pos=start_pos)
        
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)
        
        new_kv_cache = None
        if use_cache:
            new_kv_cache = (k, v)

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        if self.flash:
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True
            )
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(
                self.causal_mask[:, :, :seq_len, :seq_len] == 0,
                float('-inf')
            )
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            
            attn_output = attn_weights @ v
        
        attn_output = attn_output.transpose(1, 2)
        
        attn_output = attn_output.contiguous().view(batch, seq_len, self.n_heads * self.head_dim)
        
        output = self.o_proj(attn_output)
        output = self.resid_dropout(output)
        
        return output, new_kv_cache

if __name__ == "__main__":
    from config import ModelConfig
    
    print("=" * 60)
    print("      Grouped Query Attention Unit Test")
    print("=" * 60)
    
    config = ModelConfig()
    print(f"d_model       : {config.d_model}")
    print(f"n_heads       : {config.n_heads}")
    print(f"n_kv_heads    : {config.n_kv_heads}")
    print(f"head_dim      : {config.head_dim}")
    print(f"GQA ratio     : {config.gqa_ratio}:1")
    print("-" * 60)
    
    attn = GroupedQueryAttention(config)
    attn.eval() 
    
    print("Test 1: Basic forward pass")
    batch_size = 2
    seq_len = 16
    x = torch.randn(batch_size, seq_len, config.d_model)
    
    output, cache = attn(x, use_cache=False)
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
    assert cache is None, "Cache should be None when use_cache=False"
    print(f"  Input shape  : {x.shape}")
    print(f"  Output shape : {output.shape}")
    print(f"  Basic forward pass works")
    print("-" * 60)
    
    print("Test 2: KV caching (incremental decoding)")
    
    x_init = torch.randn(batch_size, 1, config.d_model)
    output_1, kv_cache = attn(x_init, start_pos=0, use_cache=True)
    
    k_cache, v_cache = kv_cache
    print(f"  Initial token shape : {x_init.shape}")
    print(f"  K cache shape       : {k_cache.shape}")
    print(f"  V cache shape       : {v_cache.shape}")
    
    x_next = torch.randn(batch_size, 1, config.d_model)
    output_2, kv_cache = attn(x_next, start_pos=1, kv_cache=kv_cache, use_cache=True)
    
    k_cache, v_cache = kv_cache
    print(f"  After 2nd token:")
    print(f"    K cache shape     : {k_cache.shape}")
    print(f"    V cache shape     : {v_cache.shape}")
    assert k_cache.shape[1] == 2, "Cache should contain 2 tokens"
    print(f"   KV caching works")
    print("-" * 60)
    
    print("Test 3: Flash Attention")
    print(f"  Flash attention available: {attn.flash}")
    if attn.flash:
        print(f"   Using optimized flash attention")
    else:
        print(f"    Using manual attention (slower)")
    print("-" * 60)
    
    total_params = sum(p.numel() for p in attn.parameters())
    print(f"Test 4: Parameters")
    print(f"  Total attention params: {total_params:,}")
    print(f"   All tests passed")
    print("=" * 60)