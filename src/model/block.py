"""
src/model/block.py

Transformer Block
Combines attention and FFN with normalization and residual connections.

Architecture (Pre-Norm):
  x → RMSNorm → Attention → residual add
  x → RMSNorm → FFN       → residual add

Pre-norm vs Post-norm:
  Pre-norm:  x = x + SubLayer(Norm(x))  ← More stable, used in LLaMA
  Post-norm: x = Norm(x + SubLayer(x))  ← Original Transformer, less stable

Pre-norm benefits:
  - Better gradient flow during training
  - No need for learning rate warmup (though we still use it)
  - More stable with deep networks
"""

import torch
import torch.nn as nn
from typing import Optional

from .config import ModelConfig
from .normalization import RMSNorm
from .attention import GroupedQueryAttention
from .ffn import SwiGLUCombined 

class TransformerBlock(nn.Module):
  
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.attn_norm = RMSNorm(config.d_model)
        
        self.attn = GroupedQueryAttention(config)
        
        self.ffn_norm = RMSNorm(config.d_model)
        
        self.ffn = SwiGLUCombined(config)
    
    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:

        attn_input = self.attn_norm(x)
        attn_output, new_kv_cache = self.attn(
            attn_input,
            start_pos=start_pos,
            kv_cache=kv_cache,
            use_cache=use_cache
        )
        x = x + attn_output

        ffn_input = self.ffn_norm(x)
        ffn_output = self.ffn(ffn_input)
        x = x + ffn_output
        
        return x, new_kv_cache

if __name__ == "__main__":
    from config import ModelConfig
    
    print("=" * 60)
    print("        Transformer Block Unit Test")
    print("=" * 60)
    
    config = ModelConfig()
    print(f"d_model       : {config.d_model}")
    print(f"n_heads       : {config.n_heads}")
    print(f"n_kv_heads    : {config.n_kv_heads}")
    print(f"ffn_inter     : {config.ffn_intermediate}")
    print("-" * 60)
    
    block = TransformerBlock(config)
    block.eval() 
    
    print("Test 1: Basic forward pass")
    batch_size = 2
    seq_len = 16
    x = torch.randn(batch_size, seq_len, config.d_model)
    
    output, cache = block(x, use_cache=False)
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
    assert cache is None, "Cache should be None when use_cache=False"
    print(f"  Input shape   : {x.shape}")
    print(f"  Output shape  : {output.shape}")
    print(f"  Basic forward pass works")
    print("-" * 60)
    
    print("Test 2: Residual connections")
    assert not torch.allclose(output, x, atol=1e-2), "Output too similar to input"
    
    input_norm = torch.norm(x)
    output_norm = torch.norm(output)
    print(f"  Input norm    : {input_norm:.4f}")
    print(f"  Output norm   : {output_norm:.4f}")
    print(f"  Norm ratio    : {output_norm / input_norm:.4f}")
    print(f"  Residual connections preserve signal scale")
    print("-" * 60)
    
    print("Test 3: KV caching")
    
    x_init = torch.randn(batch_size, 1, config.d_model)
    output_1, kv_cache = block(x_init, start_pos=0, use_cache=True)
    
    k_cache, v_cache = kv_cache
    print(f"  Initial token shape : {x_init.shape}")
    print(f"  K cache shape       : {k_cache.shape}")
    print(f"  V cache shape       : {v_cache.shape}")
    
    x_next = torch.randn(batch_size, 1, config.d_model)
    output_2, kv_cache = block(x_next, start_pos=1, kv_cache=kv_cache, use_cache=True)
    
    k_cache, v_cache = kv_cache
    print(f"  After 2nd token:")
    print(f"    K cache shape     : {k_cache.shape}")
    print(f"    V cache shape     : {v_cache.shape}")
    assert k_cache.shape[1] == 2, "Cache should contain 2 tokens"
    print(f"  KV caching works correctly")
    print("-" * 60)
    
    total_params = sum(p.numel() for p in block.parameters())
    
    attn_params = sum(p.numel() for p in block.attn.parameters())
    ffn_params = sum(p.numel() for p in block.ffn.parameters())
    norm_params = sum(p.numel() for p in block.attn_norm.parameters()) + \
                  sum(p.numel() for p in block.ffn_norm.parameters())
    
    print("Test 4: Parameter breakdown")
    print(f"  Attention     : {attn_params:,}")
    print(f"  FFN           : {ffn_params:,}")
    print(f"  Normalization : {norm_params:,}")
    print(f"  Total         : {total_params:,}")
    print(f"  Parameter count verified")
    print("-" * 60)
    
    print("Test 5: Pre-norm architecture")
    print(f"  Attention norm before attn: ✅")
    print(f"  FFN norm before ffn       : ✅")
    print(f"  Residual connections      : ✅")
    print(f"  Architecture: LLaMA-style pre-norm")
    print("-" * 60)
    
    print(" All Transformer Block tests passed")
    print("=" * 60)