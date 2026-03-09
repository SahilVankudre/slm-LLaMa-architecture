"""
src/model/ffn.py

SwiGLU Feed-Forward Network
Used in LLaMA, PaLM, and other modern transformers.

Standard FFN:
  FFN(x) = ReLU(W1 @ x) @ W2

SwiGLU FFN:
  FFN(x) = (SiLU(gate(x)) ⊙ up(x)) @ down
  
Where:
  gate: Linear(d_model → intermediate)
  up:   Linear(d_model → intermediate)
  down: Linear(intermediate → d_model)
  SiLU: Sigmoid Linear Unit (also called Swish)
  ⊙:    Element-wise multiplication

Key differences from standard FFN:
  1. Uses SiLU activation instead of ReLU
  2. Gating mechanism (element-wise multiplication)
  3. Three linear layers instead of two
  4. Intermediate dimension typically: int(2/3 * 4 * d_model)

Advantages:
  - Better empirical performance than ReLU
  - Gate acts as a learned feature selector
  - Smooth activation (SiLU) helps gradient flow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class SwiGLU(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.d_model = config.d_model
        self.intermediate = config.ffn_intermediate

        self.gate_proj = nn.Linear(config.d_model, config.ffn_intermediate, bias=config.bias)
        self.up_proj = nn.Linear(config.d_model, config.ffn_intermediate, bias=config.bias)
        self.down_proj = nn.Linear(config.ffn_intermediate, config.d_model, bias=config.bias)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        gate = F.silu(self.gate_proj(x))

        up = self.up_proj(x)

        hidden = gate * up

        output = self.down_proj(hidden)

        output = self.dropout(output)
        
        return output

class SwiGLUCombined(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.d_model = config.d_model
        self.intermediate = config.ffn_intermediate
        
        self.gate_up_proj = nn.Linear(
            config.d_model, 
            2 * config.ffn_intermediate, 
            bias=config.bias
        )
        
        self.down_proj = nn.Linear(config.ffn_intermediate, config.d_model, bias=config.bias)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        gate_up = self.gate_up_proj(x)

        gate, up = gate_up.chunk(2, dim=-1)
        
        hidden = F.silu(gate) * up
        
        output = self.down_proj(hidden)
        
        output = self.dropout(output)
        
        return output

if __name__ == "__main__":
    from config import ModelConfig
    
    print("=" * 60)
    print("           SwiGLU FFN Unit Test")
    print("=" * 60)
    
    config = ModelConfig()
    print(f"d_model       : {config.d_model}")
    print(f"intermediate  : {config.ffn_intermediate}")
    print(f"Expansion     : {config.ffn_intermediate / config.d_model:.2f}x")
    print("-" * 60)
    
    ffn_standard = SwiGLU(config)
    ffn_combined = SwiGLUCombined(config)
    
    ffn_standard.eval()
    ffn_combined.eval()
    
    batch_size = 2
    seq_len = 16
    x = torch.randn(batch_size, seq_len, config.d_model)
    
    print("Test 1: Standard SwiGLU")
    output_standard = ffn_standard(x)
    assert output_standard.shape == x.shape, f"Shape mismatch: {output_standard.shape} vs {x.shape}"
    print(f"  Input shape   : {x.shape}")
    print(f"  Output shape  : {output_standard.shape}")
    print(f"  Standard SwiGLU works")
    print("-" * 60)
    
    print("Test 2: Combined SwiGLU")
    output_combined = ffn_combined(x)
    assert output_combined.shape == x.shape, f"Shape mismatch: {output_combined.shape} vs {x.shape}"
    print(f"  Input shape   : {x.shape}")
    print(f"  Output shape  : {output_combined.shape}")
    print(f"  Combined SwiGLU works")
    print("-" * 60)
    
    params_standard = sum(p.numel() for p in ffn_standard.parameters())
    params_combined = sum(p.numel() for p in ffn_combined.parameters())
    
    print("Test 3: Parameter count")
    print(f"  Standard      : {params_standard:,}")
    print(f"  Combined      : {params_combined:,}")
    print(f"  Difference    : {abs(params_standard - params_combined):,}")
    assert params_standard == params_combined, "Param count should be identical"
    print(f"  Parameter counts match")
    print("-" * 60)
    
    print("Test 4: SiLU activation")
    test_input = torch.randn(10)
    silu_output = F.silu(test_input)
    assert silu_output.shape == test_input.shape
    
    print(f"  SiLU(0)       : {F.silu(torch.tensor(0.0)):.4f}")
    print(f"  SiLU(1)       : {F.silu(torch.tensor(1.0)):.4f}")
    print(f"  SiLU(-1)      : {F.silu(torch.tensor(-1.0)):.4f}")
    print(f"  SiLU activation works")
    print("-" * 60)
    
    print("All FFN tests passed")
    print("=" * 60)