"""
src/model/normalization.py

RMSNorm — Root Mean Square Layer Normalization
Used in LLaMA, Mistral, and other modern transformers.

Simpler than LayerNorm:
  - No mean subtraction
  - No learned bias
  - Only learns a scale (weight) parameter

Formula:
  RMS(x) = sqrt(mean(x²) + ε)
  output = (x / RMS(x)) * weight

Where:
  x      : input tensor
  weight : learnable scale parameter (initialized to 1)
  ε      : small constant for numerical stability (1e-6)
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x: torch.Tensor) -> torch.Tensor:

        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        return x / rms
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        output = self._norm(x.float()).type_as(x)

        return output * self.weight

if __name__ == "__main__":
    print("=" * 50)
    print("          RMSNorm Unit Test")
    print("=" * 50)
    
    batch_size = 2
    seq_len = 4
    dim = 384
    
    norm = RMSNorm(dim)
    
    x = torch.randn(batch_size, seq_len, dim)
    
    output = norm(x)
    
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
    
    rms = torch.sqrt(torch.mean(output ** 2, dim=-1))
    
    print(f"Input shape    : {x.shape}")
    print(f"Output shape   : {output.shape}")
    print(f"Input RMS mean : {torch.sqrt(torch.mean(x**2, dim=-1)).mean().item():.4f}")
    print(f"Output RMS mean: {rms.mean().item():.4f}")
    print(f"Weight shape   : {norm.weight.shape}")
    print(f"Weight init    : all ones = {torch.allclose(norm.weight, torch.ones_like(norm.weight))}")
    print("-" * 50)
    print("RMSNorm working correctly")
    print("=" * 50)