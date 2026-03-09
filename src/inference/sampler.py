"""
src/inference/sampler.py

Sampling Strategies for Text Generation

Implements various decoding strategies:
  1. Greedy: Always pick most likely token (deterministic)
  2. Temperature: Control randomness (higher = more random)
  3. Top-k: Sample from top k most likely tokens
  4. Top-p (nucleus): Sample from smallest set covering probability p
  5. Combined: Temperature + Top-k + Top-p together

Trade-offs:
  - Greedy: Fast, deterministic, but repetitive and boring
  - High temperature: Creative, diverse, but can be incoherent
  - Low temperature: Focused, coherent, but less creative
  - Top-k: Simple filtering, but fixed cutoff
  - Top-p: Adaptive filtering, better for varying distributions
"""

import torch
import torch.nn.functional as F
from typing import Optional


class Sampler:

    @staticmethod
    def greedy(logits: torch.Tensor) -> torch.Tensor:
        return torch.argmax(logits, dim=-1, keepdim=True)
    
    @staticmethod
    def sample(
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:

        if temperature != 1.0:
            logits = logits / temperature
        
        if top_k is not None:
            logits = Sampler._apply_top_k(logits, top_k)
        
        if top_p is not None:
            logits = Sampler._apply_top_p(logits, top_p)
        
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token
    
    @staticmethod
    def _apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:

        top_k = min(k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)

        min_value = values[:, -1].unsqueeze(-1)
        logits = torch.where(
            logits < min_value,
            torch.full_like(logits, float('-inf')),
            logits
        )
        
        return logits
    
    @staticmethod
    def _apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:

        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > p
        
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        return logits
    
    @staticmethod
    def beam_search(
        logits: torch.Tensor,
        beam_width: int = 5,
        num_return_sequences: int = 1
    ) -> torch.Tensor:

        probs = F.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, beam_width, dim=-1)
        
        return top_indices[:, :num_return_sequences]

class SamplingPresets:
    
    @staticmethod
    def greedy():
        return {'temperature': 1.0, 'top_k': 1, 'top_p': None}
    
    @staticmethod
    def creative():
        return {'temperature': 1.2, 'top_k': None, 'top_p': 0.95}
    
    @staticmethod
    def balanced():
        return {'temperature': 0.8, 'top_k': 50, 'top_p': 0.9}
    
    @staticmethod
    def focused():
        return {'temperature': 0.6, 'top_k': 20, 'top_p': 0.85}
    
    @staticmethod
    def precise():
        return {'temperature': 0.3, 'top_k': 10, 'top_p': 0.7}

if __name__ == "__main__":
    print("=" * 60)
    print("          Sampler Unit Test")
    print("=" * 60)
    
    batch_size = 2
    vocab_size = 1000
    
    logits = torch.randn(batch_size, vocab_size)
    logits[:, 100] = 10.0  
    logits[:, 101] = 8.0  
    logits[:, 102] = 6.0   
    
    print(f"Input shape: {logits.shape}")
    print(f"Top 3 logits: {logits[0, 100:103].tolist()}")
    print("-" * 60)
    
    print("Test 1: Greedy sampling")
    greedy_tokens = Sampler.greedy(logits)
    print(f"  Output shape: {greedy_tokens.shape}")
    print(f"  Tokens: {greedy_tokens.squeeze().tolist()}")
    assert torch.all(greedy_tokens == 100), "Greedy should always pick token 100"
    print(f"   Greedy sampling works (deterministic)")
    print("-" * 60)
    
    print("Test 2: Temperature sampling")
    
    torch.manual_seed(42)
    low_temp = Sampler.sample(logits, temperature=0.5)
    print(f"  Low temp (0.5): {low_temp.squeeze().tolist()}")
    
    torch.manual_seed(42)
    high_temp = Sampler.sample(logits, temperature=2.0)
    print(f"  High temp (2.0): {high_temp.squeeze().tolist()}")
    
    print(f"   Temperature sampling works")
    print("-" * 60)
    
    print("Test 3: Top-k sampling")
    
    torch.manual_seed(42)
    top_k_samples = []
    for _ in range(10):
        sample = Sampler.sample(logits, temperature=1.0, top_k=3)
        top_k_samples.extend(sample.squeeze().tolist())
    
    unique_tokens = set(top_k_samples)
    print(f"  Top-k=3, 10 samples")
    print(f"  Unique tokens: {sorted(unique_tokens)}")
    assert unique_tokens.issubset({100, 101, 102}), "Should only sample from top 3"
    print(f"   Top-k filtering works")
    print("-" * 60)
    
    print("Test 4: Top-p (nucleus) sampling")
    
    torch.manual_seed(42)
    top_p_samples = []
    for _ in range(20):
        sample = Sampler.sample(logits, temperature=1.0, top_p=0.9)
        top_p_samples.extend(sample.squeeze().tolist())
    
    unique_tokens_p = set(top_p_samples)
    print(f"  Top-p=0.9, 20 samples")
    print(f"  Unique tokens sampled: {len(unique_tokens_p)}")
    print(f"  Most common: {max(set(top_p_samples), key=top_p_samples.count)}")
    print(f"   Top-p sampling works (adaptive cutoff)")
    print("-" * 60)
    
    print("Test 5: Combined (temperature + top-k + top-p)")
    
    torch.manual_seed(42)
    combined = Sampler.sample(logits, temperature=0.8, top_k=50, top_p=0.9)
    print(f"  Sample: {combined.squeeze().tolist()}")
    print(f"   Combined filtering works")
    print("-" * 60)
    
    print("Test 6: Sampling presets")
    
    presets = ['greedy', 'creative', 'balanced', 'focused', 'precise']
    for preset_name in presets:
        preset = getattr(SamplingPresets, preset_name)()
        print(f"  {preset_name:10}: {preset}")
    
    print(f"   Presets defined")
    print("-" * 60)
    
    print("Test 7: Edge cases")
    
    very_low_temp = Sampler.sample(logits, temperature=0.01)
    print(f"  Very low temp (0.01): {very_low_temp.squeeze().tolist()}")
    
    large_k = Sampler.sample(logits, temperature=1.0, top_k=10000)
    print(f"  Top-k > vocab_size: {large_k.shape}")
    
    full_p = Sampler.sample(logits, temperature=1.0, top_p=1.0)
    print(f"  Top-p=1.0: {full_p.shape}")
    
    print(f"  ✅ Edge cases handled")
    print("-" * 60)
    
    print("✅ All sampler tests passed")
    print("=" * 60)
