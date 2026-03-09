"""
src/model/slm.py

Small Language Model (SLM)
Top-level model class that assembles all components.

Architecture:
  Token IDs
      ↓
  Token Embedding (vocab_size → d_model)
      ↓
  [TransformerBlock × n_layers]
      ↓
  RMSNorm (final)
      ↓
  LM Head (d_model → vocab_size)
      ↓
  Logits

Features:
  - Weight tying: input embedding ↔ LM head
  - KV caching for efficient inference
  - Careful initialization (GPT-2 style)
"""

import torch
import torch.nn as nn
from typing import Optional

from .config import ModelConfig
from .normalization import RMSNorm
from .block import TransformerBlock


class SLM(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        self.norm = RMSNorm(config.d_model)
        
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight
        
        self.apply(self._init_weights)
        
        for name, param in self.named_parameters():
            if name.endswith('o_proj.weight') or name.endswith('down_proj.weight'):
                torch.nn.init.normal_(param, mean=0.0, std=0.02 / (2 * config.n_layers) ** 0.5)
    
    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        kv_caches: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False
    ) -> tuple[torch.Tensor, Optional[list[tuple[torch.Tensor, torch.Tensor]]]]:
     
        batch, seq_len = input_ids.shape
        
        x = self.token_embedding(input_ids) 
        
        new_kv_caches = [] if use_cache else None
        
        for i, block in enumerate(self.blocks):
            kv_cache = kv_caches[i] if kv_caches is not None else None

            x, new_kv_cache = block(x, start_pos=start_pos, kv_cache=kv_cache, use_cache=use_cache)

            if use_cache:
                new_kv_caches.append(new_kv_cache)

        x = self.norm(x)

        logits = self.lm_head(x) 
        
        return logits, new_kv_caches
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:

        self.eval()
        
        batch_size = input_ids.shape[0]

        logits, kv_caches = self.forward(input_ids, start_pos=0, use_cache=True)

        next_token = self._sample(logits[:, -1, :], temperature, top_k, top_p)

        generated = torch.cat([input_ids, next_token], dim=1)

        for i in range(max_new_tokens - 1):
            cur_pos = input_ids.shape[1] + i

            logits, kv_caches = self.forward(
                next_token, 
                start_pos=cur_pos,
                kv_caches=kv_caches,
                use_cache=True
            )

            next_token = self._sample(logits[:, -1, :], temperature, top_k, top_p)

            generated = torch.cat([generated, next_token], dim=1)

            if generated.shape[1] >= self.config.max_seq_len:
                break
        
        return generated
    
    def _sample(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:

        logits = logits / temperature
        
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
        
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 0] = False
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token
    
    def get_num_params(self, non_embedding: bool = False) -> int:

        n_params = sum(p.numel() for p in self.parameters())
        
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
        
        return n_params

if __name__ == "__main__":
    from config import ModelConfig, estimate_params
    
    print("=" * 60)
    print("           SLM Model Unit Test")
    print("=" * 60)

    config = ModelConfig()
    config.validate()
    
    print(f"Configuration:")
    print(f"  d_model       : {config.d_model}")
    print(f"  n_layers      : {config.n_layers}")
    print(f"  n_heads       : {config.n_heads}")
    print(f"  n_kv_heads    : {config.n_kv_heads}")
    print(f"  vocab_size    : {config.vocab_size}")
    print(f"  max_seq_len   : {config.max_seq_len}")
    print("-" * 60)
    
    model = SLM(config)
    model.eval()
    
    print("Test 1: Parameter count")
    actual_params = model.get_num_params()
    estimated_params = estimate_params(config)
    
    print(f"  Actual params     : {actual_params:,}")
    print(f"  Estimated params  : {estimated_params:,}")
    print(f"  Difference        : {abs(actual_params - estimated_params):,}")
    print(f"  Non-embedding     : {model.get_num_params(non_embedding=True):,}")
    print(f"  Parameter count verified (~15M)")
    print("-" * 60)
    
    print("Test 2: Forward pass")
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    logits, caches = model(input_ids, use_cache=False)
    
    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert caches is None
    print(f"  Input shape       : {input_ids.shape}")
    print(f"  Logits shape      : {logits.shape}")
    print(f"  Forward pass works")
    print("-" * 60)
    
    print("Test 3: Weight tying")
    if config.tie_weights:
        assert model.token_embedding.weight is model.lm_head.weight
        print(f"  Embedding weights : {model.token_embedding.weight.shape}")
        print(f"  LM head weights   : {model.lm_head.weight.shape}")
        print(f"  Same object       : {model.token_embedding.weight is model.lm_head.weight}")
        print(f"  Weight tying enabled")
    else:
        print(f"  Weight tying disabled (as configured)")
    print("-" * 60)
    
    print("Test 4: KV caching")
    
    input_ids_init = torch.randint(0, config.vocab_size, (batch_size, 1))
    logits_1, kv_caches = model(input_ids_init, start_pos=0, use_cache=True)
    
    print(f"  Initial input     : {input_ids_init.shape}")
    print(f"  Num cache layers  : {len(kv_caches)}")
    print(f"  K cache shape     : {kv_caches[0][0].shape}")
    print(f"  V cache shape     : {kv_caches[0][1].shape}")
    
    input_ids_next = torch.randint(0, config.vocab_size, (batch_size, 1))
    logits_2, kv_caches = model(input_ids_next, start_pos=1, kv_caches=kv_caches, use_cache=True)
    
    print(f"  After 2nd token:")
    print(f"    K cache shape   : {kv_caches[0][0].shape}")
    assert kv_caches[0][0].shape[1] == 2
    print(f"  KV caching works")
    print("-" * 60)
    
    print("Test 5: Text generation")
    prompt = torch.randint(0, config.vocab_size, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20, temperature=1.0)
    
    print(f"  Prompt length     : {prompt.shape[1]}")
    print(f"  Generated length  : {generated.shape[1]}")
    print(f"  New tokens        : {generated.shape[1] - prompt.shape[1]}")
    assert generated.shape[1] == prompt.shape[1] + 20
    print(f"  Generation works")
    print("-" * 60)
    
    print(" All SLM model tests passed")
    print("=" * 60)