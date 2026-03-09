"""
src/model/config.py

ModelConfig — single source of truth for all model architecture parameters.
Every model component (attention, FFN, block, SLM) imports from here.

Target: ~15M parameters
Architecture: LLaMA-style transformer
  - RMSNorm (pre-norm)
  - RoPE positional embeddings
  - Grouped Query Attention (GQA)
  - SwiGLU FFN
  - KV Cache (inference)
"""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:

    vocab_size: int = 8192

    d_model: int = 384            
    n_layers: int = 8           
    n_heads: int = 6                
    n_kv_heads: int = 2           
                            
    ffn_intermediate: int = 1024

    max_seq_len: int = 512       

    dropout: float = 0.1  

    bias: bool = False              
    tie_weights: bool = True        

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )
        return self.d_model // self.n_heads

    @property
    def gqa_ratio(self) -> int:
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        )
        return self.n_heads // self.n_kv_heads

    def validate(self):
        assert self.d_model > 0
        assert self.n_layers > 0
        assert self.n_heads > 0
        assert self.n_kv_heads > 0
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert 0.0 <= self.dropout < 1.0
        assert self.ffn_intermediate > 0
        assert self.vocab_size > 0
        assert self.max_seq_len > 0

def get_default_config() -> ModelConfig:
    cfg = ModelConfig()
    cfg.validate()
    return cfg

def estimate_params(cfg: ModelConfig) -> int:

    embed = cfg.vocab_size * cfg.d_model

    q_proj  = cfg.d_model * cfg.n_heads    * cfg.head_dim
    kv_proj = cfg.d_model * cfg.n_kv_heads * cfg.head_dim * 2 
    o_proj  = cfg.d_model * cfg.d_model
    attn    = q_proj + kv_proj + o_proj

    ffn = (cfg.d_model * cfg.ffn_intermediate * 2) + (cfg.ffn_intermediate * cfg.d_model)

    norms = 2 * cfg.d_model

    block = attn + ffn + norms

    final_norm = cfg.d_model
    lm_head    = 0 if cfg.tie_weights else cfg.vocab_size * cfg.d_model

    total = embed + (cfg.n_layers * block) + final_norm + lm_head
    return total

if __name__ == "__main__":
    cfg = get_default_config()
    params = estimate_params(cfg)

    print("=" * 45)
    print("         SLM Model Configuration")
    print("=" * 45)
    print(f"  d_model          : {cfg.d_model}")
    print(f"  n_layers         : {cfg.n_layers}")
    print(f"  n_heads          : {cfg.n_heads}")
    print(f"  n_kv_heads       : {cfg.n_kv_heads}")
    print(f"  head_dim         : {cfg.head_dim}")
    print(f"  gqa_ratio        : {cfg.gqa_ratio}:1")
    print(f"  ffn_intermediate : {cfg.ffn_intermediate}")
    print(f"  max_seq_len      : {cfg.max_seq_len}")
    print(f"  vocab_size       : {cfg.vocab_size}")
    print(f"  dropout          : {cfg.dropout}")
    print(f"  bias             : {cfg.bias}")
    print(f"  tie_weights      : {cfg.tie_weights}")
    print("-" * 45)
    print(f"  Estimated params : {params / 1e6:.2f}M")
    print("=" * 45)