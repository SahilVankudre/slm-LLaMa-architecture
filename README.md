# SLM - Small Language Model with Agentic Capabilities

A production-grade small language model (~15M parameters) built from scratch with modern transformer architecture and tool-use capabilities.

## Project Overview

This project implements a complete small language model training and deployment pipeline:

- **Modern Architecture**: LLaMA-style transformer with RoPE, GQA, RMSNorm, and SwiGLU
- **Efficient Inference**: KV caching for fast autoregressive generation
- **Agentic System**: ReAct loop with tool use (calculator, memory, clock, search)
- **Production-Ready**: Mixed precision training, checkpointing, resumable training

### Key Features

- **~15M parameters** - Fast training and inference on consumer hardware
- **Grouped Query Attention (GQA)** - 3× memory savings vs standard attention
- **RoPE positional embeddings** - Better length generalization
- **SwiGLU activation** - Superior to ReLU for transformers
- **Tool use** - Calculator, memory, clock, and extensible tool registry
- **Flexible sampling** - Greedy, temperature, top-k, top-p, nucleus
- **Complete training pipeline** - From raw data to deployed model

## Architecture

```
Model Size: ~15.7M parameters
├── Embedding: 3.1M (vocab=8192, d_model=384)
├── 8× Transformer Blocks: 12.4M
│   ├── RMSNorm (pre-attention)
│   ├── Grouped Query Attention (6 Q heads, 2 KV heads)
│   │   └── RoPE positional encoding
│   ├── RMSNorm (pre-FFN)
│   └── SwiGLU FFN (intermediate=1024)
└── Output: Tied with embedding

Training Dataset: TinyStories (~2.1M short stories)
Context Window: 512 tokens
```

## Quick Start

### Installation

```bash
# Clone repository
git clone <your-repo-url>
cd slm

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Quick test (100 steps, small dataset)
python scripts/train.py --test

# Full training
python scripts/train.py \
    --batch_size 32 \
    --max_steps 20000 \
    --learning_rate 3e-4 \
    --use_amp

# Resume from checkpoint
python scripts/train.py --resume_from checkpoints/checkpoint_step_10000.pt
```

### Inference

```bash
# Interactive chat (normal mode)
python scripts/chat.py --checkpoint checkpoints/best_model.pt --mode normal

# Agent mode (with tools)
python scripts/chat.py --checkpoint checkpoints/best_model.pt --mode agent
```

## Project Structure

```
slm/
├── src/
│   ├── model/
│   │   ├── config.py           # Model configuration
│   │   ├── normalization.py    # RMSNorm
│   │   ├── rope.py             # Rotary positional embeddings
│   │   ├── attention.py        # Grouped Query Attention + KV cache
│   │   ├── ffn.py              # SwiGLU feed-forward
│   │   ├── block.py            # Transformer block
│   │   └── slm.py              # Top-level model
│   │
│   ├── tokenizer/
│   │   └── tokenizer.py        # BPE tokenizer (tiktoken wrapper)
│   │
│   ├── training/
│   │   ├── dataset.py          # TinyStories data pipeline
│   │   ├── trainer.py          # Training loop
│   │   └── scheduler.py        # Cosine LR + warmup
│   │
│   ├── inference/
│   │   ├── sampler.py          # Sampling strategies
│   │   └── generate.py         # Generation interface
│   │
│   └── agent/
│       ├── tools.py            # Tool registry + implementations
│       ├── parser.py           # Output parser for tool calls
│       └── agent.py            # ReAct loop
│
├── scripts/
│   ├── train.py                # Training entry point
│   └── chat.py                 # Interactive chat CLI
│
├── requirements.txt
└── README.md
```

## Usage Examples

### Python API

```python
from src.model.slm import SLM
from src.model.config import ModelConfig
from src.tokenizer.tokenizer import Tokenizer
from src.inference.generate import Generator

# Load model
config = ModelConfig()
model = SLM(config)
model.load_state_dict(torch.load("checkpoints/best_model.pt"))

# Create generator
tokenizer = Tokenizer()
generator = Generator(model, tokenizer, device="cuda")

# Generate text
output = generator.generate(
    "Once upon a time",
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.9
)
print(output)
```

### Agent System

```python
from src.agent.agent import Agent

# Create agent
agent = Agent(generator, verbose=True)

# Use tools
response = agent.run("What is 15 times 7?")
# Model will use calculator tool and respond with "105"

# Multi-turn conversation
agent.chat("Remember my name is Alice")  # Uses memory tool
agent.chat("What time is it?")           # Uses clock tool
agent.chat("What's my name?")            # Retrieves from memory
```

### Chat Commands

When running `scripts/chat.py`:

```
/help       - Show available commands
/reset      - Clear conversation history
/agent      - Switch to agent mode
/normal     - Switch to normal mode
/preset X   - Use sampling preset (greedy, creative, balanced, focused, precise)
/verbose    - Toggle debug output
/exit       - Exit chat
```

## Training Configuration

### Model Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 384 | Hidden dimension |
| `n_layers` | 8 | Transformer blocks |
| `n_heads` | 6 | Query attention heads |
| `n_kv_heads` | 2 | Key/value heads (GQA) |
| `max_seq_len` | 512 | Context window |
| `vocab_size` | 50265 | Tokenizer vocabulary |

### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 32 | Batch size |
| `gradient_accumulation_steps` | 4 | Gradient accumulation |
| `max_steps` | 20000 | Training steps |
| `learning_rate` | 3e-4 | Peak learning rate |
| `warmup_steps` | 2000 | LR warmup steps |
| `weight_decay` | 0.1 | AdamW weight decay |

## Extending the Project

### Adding Custom Tools

```python
from src.agent.tools import Tool

class CustomTool(Tool):
    def __init__(self):
        super().__init__()
        self.name = "my_tool"
        self.description = "Does something useful"
        self.parameters = {"arg1": "Description of arg1"}
    
    def execute(self, arg1: str) -> str:
        # Your tool logic here
        return f"Processed: {arg1}"

# Register with agent
from src.agent.tools import ToolRegistry
registry = ToolRegistry()
registry.register(CustomTool())
```

### Custom Sampling Strategies

```python
from src.inference.sampler import Sampler

# Create custom sampler
def my_sample(logits, param=1.0):
    # Your sampling logic
    return sampled_tokens

# Or use built-in
output = generator.generate(
    prompt,
    temperature=0.7,
    top_k=50,
    top_p=0.9
)
```

## Performance

Training performance on A100 GPU:

- **Throughput**: ~50K tokens/sec
- **Memory**: ~8GB VRAM (with mixed precision)
- **Training time**: ~4 hours for 20K steps
- **Inference**: ~100 tokens/sec (with KV cache)

## License

This project is built for educational and portfolio purposes.

## Acknowledgments

- Architecture inspired by LLaMA (Meta AI)
- Training data: TinyStories (Eldan & Li, Microsoft Research)
- Built from scratch following modern best practices

## Contact

Sahil Vankudre

---

**Note**: This is a learning project demonstrating deep understanding of transformer architecture, training pipelines, and agentic AI systems. The model is small by design for fast iteration and educational purposes.
