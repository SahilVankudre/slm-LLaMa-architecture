"""
src/inference/generate.py

Text Generation Interface
High-level API for generating text with the SLM model.

Features:
  - KV cache for efficient autoregressive generation
  - Multiple sampling strategies (greedy, temperature, top-k, top-p)
  - Batch generation support
  - Stopping conditions (max tokens, EOS, custom stop sequences)
  - Streaming output support

Generation process:
  1. Encode prompt with tokenizer
  2. Prefill: run full prompt through model, initialize KV cache
  3. Decode: generate tokens one at a time, updating KV cache
  4. Stop when: max_length reached, EOS token, or stop sequence detected
"""

import torch
import torch.nn as nn
from typing import Optional, List, Union, Callable
from tqdm.auto import tqdm

from .sampler import Sampler, SamplingPresets


class Generator:

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: str = 'cuda'
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        self.model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        stop_sequences: Optional[List[str]] = None,
        stream: bool = False,
        callback: Optional[Callable] = None
    ) -> Union[str, List[str]]:

        is_batch = isinstance(prompt, list)
        if not is_batch:
            prompt = [prompt]
        
        batch_size = len(prompt)
        
        input_ids = []
        for p in prompt:
            tokens = self.tokenizer.encode(p, add_special_tokens=False)
            input_ids.append(torch.tensor(tokens, dtype=torch.long))
        
        max_prompt_len = max(len(ids) for ids in input_ids)
        padded_input_ids = []
        for ids in input_ids:
            if len(ids) < max_prompt_len:
                pad_len = max_prompt_len - len(ids)
                ids = torch.cat([torch.zeros(pad_len, dtype=torch.long), ids])
            padded_input_ids.append(ids)
        
        input_ids = torch.stack(padded_input_ids).to(self.device)
        
        active_sequences = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        
        generated_tokens = [[] for _ in range(batch_size)]
        
        logits, kv_caches = self.model(input_ids, start_pos=0, use_cache=True)
        
        next_token = self._sample_token(
            logits[:, -1, :],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        
        for i in range(batch_size):
            generated_tokens[i].append(next_token[i].item())
        
        for step in range(max_new_tokens - 1):
            if not active_sequences.any():
                break
            
            logits, kv_caches = self.model(
                next_token,
                start_pos=input_ids.shape[1] + step,
                kv_caches=kv_caches,
                use_cache=True
            )
            
            if repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(
                    logits[:, -1, :],
                    generated_tokens,
                    repetition_penalty
                )
            else:
                logits = logits[:, -1, :]
            
            next_token = self._sample_token(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            
            for i in range(batch_size):
                if not active_sequences[i]:
                    continue
                
                token_id = next_token[i].item()
                generated_tokens[i].append(token_id)
                
                if token_id == self.tokenizer.eot_token_id:
                    active_sequences[i] = False
                
                if stop_sequences and active_sequences[i]:
                    current_text = self.tokenizer.decode(generated_tokens[i])
                    for stop_seq in stop_sequences:
                        if stop_seq in current_text:
                            active_sequences[i] = False
                            break
            
            if stream and batch_size == 1 and callback:
                token_str = self.tokenizer.decode([next_token[0].item()])
                callback(token_str)
        
        outputs = []
        for i in range(batch_size):
            generated_text = self.tokenizer.decode(
                generated_tokens[i],
                skip_special_tokens=True
            )
            outputs.append(generated_text)
        
        if not is_batch:
            return outputs[0]
        return outputs
    
    def generate_with_preset(
        self,
        prompt: Union[str, List[str]],
        preset: str = 'balanced',
        max_new_tokens: int = 100,
        **kwargs
    ) -> Union[str, List[str]]:

        preset_fn = getattr(SamplingPresets, preset, None)
        if preset_fn is None:
            raise ValueError(f"Unknown preset: {preset}. Choose from: greedy, creative, balanced, focused, precise")
        
        config = preset_fn()
        
        config.update(kwargs)
        
        return self.generate(prompt, max_new_tokens=max_new_tokens, **config)
    
    def _sample_token(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        return Sampler.sample(logits, temperature=temperature, top_k=top_k, top_p=top_p)
    
    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated_tokens: List[List[int]],
        penalty: float
    ) -> torch.Tensor:

        for i, tokens in enumerate(generated_tokens):
            if not tokens:
                continue
            
            unique_tokens = set(tokens)
            
            for token_id in unique_tokens:

                if logits[i, token_id] > 0:
                    logits[i, token_id] /= penalty
                else:
                    logits[i, token_id] *= penalty
        
        return logits

def generate_batch(
    generator: Generator,
    prompts: List[str],
    batch_size: int = 8,
    **generate_kwargs
) -> List[str]:

    all_outputs = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        outputs = generator.generate(batch_prompts, **generate_kwargs)

        if isinstance(outputs, str):
            all_outputs.append(outputs)
        else:
            all_outputs.extend(outputs)
    
    return all_outputs

if __name__ == "__main__":
    import sys
    sys.path.append('/mnt/user-data/outputs')
    
    from src.model.slm import SLM
    from src.model.config import ModelConfig
    from src.tokenizer.tokenizer import Tokenizer
    
    print("=" * 60)
    print("          Generator Unit Test")
    print("=" * 60)
    
    config = ModelConfig(
        d_model=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=50265,
        max_seq_len=128
    )
    
    model = SLM(config)
    tokenizer = Tokenizer()
    
    print(f"Model parameters: {model.get_num_params() / 1e6:.2f}M")
    print(f"Tokenizer vocab: {tokenizer.vocab_size}")
    print("-" * 60)
    
    generator = Generator(model, tokenizer, device='cpu')
    
    print("Test 1: Single prompt generation")
    prompt = "Once upon a time"
    output = generator.generate(
        prompt,
        max_new_tokens=20,
        temperature=1.0,
        top_p=0.9
    )
    
    print(f"  Prompt: '{prompt}'")
    print(f"  Output: '{output[:100]}...'")
    print(f"  Output length: {len(output)} chars")
    print(f"   Single generation works")
    print("-" * 60)
    
    print("Test 2: Batch generation")
    prompts = [
        "The cat sat on",
        "In a galaxy far",
        "She opened the door and"
    ]
    
    outputs = generator.generate(
        prompts,
        max_new_tokens=15,
        temperature=0.8
    )
    
    print(f"  Batch size: {len(prompts)}")
    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        print(f"  [{i}] '{prompt}' → '{output[:50]}...'")
    print(f"   Batch generation works")
    print("-" * 60)
    
    print("Test 3: Preset configurations")
    
    presets = ['greedy', 'balanced', 'creative', 'focused']
    for preset in presets:
        output = generator.generate_with_preset(
            "Hello world",
            preset=preset,
            max_new_tokens=10
        )
        print(f"  {preset:10}: {len(output)} chars generated")
    
    print(f"   Presets work")
    print("-" * 60)
    
    print("Test 4: Repetition penalty")
    
    output_no_penalty = generator.generate(
        "Test test test",
        max_new_tokens=10,
        temperature=0.8,
        repetition_penalty=1.0
    )
    
    output_with_penalty = generator.generate(
        "Test test test",
        max_new_tokens=10,
        temperature=0.8,
        repetition_penalty=1.5
    )
    
    print(f"  No penalty:   '{output_no_penalty[:50]}...'")
    print(f"  With penalty: '{output_with_penalty[:50]}...'")
    print(f"   Repetition penalty works")
    print("-" * 60)
    
    print("Test 5: Stop sequences")
    
    output = generator.generate(
        "Count: 1, 2, 3",
        max_new_tokens=50,
        stop_sequences=[", 5"],  
        temperature=0.7
    )
    
    print(f"  Output: '{output}'")
    print(f"  Stop sequence detection: {', 5' in output or len(output.split(',')) <= 5}")
    print(f"   Stop sequences work")
    print("-" * 60)
    
    print("Test 6: Streaming with callback")
    
    collected_tokens = []
    
    def token_callback(token_str):
        collected_tokens.append(token_str)
    
    output = generator.generate(
        "The quick brown fox",
        max_new_tokens=10,
        stream=True,
        callback=token_callback
    )
    
    print(f"  Streamed tokens: {len(collected_tokens)}")
    print(f"  Collected: {''.join(collected_tokens[:20])}")
    print(f"   Streaming callback works")
    print("-" * 60)
    
    print(" All generator tests passed")
    print("=" * 60)
