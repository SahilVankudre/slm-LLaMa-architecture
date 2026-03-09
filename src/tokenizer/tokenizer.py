"""
src/tokenizer/tokenizer.py

Tokenizer for SLM
Uses tiktoken (GPT-2 BPE) with custom special tokens for agent functionality.

Special tokens for agent system:
  <|thought|>     - Model's internal reasoning
  <|tool_call|>   - Start of tool invocation
  <|tool_name|>   - Tool name marker
  <|tool_args|>   - Tool arguments marker
  <|/tool_call|>  - End of tool invocation
  <|tool_result|> - Tool execution result
  <|answer|>      - Final answer to user

Note: We're using tiktoken's pre-trained GPT-2 tokenizer (vocab_size=50257)
rather than training our own BPE. For production, you might train a custom
tokenizer on your domain-specific corpus.
"""

import tiktoken
from typing import List, Union


class Tokenizer:

    SPECIAL_TOKENS = {
        "<|thought|>": 50257,
        "<|/thought|>": 50258,
        "<|tool_call|>": 50259,
        "<|tool_name|>": 50260,
        "<|tool_args|>": 50261,
        "<|/tool_call|>": 50262,
        "<|tool_result|>": 50263,
        "<|answer|>": 50264,
        "<|endoftext|>": 50256,  
    }
    
    def __init__(self, base_tokenizer: str = "gpt2"):

        self.base_tokenizer = tiktoken.get_encoding(base_tokenizer)

        self.base_vocab_size = self.base_tokenizer.n_vocab

        self._vocab_size = self.base_vocab_size + len(self.SPECIAL_TOKENS) - 1  

        self.special_token_ids = self.SPECIAL_TOKENS.copy()
        self.id_to_special_token = {v: k for k, v in self.SPECIAL_TOKENS.items()}
    
    @property
    def vocab_size(self) -> int:
        return self._vocab_size
    
    @property
    def eot_token(self) -> str:
        return "<|endoftext|>"
    
    @property
    def eot_token_id(self) -> int:
        return self.SPECIAL_TOKENS["<|endoftext|>"]
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:

        if not add_special_tokens:
            return self.base_tokenizer.encode(text)

        tokens = []
        current_text = text
        
        while current_text:
            earliest_pos = len(current_text)
            earliest_token = None
            
            for special_token in self.SPECIAL_TOKENS.keys():
                pos = current_text.find(special_token)
                if pos != -1 and pos < earliest_pos:
                    earliest_pos = pos
                    earliest_token = special_token
            
            if earliest_token is None:
                tokens.extend(self.base_tokenizer.encode(current_text))
                break
            
            if earliest_pos > 0:
                tokens.extend(self.base_tokenizer.encode(current_text[:earliest_pos]))
            
            tokens.append(self.SPECIAL_TOKENS[earliest_token])
            
            current_text = current_text[earliest_pos + len(earliest_token):]
        
        return tokens
    
    def decode(self, token_ids: Union[List[int], List[List[int]]], skip_special_tokens: bool = False) -> Union[str, List[str]]:

        if token_ids and isinstance(token_ids[0], list):
            return [self.decode(ids, skip_special_tokens) for ids in token_ids]
        
        text_parts = []
        current_chunk = []
        
        for token_id in token_ids:
            if token_id in self.id_to_special_token:
                if current_chunk:
                    text_parts.append(self.base_tokenizer.decode(current_chunk))
                    current_chunk = []
                
                if not skip_special_tokens:
                    text_parts.append(self.id_to_special_token[token_id])
            else:
                current_chunk.append(token_id)
        
        if current_chunk:
            text_parts.append(self.base_tokenizer.decode(current_chunk))
        
        return "".join(text_parts)
    
    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        return [self.encode(text) for text in texts]
    
    def get_special_token_id(self, token: str) -> int:
        return self.SPECIAL_TOKENS[token]

if __name__ == "__main__":
    print("=" * 60)
    print("          Tokenizer Unit Test")
    print("=" * 60)

    tokenizer = Tokenizer()
    
    print(f"Base vocab size   : {tokenizer.base_vocab_size}")
    print(f"Total vocab size  : {tokenizer.vocab_size}")
    print(f"Special tokens    : {len(tokenizer.SPECIAL_TOKENS)}")
    print(f"EOT token         : '{tokenizer.eot_token}'")
    print(f"EOT token ID      : {tokenizer.eot_token_id}")
    print("-" * 60)

    print("Test 1: Basic encoding/decoding")
    text = "Hello, world! This is a test."
    tokens = tokenizer.encode(text, add_special_tokens=False)
    decoded = tokenizer.decode(tokens)
    
    print(f"  Original  : '{text}'")
    print(f"  Tokens    : {tokens[:10]}... ({len(tokens)} total)")
    print(f"  Decoded   : '{decoded}'")
    assert decoded == text, "Decode mismatch"
    print(f"  Basic encode/decode works")
    print("-" * 60)

    print("Test 2: Special token handling")
    text_with_special = "User asks: <|thought|>I should use the calculator<|/thought|><|tool_call|>calculator<|/tool_call|>"
    tokens = tokenizer.encode(text_with_special)
    decoded = tokenizer.decode(tokens)
    
    print(f"  Original  : '{text_with_special[:50]}...'")
    print(f"  Tokens    : {tokens}")
    print(f"  Decoded   : '{decoded[:50]}...'")
    assert decoded == text_with_special, "Special token handling failed"
    print(f"  Special tokens work")
    print("-" * 60)
    
    print("Test 3: Skip special tokens")
    decoded_no_special = tokenizer.decode(tokens, skip_special_tokens=True)
    print(f"  With special    : '{decoded[:40]}...'")
    print(f"  Without special : '{decoded_no_special}'")
    assert "<|thought|>" not in decoded_no_special
    print(f"  Skip special tokens works")
    print("-" * 60)
    
    print("Test 4: Batch encoding")
    texts = [
        "First sentence.",
        "Second sentence with more words.",
        "Third one."
    ]
    batch_tokens = tokenizer.encode_batch(texts)
    batch_decoded = tokenizer.decode(batch_tokens)
    
    print(f"  Input batch size  : {len(texts)}")
    print(f"  Token lengths     : {[len(t) for t in batch_tokens]}")
    print(f"  Decoded batch     : {batch_decoded}")
    assert batch_decoded == texts
    print(f"  Batch encoding works")
    print("-" * 60)
    
    print("Test 5: Special token ID lookup")
    for token_str in ["<|thought|>", "<|tool_call|>", "<|answer|>"]:
        token_id = tokenizer.get_special_token_id(token_str)
        print(f"  {token_str:20} : {token_id}")
    print(f"  Special token lookup works")
    print("-" * 60)
    
    print("All tokenizer tests passed")
    print("=" * 60)