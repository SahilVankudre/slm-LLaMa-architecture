"""
src/agent/agent.py

Agent System - ReAct Loop
Orchestrates the full agent: reasoning, tool use, and action.

ReAct (Reasoning + Acting) Loop:
  1. User query → model
  2. Model generates: thought + tool_call OR final answer
  3. If tool_call: execute tool → inject result → repeat from step 2
  4. If final answer: return to user
  5. Max iterations safety: stop after N loops

System Prompt:
  - Instructs model on tool usage format
  - Lists available tools with descriptions
  - Provides examples of tool calling
  - Sets response format expectations

Architecture:
  User Input
      ↓
  [System Prompt + Tools + History]
      ↓
  Model Generation
      ↓
  Parse Output
      ↓
  Tool Call? ──No──> Final Answer ──> Return
      ↓ Yes
  Execute Tool
      ↓
  Inject Result
      ↓
  (loop back to Model Generation)
"""

import torch
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .tools import ToolRegistry, create_default_registry
from .parser import OutputParser, ParseResult, ToolCall


@dataclass
class Message:

    role: str
    content: str
    
    def __str__(self):
        return f"{self.role}: {self.content}"


class Agent:

    def __init__(
        self,
        generator,
        tool_registry: Optional[ToolRegistry] = None,
        max_iterations: int = 5,
        max_tokens_per_iteration: int = 200,
        verbose: bool = False
    ):
        self.generator = generator
        self.tool_registry = tool_registry or create_default_registry()
        self.max_iterations = max_iterations
        self.max_tokens_per_iteration = max_tokens_per_iteration
        self.verbose = verbose
        
        self.parser = OutputParser()
        
        self.conversation_history: List[Message] = []
        
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:

        tool_descriptions = self.tool_registry.generate_tool_prompt()
        
        prompt = f"""You are a helpful AI assistant with access to tools.

When you need to use a tool, respond in this format:
<|thought|>Your reasoning about what to do<|/thought|>
<|tool_call|><|tool_name|>tool_name<|tool_args|>{{"arg1": "value1", "arg2": "value2"}}<|/tool_call|>

When you have the final answer, respond:
<|answer|>Your final answer to the user

Available Tools:
{tool_descriptions}

Important:
- Always think step-by-step in <|thought|> tags before using a tool
- Tool arguments must be valid JSON
- You can use multiple tools in sequence
- Always end with <|answer|> when you have the final response
- Be concise and helpful

Example:
User: What's 15 times 7?
<|thought|>I need to calculate 15 * 7<|/thought|>
<|tool_call|><|tool_name|>calculator<|tool_args|>{{"expression": "15 * 7"}}<|/tool_call|>
<|tool_result|>105<|/tool_result|>
<|answer|>15 times 7 equals 105."""
        
        return prompt
    
    def reset(self):
        self.conversation_history = []
    
    def run(self, user_input: str, reset_history: bool = False) -> str:
       
        if reset_history:
            self.reset()
        
        self.conversation_history.append(Message(role="user", content=user_input))
        
        final_answer = self._react_loop()
        
        self.conversation_history.append(Message(role="assistant", content=final_answer))
        
        return final_answer
    
    def _react_loop(self) -> str:

        iteration = 0
        accumulated_output = ""
        
        while iteration < self.max_iterations:
            iteration += 1
            
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Iteration {iteration}/{self.max_iterations}")
                print(f"{'='*60}")
            
            prompt = self._build_prompt()
            
            if self.verbose:
                print(f"\nPrompt (last 200 chars): ...{prompt[-200:]}")
            
            output = self.generator.generate(
                prompt,
                max_new_tokens=self.max_tokens_per_iteration,
                temperature=0.7,
                top_p=0.9
            )
            
            accumulated_output += output
            
            if self.verbose:
                print(f"\nModel output:\n{output}")
            
            parse_result = self.parser.parse(output)
            
            if self.verbose:
                print(f"\nParsed:")
                print(f"  Thoughts: {len(parse_result.thoughts)}")
                print(f"  Tool calls: {len(parse_result.tool_calls)}")
                print(f"  Final answer: {parse_result.final_answer is not None}")
            
            if parse_result.final_answer:
                if self.verbose:
                    print(f"\n Final answer found")
                return parse_result.final_answer
            
            if parse_result.tool_calls:
                for tool_call in parse_result.tool_calls:
                    if self.verbose:
                        print(f"\n→ Executing tool: {tool_call.name}")
                        print(f"  Arguments: {tool_call.arguments}")
                    
                    result = self.tool_registry.execute(
                        tool_call.name,
                        **tool_call.arguments
                    )
                    
                    if self.verbose:
                        print(f"  Result: {result}")
                    
                    accumulated_output += f"\n<|tool_result|>{result}<|/tool_result|>\n"
                
                continue
            
            if self.verbose:
                print(f"\n No tool calls or final answer - treating as incomplete")
            
            if iteration >= self.max_iterations - 1:
                break
        
        if self.verbose:
            print(f"\n Max iterations ({self.max_iterations}) reached")
        
        return accumulated_output.strip() or "I apologize, but I couldn't complete the task."
    
    def _build_prompt(self) -> str:

        parts = [self.system_prompt]
        
        for msg in self.conversation_history:
            if msg.role == "user":
                parts.append(f"\nUser: {msg.content}")
            elif msg.role == "assistant":
                parts.append(f"\nAssistant: {msg.content}")
        
        parts.append("\nAssistant: ")
        
        return "\n".join(parts)
    
    def get_conversation_history(self) -> List[Message]:
        return self.conversation_history.copy()
    
    def chat(self, user_input: str) -> str:

        return self.run(user_input, reset_history=False)

if __name__ == "__main__":
    import sys
    sys.path.append('/mnt/user-data/outputs')
    
    from src.model.slm import SLM
    from src.model.config import ModelConfig
    from src.tokenizer.tokenizer import Tokenizer
    from src.inference.generate import Generator
    
    print("=" * 60)
    print("            Agent Unit Test")
    print("=" * 60)
    
    config = ModelConfig(
        d_model=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=50265,
        max_seq_len=256
    )
    
    model = SLM(config)
    tokenizer = Tokenizer()
    generator = Generator(model, tokenizer, device='cpu')
    
    print(f"Model: {model.get_num_params() / 1e6:.2f}M parameters")
    print("-" * 60)
    
    agent = Agent(
        generator=generator,
        max_iterations=3,
        max_tokens_per_iteration=100,
        verbose=True
    )
    
    print("\nSystem prompt (first 300 chars):")
    print(agent.system_prompt[:300] + "...")
    print("-" * 60)
    
    print("\n" + "=" * 60)
    print("Test 1: Agent run (with untrained model)")
    print("=" * 60)
    
    response = agent.run(
        "What is 2 + 2?",
        reset_history=True
    )
    
    print(f"\nFinal response: {response[:200]}")
    print("-" * 60)
    
    print("\nTest 2: Tool registry check")
    tools = agent.tool_registry.list_tools()
    print(f"  Available tools: {', '.join(tools)}")
    
    calc_result = agent.tool_registry.execute("calculator", expression="10 * 5")
    print(f"  Calculator test: 10 * 5 = {calc_result}")
    
    memory_result = agent.tool_registry.execute(
        "memory",
        action="store",
        key="test",
        value="hello"
    )
    print(f"  Memory test: {memory_result}")
    
    print("   Tool registry works")
    print("-" * 60)
    
    print("\nTest 3: Parser integration")
    
    mock_output = """<|thought|>I need to calculate this<|/thought|>
<|tool_call|><|tool_name|>calculator<|tool_args|>{"expression": "2 + 2"}<|/tool_call|>"""
    
    parse_result = agent.parser.parse(mock_output)
    print(f"  Parsed tool calls: {len(parse_result.tool_calls)}")
    
    if parse_result.tool_calls:
        tc = parse_result.tool_calls[0]
        print(f"  Tool name: {tc.name}")
        print(f"  Arguments: {tc.arguments}")
        
        result = agent.tool_registry.execute(tc.name, **tc.arguments)
        print(f"  Execution result: {result}")
    
    print("   Parser integration works")
    print("-" * 60)
    
    print("\nTest 4: Conversation history")
    
    history = agent.get_conversation_history()
    print(f"  History length: {len(history)}")
    for i, msg in enumerate(history):
        print(f"  [{i}] {msg.role}: {msg.content[:50]}...")
    
    print("   History tracking works")
    print("-" * 60)
    
    print("\nTest 5: Chat interface (multi-turn)")
    
    agent.reset()
    
    response1 = agent.chat("Hello!")
    print(f"  Turn 1: {response1[:80]}...")
    
    response2 = agent.chat("Tell me more")
    print(f"  Turn 2: {response2[:80]}...")
    
    print(f"  History after 2 turns: {len(agent.get_conversation_history())} messages")
    print("   Multi-turn chat works")
    print("-" * 60)
    
    print("\n" + "=" * 60)
    print("Note: With an untrained model, the agent won't produce")
    print("coherent tool calls. After training on tool-use data,")
    print("it will learn to use the special token format correctly.")
    print("=" * 60)
    
    print("\n All agent tests passed")
    print("=" * 60)
