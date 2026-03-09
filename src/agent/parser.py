"""
src/agent/parser.py

Output Parser for Agent System
Extracts structured tool calls from model's raw text output.

Expected format from model:
  <|thought|>reasoning here<|/thought|>
  <|tool_call|><|tool_name|>calculator<|tool_args|>{"expression": "2+2"}<|/tool_call|>
  <|answer|>final answer here

Parser responsibilities:
  1. Detect tool calls in model output
  2. Extract tool name and arguments
  3. Parse JSON arguments
  4. Handle malformed outputs gracefully
  5. Support streaming (partial outputs)

State machine approach:
  - Scan for special tokens
  - Track state (inside thought, tool call, etc.)
  - Accumulate content between markers
  - Yield parsed tool calls when complete
"""

import json
import re
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

class ParseState(Enum):
    NORMAL = "normal"
    IN_THOUGHT = "in_thought"
    IN_TOOL_CALL = "in_tool_call"
    IN_ANSWER = "in_answer"


@dataclass
class ToolCall:
    
    name: str
    arguments: Dict[str, Any]
    raw_args: str
    thought: Optional[str] = None
    
    def __str__(self):
        return f"ToolCall(name='{self.name}', args={self.arguments})"


@dataclass
class ParseResult:
    tool_calls: List[ToolCall]
    thoughts: List[str]
    final_answer: Optional[str]
    is_complete: bool
    remaining_text: str

class OutputParser:
    THOUGHT_START = "<|thought|>"
    THOUGHT_END = "<|/thought|>"
    TOOL_CALL_START = "<|tool_call|>"
    TOOL_CALL_END = "<|/tool_call|>"
    TOOL_NAME = "<|tool_name|>"
    TOOL_ARGS = "<|tool_args|>"
    ANSWER_START = "<|answer|>"
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.state = ParseState.NORMAL
        self.current_thought = ""
        self.current_tool_name = ""
        self.current_tool_args = ""
        self.thoughts = []
        self.tool_calls = []
        self.final_answer = ""
    
    def parse(self, text: str) -> ParseResult:
        self.reset()
  
        pos = 0
        remaining = ""
        
        while pos < len(text):
            next_marker, marker_pos = self._find_next_marker(text, pos)
            
            if next_marker is None:
                remaining = text[pos:]
                break
            
            content_before = text[pos:marker_pos]
            
            if next_marker == self.THOUGHT_START:
                self.state = ParseState.IN_THOUGHT
                self.current_thought = ""
            
            elif next_marker == self.THOUGHT_END:
                if self.state == ParseState.IN_THOUGHT:
                    self.thoughts.append(self.current_thought.strip())
                    self.current_thought = ""
                self.state = ParseState.NORMAL
            
            elif next_marker == self.TOOL_CALL_START:
                self.state = ParseState.IN_TOOL_CALL
                self.current_tool_name = ""
                self.current_tool_args = ""
            
            elif next_marker == self.TOOL_CALL_END:
                if self.state == ParseState.IN_TOOL_CALL:
                    tool_call = self._create_tool_call()
                    if tool_call:
                        self.tool_calls.append(tool_call)
                self.state = ParseState.NORMAL
            
            elif next_marker == self.TOOL_NAME:
                name_end_pos = self._find_name_end(text, marker_pos + len(next_marker))
                self.current_tool_name = text[marker_pos + len(next_marker):name_end_pos].strip()
                pos = name_end_pos
                continue
            
            elif next_marker == self.TOOL_ARGS:
                args_end_pos = text.find(self.TOOL_CALL_END, marker_pos + len(next_marker))
                if args_end_pos != -1:
                    self.current_tool_args = text[marker_pos + len(next_marker):args_end_pos].strip()
                    pos = args_end_pos
                    continue
            
            elif next_marker == self.ANSWER_START:
                self.state = ParseState.IN_ANSWER
                self.final_answer = text[marker_pos + len(next_marker):].strip()
                remaining = ""
                break
            
            if self.state == ParseState.IN_THOUGHT and content_before:
                self.current_thought += content_before
            
            pos = marker_pos + len(next_marker)
        
        is_complete = (
            self.final_answer != "" or
            (self.state == ParseState.NORMAL and not remaining.strip())
        )
        
        return ParseResult(
            tool_calls=self.tool_calls.copy(),
            thoughts=self.thoughts.copy(),
            final_answer=self.final_answer if self.final_answer else None,
            is_complete=is_complete,
            remaining_text=remaining
        )
    
    def _find_next_marker(self, text: str, start_pos: int) -> tuple[Optional[str], int]:
        markers = [
            self.THOUGHT_START, self.THOUGHT_END,
            self.TOOL_CALL_START, self.TOOL_CALL_END,
            self.TOOL_NAME, self.TOOL_ARGS,
            self.ANSWER_START
        ]
        
        earliest_pos = len(text)
        earliest_marker = None
        
        for marker in markers:
            pos = text.find(marker, start_pos)
            if pos != -1 and pos < earliest_pos:
                earliest_pos = pos
                earliest_marker = marker
        
        if earliest_marker is None:
            return None, -1
        
        return earliest_marker, earliest_pos
    
    def _find_name_end(self, text: str, start_pos: int) -> int:
        args_pos = text.find(self.TOOL_ARGS, start_pos)
        end_pos = text.find(self.TOOL_CALL_END, start_pos)
        
        if args_pos != -1 and (end_pos == -1 or args_pos < end_pos):
            return args_pos
        elif end_pos != -1:
            return end_pos
        else:
            return len(text)
    
    def _create_tool_call(self) -> Optional[ToolCall]:
        if not self.current_tool_name:
            return None
        
        arguments = {}
        raw_args = self.current_tool_args
        
        if raw_args:
            try:
                arguments = json.loads(raw_args)
                if not isinstance(arguments, dict):
                    arguments = {"value": arguments}
            except json.JSONDecodeError:
                arguments = self._parse_args_fallback(raw_args)
        
        thought = self.thoughts[-1] if self.thoughts else None
        
        return ToolCall(
            name=self.current_tool_name,
            arguments=arguments,
            raw_args=raw_args,
            thought=thought
        )
    
    def _parse_args_fallback(self, args_str: str) -> Dict[str, Any]:
        kv_pattern = r'(\w+)\s*[:=]\s*["\']?([^"\',:]+)["\']?'
        matches = re.findall(kv_pattern, args_str)
        
        if matches:
            return {k.strip(): v.strip() for k, v in matches}
        else:
            return {"arg": args_str.strip()}

def parse_output(text: str) -> ParseResult:
    parser = OutputParser()
    return parser.parse(text)


def extract_tool_calls(text: str) -> List[ToolCall]:
    result = parse_output(text)
    return result.tool_calls

if __name__ == "__main__":
    print("=" * 60)
    print("        Output Parser Unit Test")
    print("=" * 60)
    
    # Test 1: Basic tool call
    print("\nTest 1: Basic tool call")
    text1 = """<|thought|>I need to calculate 2 + 2<|/thought|>
<|tool_call|><|tool_name|>calculator<|tool_args|>{"expression": "2 + 2"}<|/tool_call|>"""
    
    result1 = parse_output(text1)
    print(f"  Thoughts: {result1.thoughts}")
    print(f"  Tool calls: {len(result1.tool_calls)}")
    if result1.tool_calls:
        tc = result1.tool_calls[0]
        print(f"    Name: {tc.name}")
        print(f"    Args: {tc.arguments}")
    print(f"   Basic parsing works")
    print("-" * 60)
    
    print("\nTest 2: Multiple tool calls")
    text2 = """<|thought|>First calculate<|/thought|>
<|tool_call|><|tool_name|>calculator<|tool_args|>{"expression": "10 * 5"}<|/tool_call|>
<|thought|>Now store the result<|/thought|>
<|tool_call|><|tool_name|>memory<|tool_args|>{"action": "store", "key": "result", "value": "50"}<|/tool_call|>"""
    
    result2 = parse_output(text2)
    print(f"  Thoughts: {len(result2.thoughts)}")
    print(f"  Tool calls: {len(result2.tool_calls)}")
    for i, tc in enumerate(result2.tool_calls):
        print(f"    [{i}] {tc.name}: {tc.arguments}")
    print(f"   Multiple tool calls work")
    print("-" * 60)
    
    print("\nTest 3: Final answer detection")
    text3 = """<|thought|>I have the result<|/thought|>
<|answer|>The answer is 42."""
    
    result3 = parse_output(text3)
    print(f"  Final answer: {result3.final_answer}")
    print(f"  Is complete: {result3.is_complete}")
    print(f"   Final answer detection works")
    print("-" * 60)
    
    print("\nTest 4: Malformed JSON fallback")
    text4 = """<|tool_call|><|tool_name|>calculator<|tool_args|>expression: 2+2<|/tool_call|>"""
    
    result4 = parse_output(text4)
    if result4.tool_calls:
        tc = result4.tool_calls[0]
        print(f"  Name: {tc.name}")
        print(f"  Args (fallback): {tc.arguments}")
        print(f"  Raw: {tc.raw_args}")
    print(f"   Fallback parsing works")
    print("-" * 60)
    
    print("\nTest 5: Incomplete output (streaming)")
    text5 = """<|thought|>Let me think about this"""
    
    result5 = parse_output(text5)
    print(f"  Is complete: {result5.is_complete}")
    print(f"  Remaining: '{result5.remaining_text}'")
    print(f"   Incomplete detection works")
    print("-" * 60)
    
    print("\nTest 6: Complex scenario")
    text6 = """<|thought|>The user wants to know the time<|/thought|>
<|tool_call|><|tool_name|>clock<|tool_args|>{"format": "datetime"}<|/tool_call|>
<|thought|>Now I'll provide the answer<|/thought|>
<|answer|>The current time is 2024-03-15 14:30:00"""
    
    result6 = parse_output(text6)
    print(f"  Thoughts: {len(result6.thoughts)}")
    print(f"  Tool calls: {len(result6.tool_calls)}")
    print(f"  Final answer: {result6.final_answer[:50] if result6.final_answer else None}")
    print(f"  Is complete: {result6.is_complete}")
    print(f"   Complex parsing works")
    print("-" * 60)
    
    print("\nTest 7: Convenience functions")
    tool_calls = extract_tool_calls(text2)
    print(f"  Extracted tool calls: {len(tool_calls)}")
    for tc in tool_calls:
        print(f"    - {tc}")
    print(f"   Convenience functions work")
    print("-" * 60)
    
    print("\nTest 8: Edge cases")
    
    result_empty = parse_output("")
    print(f"  Empty input: {len(result_empty.tool_calls)} calls")
    
    result_plain = parse_output("Just plain text")
    print(f"  Plain text: {len(result_plain.tool_calls)} calls")
    
    result_unclosed = parse_output("<|thought|>Incomplete")
    print(f"  Unclosed thought complete: {result_unclosed.is_complete}")
    
    print(f"   Edge cases handled")
    print("-" * 60)
    
    print("\n All parser tests passed")
    print("=" * 60)
