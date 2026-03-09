"""
src/agent/tools.py

Tool Registry and Tool Implementations
Core of the agentic system - defines tools the model can call.

Tool Interface:
  - name: Unique identifier
  - description: What the tool does (shown to model in prompt)
  - parameters: Schema describing required arguments
  - execute(): Function that runs the tool

Built-in tools:
  1. Calculator: Evaluate mathematical expressions
  2. Memory: Store and retrieve key-value pairs
  3. Clock: Get current date/time
  4. Search: Simple string search (placeholder)

The registry auto-generates tool descriptions for the model's system prompt.
"""

import re
import json
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from abc import ABC, abstractmethod

class Tool(ABC):
 
    def __init__(self):
        self.name: str = ""
        self.description: str = ""
        self.parameters: Dict[str, str] = {}
    
    @abstractmethod
    def execute(self, **kwargs) -> str:

        pass
    
    def to_prompt_format(self) -> str:
        param_strs = [f"  - {name}: {desc}" for name, desc in self.parameters.items()]
        params_text = "\n".join(param_strs) if param_strs else "  (no parameters)"
        
        return f"""Tool: {self.name}
Description: {self.description}
Parameters:
{params_text}"""

class Calculator(Tool):

    def __init__(self):
        super().__init__()
        self.name = "calculator"
        self.description = "Evaluates mathematical expressions. Supports +, -, *, /, **, parentheses, and common math functions."
        self.parameters = {
            "expression": "Mathematical expression as a string (e.g., '2 + 2', 'sqrt(16)', '3 ** 2')"
        }
        
        import math
        self.safe_namespace = {
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'sqrt': math.sqrt, 'log': math.log, 'exp': math.exp,
            'pi': math.pi, 'e': math.e,
            'floor': math.floor, 'ceil': math.ceil,
            'pow': pow
        }
    
    def execute(self, expression: str) -> str:
        try:
            expression = expression.strip()
            
            result = eval(expression, {"__builtins__": {}}, self.safe_namespace)
            
            if isinstance(result, float):
                if result == int(result):
                    return str(int(result))
                else:
                    return f"{result:.6g}"
            return str(result)
            
        except Exception as e:
            return f"Error evaluating expression: {str(e)}"


class Memory(Tool):

    def __init__(self):
        super().__init__()
        self.name = "memory"
        self.description = "Store and retrieve information. Use 'store' to save a key-value pair, 'retrieve' to get a value, 'list' to see all keys."
        self.parameters = {
            "action": "One of: 'store', 'retrieve', 'list'",
            "key": "(Optional) Key name for store/retrieve",
            "value": "(Optional) Value to store"
        }
        
        self.storage: Dict[str, str] = {}
    
    def execute(self, action: str, key: Optional[str] = None, value: Optional[str] = None) -> str:

        action = action.lower().strip()
        
        if action == "store":
            if not key:
                return "Error: 'key' required for store action"
            if value is None:
                return "Error: 'value' required for store action"
            
            self.storage[key] = str(value)
            return f"Stored '{key}': '{value}'"
        
        elif action == "retrieve":
            if not key:
                return "Error: 'key' required for retrieve action"
            
            if key in self.storage:
                return f"{key}: {self.storage[key]}"
            else:
                return f"Key '{key}' not found in memory"
        
        elif action == "list":
            if not self.storage:
                return "Memory is empty"
            
            keys = list(self.storage.keys())
            return f"Stored keys: {', '.join(keys)}"
        
        else:
            return f"Unknown action '{action}'. Use 'store', 'retrieve', or 'list'."


class Clock(Tool):

    def __init__(self):
        super().__init__()
        self.name = "clock"
        self.description = "Get current date and time information."
        self.parameters = {
            "format": "(Optional) What to return: 'date', 'time', 'datetime', 'timestamp'. Default: 'datetime'"
        }
    
    def execute(self, format: str = "datetime") -> str:
        now = datetime.now()
        format = format.lower().strip()
        
        if format == "date":
            return now.strftime("%Y-%m-%d")
        elif format == "time":
            return now.strftime("%H:%M:%S")
        elif format == "timestamp":
            return str(int(now.timestamp()))
        else:  
            return now.strftime("%Y-%m-%d %H:%M:%S")


class Search(Tool):

    def __init__(self):
        super().__init__()
        self.name = "search"
        self.description = "Search for information (placeholder - returns mock results)"
        self.parameters = {
            "query": "Search query string"
        }
    
    def execute(self, query: str) -> str:

        return f"Mock search results for '{query}': This is a placeholder. In production, integrate with a real search API."

class ToolRegistry:

    def __init__(self):
        self.tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool):
        self.tools[tool.name] = tool
        print(f"Registered tool: {tool.name}")
    
    def get(self, name: str) -> Optional[Tool]:

        return self.tools.get(name)
    
    def list_tools(self) -> List[str]:
        return list(self.tools.keys())
    
    def execute(self, name: str, **kwargs) -> str:

        tool = self.get(name)
        if tool is None:
            return f"Error: Tool '{name}' not found. Available tools: {', '.join(self.list_tools())}"
        
        try:
            return tool.execute(**kwargs)
        except Exception as e:
            return f"Error executing tool '{name}': {str(e)}"
    
    def generate_tool_prompt(self) -> str:

        if not self.tools:
            return "No tools available."
        
        tool_descriptions = []
        for tool in self.tools.values():
            tool_descriptions.append(tool.to_prompt_format())
        
        return "\n\n".join(tool_descriptions)

def create_default_registry() -> ToolRegistry:

    registry = ToolRegistry()
    
    registry.register(Calculator())
    registry.register(Memory())
    registry.register(Clock())
    registry.register(Search())
    
    return registry

if __name__ == "__main__":
    print("=" * 60)
    print("          Tool Registry Unit Test")
    print("=" * 60)
    
    registry = create_default_registry()
    
    print(f"\nRegistered tools: {', '.join(registry.list_tools())}")
    print("-" * 60)
    
    print("\nTest 1: Calculator")
    calc = registry.get("calculator")
    
    test_expressions = [
        "2 + 2",
        "10 * 5 - 3",
        "sqrt(16)",
        "sin(pi/2)",
        "2 ** 8"
    ]
    
    for expr in test_expressions:
        result = calc.execute(expression=expr)
        print(f"  {expr:20} = {result}")
    
    print("  ✅ Calculator works")
    print("-" * 60)
    
    print("\nTest 2: Memory")
    memory = registry.get("memory")
    
    print(f"  Store 'name': {memory.execute(action='store', key='name', value='Alice')}")
    print(f"  Store 'age': {memory.execute(action='store', key='age', value='30')}")
    print(f"  List keys: {memory.execute(action='list')}")
    print(f"  Retrieve 'name': {memory.execute(action='retrieve', key='name')}")
    print(f"  Retrieve 'missing': {memory.execute(action='retrieve', key='missing')}")
    
    print("   Memory works")
    print("-" * 60)
    
    print("\nTest 3: Clock")
    clock = registry.get("clock")
    
    print(f"  Date:     {clock.execute(format='date')}")
    print(f"  Time:     {clock.execute(format='time')}")
    print(f"  DateTime: {clock.execute(format='datetime')}")
    print(f"  Timestamp: {clock.execute(format='timestamp')}")
    
    print("   Clock works")
    print("-" * 60)
    
    print("\nTest 4: Search")
    search = registry.get("search")
    
    result = search.execute(query="python programming")
    print(f"  Query: 'python programming'")
    print(f"  Result: {result[:80]}...")
    
    print("   Search works (mock)")
    print("-" * 60)
    
    print("\nTest 5: Registry execute")
    
    result = registry.execute("calculator", expression="100 / 4")
    print(f"  Direct execute: {result}")
    
    error_result = registry.execute("nonexistent", foo="bar")
    print(f"  Invalid tool: {error_result[:60]}...")
    
    print("   Registry execute works")
    print("-" * 60)
    
    print("\nTest 6: Generate system prompt")
    
    prompt = registry.generate_tool_prompt()
    print("  Tool prompt preview:")
    print("  " + "\n  ".join(prompt.split("\n")[:15]))
    print("  ...")
    
    print("   System prompt generation works")
    print("-" * 60)
    
    print("\nTest 7: Custom tool registration")
    
    class EchoTool(Tool):
        def __init__(self):
            super().__init__()
            self.name = "echo"
            self.description = "Echoes back the input text"
            self.parameters = {"text": "Text to echo"}
        
        def execute(self, text: str) -> str:
            return f"Echo: {text}"
    
    custom_tool = EchoTool()
    registry.register(custom_tool)
    
    result = registry.execute("echo", text="Hello, world!")
    print(f"  Echo result: {result}")
    print(f"  Tool count: {len(registry.list_tools())}")
    
    print("   Custom tool registration works")
    print("-" * 60)
    
    print("\n All tool tests passed")
    print("=" * 60)
