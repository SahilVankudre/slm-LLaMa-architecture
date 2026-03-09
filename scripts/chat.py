"""
scripts/chat.py

Interactive Chat Interface
Chat with the trained SLM model with or without agent capabilities.

Usage:
    python scripts/chat.py --checkpoint PATH [--mode MODE]

Modes:
  - normal: Standard text generation (no tools)
  - agent: Full agent with tool use (ReAct loop)

Commands:
  /help       - Show this help
  /reset      - Clear conversation history
  /agent      - Switch to agent mode
  /normal     - Switch to normal mode
  /preset X   - Use sampling preset (greedy, creative, balanced, focused, precise)
  /verbose    - Toggle verbose output
  /exit       - Exit chat

Example session:
  You: Hello!
  Bot: Hi there! How can I help you today?
  You: What's 15 times 7?
  Bot: <uses calculator tool> 15 times 7 equals 105.
"""

import os
import sys
import argparse
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.slm import SLM
from src.model.config import ModelConfig
from src.tokenizer.tokenizer import Tokenizer
from src.inference.generate import Generator
from src.agent.agent import Agent


class ChatInterface:
    """
    Interactive chat interface for SLM.
    
    Args:
        generator: Generator instance
        mode: 'normal' or 'agent'
        preset: Sampling preset name
        verbose: Print debug info
    """
    
    def __init__(
        self,
        generator: Generator,
        mode: str = "normal",
        preset: str = "balanced",
        verbose: bool = False
    ):
        self.generator = generator
        self.mode = mode
        self.preset = preset
        self.verbose = verbose
        
        # Agent (lazy initialization)
        self._agent = None
        
        # Conversation history for normal mode
        self.conversation_history = []
    
    @property
    def agent(self):
        if self._agent is None:
            self._agent = Agent(
                generator=self.generator,
                max_iterations=5,
                max_tokens_per_iteration=200,
                verbose=self.verbose
            )
        return self._agent
    
    def reset(self):
        self.conversation_history = []
        if self._agent is not None:
            self._agent.reset()
        print(" Conversation reset")
    
    def set_mode(self, mode: str):
        if mode not in ["normal", "agent"]:
            print(f"  Invalid mode: {mode}. Use 'normal' or 'agent'")
            return
        
        self.mode = mode
        print(f"✓ Switched to {mode} mode")
    
    def set_preset(self, preset: str):
        valid_presets = ["greedy", "creative", "balanced", "focused", "precise"]
        if preset not in valid_presets:
            print(f"  Invalid preset: {preset}")
            print(f"   Valid presets: {', '.join(valid_presets)}")
            return
        
        self.preset = preset
        print(f" Using '{preset}' sampling preset")
    
    def toggle_verbose(self):
        self.verbose = not self.verbose
        if self._agent is not None:
            self._agent.verbose = self.verbose
        print(f" Verbose mode: {'ON' if self.verbose else 'OFF'}")
    
    def process_command(self, user_input: str) -> bool:
        if not user_input.startswith("/"):
            return False
        
        parts = user_input[1:].split()
        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        if cmd == "help":
            self.show_help()
        elif cmd == "reset":
            self.reset()
        elif cmd == "agent":
            self.set_mode("agent")
        elif cmd == "normal":
            self.set_mode("normal")
        elif cmd == "preset":
            if args:
                self.set_preset(args[0])
            else:
                print("  Usage: /preset <preset_name>")
        elif cmd == "verbose":
            self.toggle_verbose()
        elif cmd == "exit" or cmd == "quit":
            return "exit"
        else:
            print(f"  Unknown command: /{cmd}")
            print("   Type /help for available commands")
        
        return True
    
    def show_help(self):
        print("\n" + "=" * 60)
        print("Available Commands:")
        print("=" * 60)
        print("  /help       - Show this help")
        print("  /reset      - Clear conversation history")
        print("  /agent      - Switch to agent mode (with tools)")
        print("  /normal     - Switch to normal generation mode")
        print("  /preset X   - Set sampling preset:")
        print("                greedy, creative, balanced, focused, precise")
        print("  /verbose    - Toggle verbose output (debug info)")
        print("  /exit       - Exit chat")
        print("=" * 60)
        print(f"Current mode: {self.mode}")
        print(f"Current preset: {self.preset}")
        print(f"Verbose: {self.verbose}")
        print("=" * 60 + "\n")
    
    def chat(self, user_input: str) -> str:

        if self.mode == "agent":
            response = self.agent.chat(user_input)
        else:
            prompt_parts = []
            for msg in self.conversation_history:
                prompt_parts.append(f"{msg['role']}: {msg['content']}")
            prompt_parts.append(f"User: {user_input}")
            prompt_parts.append("Assistant:")
            
            prompt = "\n".join(prompt_parts)
            
            response = self.generator.generate_with_preset(
                prompt,
                preset=self.preset,
                max_new_tokens=150
            )
            
            self.conversation_history.append({"role": "User", "content": user_input})
            self.conversation_history.append({"role": "Assistant", "content": response})
        
        return response


def parse_args():
    parser = argparse.ArgumentParser(description="Chat with SLM")
    
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--mode", type=str, default="normal", choices=["normal", "agent"],
                       help="Chat mode: normal or agent (with tools)")
    parser.add_argument("--preset", type=str, default="balanced",
                       choices=["greedy", "creative", "balanced", "focused", "precise"],
                       help="Sampling preset")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--device", type=str, default="auto", help="Device: cuda, cpu, or auto")
    
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str):

    print(f" Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        if isinstance(saved_config, dict):
            model_config = checkpoint['model_state_dict']
            config = ModelConfig()
        else:
            config = saved_config
    else:
        print("  No config in checkpoint, using default")
        config = ModelConfig()
    
    model = SLM(config)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    info = {
        'step': checkpoint.get('current_step', 'unknown'),
        'val_loss': checkpoint.get('best_val_loss', 'unknown')
    }
    
    print(f" Loaded model from step {info['step']}")
    if info['val_loss'] != 'unknown':
        print(f"  Best validation loss: {info['val_loss']:.4f}")
    
    return model, config, info


def main():
    args = parse_args()
    
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print("\n" + "=" * 60)
    print("          SLM Chat Interface")
    print("=" * 60)
    print(f"Device: {device}")
    
    try:
        model, config, info = load_model(args.checkpoint, device)
    except Exception as e:
        print(f" Error loading checkpoint: {e}")
        print("\nTip: Make sure the checkpoint path is correct and the file exists.")
        return
    
    tokenizer = Tokenizer()
    
    generator = Generator(model, tokenizer, device=device)
    
    chat = ChatInterface(
        generator=generator,
        mode=args.mode,
        preset=args.preset,
        verbose=args.verbose
    )
    
    print(f"Mode: {args.mode}")
    print(f"Preset: {args.preset}")
    print("-" * 60)
    print("Type /help for commands, /exit to quit")
    print("=" * 60 + "\n")
    
    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                print("\n")
                break
            
            if not user_input:
                continue
            
            cmd_result = chat.process_command(user_input)
            if cmd_result == "exit":
                print("\n Goodbye!")
                break
            elif cmd_result:
                continue
            
            print("Bot: ", end="", flush=True)
            
            try:
                response = chat.chat(user_input)
                print(response)
            except Exception as e:
                print(f"\n Error generating response: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
            
            print()  
    
    except KeyboardInterrupt:
        print("\n\n Goodbye!")
    
    print()

if __name__ == "__main__":
    main()
