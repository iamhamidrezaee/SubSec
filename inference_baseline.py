#!/usr/bin/env python3
"""
SubSec Baseline: Naive Full-History Inference (for comparison)

This is the BASELINE implementation that uses the standard approach:
- Full conversation history concatenated into every prompt
- No summarization or context bounding
- No FlashAttention or kernel optimizations

Purpose: Demonstrates the problem that SubSec solves by showing:
- TTFT degradation over conversation turns
- Memory issues (OOM) on hybrid models with long contexts
- Inefficient compute usage from re-processing old context

This baseline is INTENTIONALLY unoptimized to provide a fair comparison point.

Usage:
    python inference_baseline.py
    MODEL_NAME_OR_PATH=ibm-granite/granite-4.0-micro python inference_baseline.py
"""

import os
# Set environment variable BEFORE any other imports to prevent torchvision import
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

import torch
import time
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

device = "cuda" if torch.cuda.is_available() else "cpu"
# Allow overriding the model via environment variable for fair comparisons
model_path = os.environ.get("MODEL_NAME_OR_PATH", "ibm-granite/granite-4.0-micro")

print(f"Loading model from {model_path}...")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_path)


def format_chat_messages(messages):
    """
    Format messages into a single prompt for both chat and base models.

    Baseline uses the same generic formatting strategy as the optimized
    path when no tokenizer.chat_template is defined, for model-agnostic
    behavior.
    """
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    lines = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            prefix = "User: "
        elif role == "assistant":
            prefix = "Assistant: "
        elif role == "system":
            prefix = "System: "
        else:
            prefix = ""
        lines.append(prefix + content)

    lines.append("Assistant: ")
    return "\n".join(lines)


# Plain model load: no Flash Attention, no custom KV cache, no speculative decoding.
# We still use device_map='auto' on CUDA to avoid OOM on larger Granite-4-H models.
torch_dtype = torch.float16 if device == "cuda" else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch_dtype,
    device_map="auto" if device == "cuda" else None,
    low_cpu_mem_usage=True,
)
model.eval()

print("Model loaded successfully!\n")

# Conversation history
conversation_history = []

def reset_cache():
    """Reset the conversation history"""
    global conversation_history
    conversation_history = []

def trim_conversation_history():
    """Trim conversation history to prevent context from becoming too long"""
    global conversation_history
    # Keep only the last 10 turns (5 user + 5 assistant messages) to prevent context from growing too large
    max_turns = 10
    if len(conversation_history) > max_turns:
        conversation_history = conversation_history[-max_turns:]

def generate_streaming_response(user_message):
    """Generate streaming response with metrics"""
    # Add user message to history
    conversation_history.append({"role": "user", "content": user_message})

    # Format chat with history (chat-template aware + base-model fallback)
    chat_text = format_chat_messages(conversation_history)
    input_tokens = tokenizer(chat_text, return_tensors="pt").to(device)

    # Setup streamer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Generation kwargs (plain defaults; caching is enabled by default)
    generation_kwargs = {
        **input_tokens,
        "max_new_tokens": 512,
        "streamer": streamer,
        "do_sample": False,
        "use_cache": True,
    }

    # Generate in a separate thread to enable streaming
    generation_thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    generation_thread.start()

    # Track metrics
    start_time = time.time()
    first_token_time = None
    token_count = 0
    generated_text = ""

    print("Assistant: ", end="", flush=True)

    # Stream tokens
    for token in streamer:
        if first_token_time is None:
            first_token_time = time.time()
            time_to_first_token = first_token_time - start_time
            print(f"[TTFT: {time_to_first_token*1000:.1f}ms] ", end="", flush=True)

        print(token, end="", flush=True)
        generated_text += token
        token_count += 1

    generation_thread.join()
    end_time = time.time()

    # Calculate metrics
    total_time = end_time - start_time
    generation_time = end_time - (first_token_time or start_time)
    tokens_per_second = token_count / generation_time if generation_time > 0 else 0

    # Add assistant response to history
    conversation_history.append({"role": "assistant", "content": generated_text})

    # Trim conversation history to prevent it from growing too large
    trim_conversation_history()

    # Print metrics
    print(f"\n\n[Metrics] Tokens: {token_count} | "
          f"Time: {total_time*1000:.1f}ms | "
          f"Tokens/sec: {tokens_per_second:.1f} | "
          f"TTFT: {(first_token_time - start_time)*1000:.1f}ms\n")

def main():
    print("="*60)
    print("SubSec Chat Interface (Baseline)")
    print(f"Model: {model_path}")
    print("No Flash Attention, no custom KV cache, no speculative decoding")
    print("Type 'quit' or 'exit' to end the conversation")
    print("Type 'clear', 'reset', or 'new' to clear conversation history")
    print("="*60 + "\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            elif user_input.lower() in ['clear', 'reset', 'new']:
                reset_cache()
                print("Conversation history and cache cleared. Starting fresh!\n")
                continue

            generate_streaming_response(user_input)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")

if __name__ == "__main__":
    main()


