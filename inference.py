import os
# Set environment variable BEFORE any other imports to prevent torchvision import
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

import time
import threading
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers.utils import is_flash_attn_2_available

# ------------------------------
# Device and model configuration
# ------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# Enforce non-hybrid, full-attention model to avoid Mamba/SSM paths and their warnings
# Allow override via env var if needed, but default to non-hybrid.
model_path = os.environ.get("MODEL_NAME_OR_PATH", "ibm-granite/granite-4.0-micro")

print(f"Loading model from {model_path}...")
print(f"Using device: {device}")

# ------------------------------
# Performance knobs for latency
# ------------------------------
if device == "cuda":
    # Enable TF32 where beneficial (does not affect FP16/BF16 precision, speeds up some matmuls)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

def select_inference_dtype() -> torch.dtype:
    if device == "cuda":
        # Prefer BF16 on modern GPUs; else fall back to FP16
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32

torch_dtype = select_inference_dtype()

# ------------------------------
# Tokenizer
# ------------------------------
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
# Ensure pad token is defined to silence warnings and speed up batch/padding logic
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ------------------------------
# Model load with FlashAttention 2
# ------------------------------
use_flash_attn_2 = device == "cuda" and is_flash_attn_2_available()
attn_impl = "flash_attention_2" if use_flash_attn_2 else "eager"

if use_flash_attn_2:
    print("Attempting to load with Flash Attention 2...")
else:
    print("Flash Attention 2 not available; using standard attention.")

try:
    # Use device_map='auto' to load directly to GPU memory without extra host<->device copies
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    if use_flash_attn_2:
        print("✓ Flash Attention 2 enabled successfully!")
except Exception as e:
    print(f"⚠ Could not enable requested attention implementation ({attn_impl}): {e}")
    print("Falling back to standard attention (install flash-attn for better performance).")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )

model.eval()

# Generation defaults for low latency
if getattr(model, "generation_config", None) is not None:
    gen_cfg = model.generation_config
    # Ensure eos/pad tokens are valid
    if gen_cfg.pad_token_id is None and tokenizer.pad_token_id is not None:
        gen_cfg.pad_token_id = tokenizer.pad_token_id
    if gen_cfg.eos_token_id is None and tokenizer.eos_token_id is not None:
        gen_cfg.eos_token_id = tokenizer.eos_token_id
    # Latency-optimized default generation behavior
    gen_cfg.use_cache = True
    gen_cfg.num_beams = 1
    gen_cfg.do_sample = False

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

    # Format chat with history
    chat_text = tokenizer.apply_chat_template(
        conversation_history,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize on CPU; move to GPU non-blocking for minimal overhead
    input_tokens = tokenizer(chat_text, return_tensors="pt")
    if device == "cuda":
        input_tokens = {k: v.to("cuda", non_blocking=True) for k, v in input_tokens.items()}

    # Setup streamer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Generation kwargs
    # The model will handle caching internally with use_cache=True
    generation_kwargs = {
        **input_tokens,
        "max_new_tokens": 256,  # lower default for latency; adjust as needed
        "streamer": streamer,
        "do_sample": False,
        "use_cache": True,
        "num_beams": 1,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }

    # Generate in a separate thread to enable streaming
    start_time = time.time()
    first_token_time = None
    token_count = 0
    generated_text = ""

    def _generate():
        with torch.inference_mode():
            model.generate(**generation_kwargs)

    generation_thread = threading.Thread(target=_generate)
    generation_thread.start()

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
    ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0.0
    print(f"\n\n[Metrics] Tokens: {token_count} | "
          f"Time: {total_time*1000:.1f}ms | "
          f"Tokens/sec: {tokens_per_second:.1f} | "
          f"TTFT: {ttft_ms:.1f}ms\n")

def main():
    print("="*60)
    print("SubSec Chat Interface")
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
    
    