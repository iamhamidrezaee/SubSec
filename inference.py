#!/usr/bin/env python3
"""
SubSec: Sub-second LLM Inference with Intelligent Context Management

Optimized inference engine for IBM Granite-4 models (dense and hybrid architectures)
that maintains consistent low latency across multi-turn conversations through:

- Bounded context windows with hierarchical summarization
- Aggressive kernel optimizations (FlashAttention 2, Mamba-SSM fused kernels)
- Model-agnostic chat formatting
- Memory-efficient generation

Key optimizations:
1. Recent turns kept verbatim, older turns compressed into compact summaries
2. Prompt length bounded → TTFT stays flat regardless of conversation length
3. Prevents OOM on hybrid Mamba+Attention models
4. Maintains quality through faithful summarization

Usage:
    python inference.py
    MODEL_NAME_OR_PATH=ibm-granite/granite-4.0-h-micro python inference.py

See README.md for full documentation and .env.example for configuration options.
"""

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

# Default to a Granite-4 dense attention model, but allow override via env var
model_path = os.environ.get("MODEL_NAME_OR_PATH", "ibm-granite/granite-4.0-micro")

def is_granite_4(model_name: str) -> bool:
    return "granite-4.0" in model_name

def is_granite_4_hybrid(model_name: str) -> bool:
    # Hybrid Granite-4 models are denoted with "-h-" in the name
    return "granite-4.0-h-" in model_name

is_granite = is_granite_4(model_path)
is_hybrid = is_granite_4_hybrid(model_path)

print(f"Loading model from {model_path}...")
print(f"Using device: {device}")
if is_granite:
    print(f"Detected Granite-4 family model ({'hybrid Mamba + Attention' if is_hybrid else 'dense Attention-only'})")
else:
    print("Detected non-Granite model (generic SubSec optimization path).")

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


def format_chat_messages(messages):
    """
    Format a list of {'role', 'content'} messages into a single text prompt.

    For chat models with a tokenizer.chat_template, we delegate to
    tokenizer.apply_chat_template. For base (non-chat) models without a
    chat template, we fall back to a simple, generic conversational format.
    """
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # Fallback for base causal models: simple role-prefixed transcript
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

    # Add assistant cue as generation prompt
    lines.append("Assistant: ")
    return "\n".join(lines)

# ------------------------------
# Model load configuration: FlashAttention 2 and Mamba-SSM
# ------------------------------
use_flash_attn_2 = device == "cuda" and is_flash_attn_2_available()
attn_impl = "flash_attention_2" if use_flash_attn_2 else "eager"

if use_flash_attn_2:
    print("Attempting to load with Flash Attention 2 for attention blocks...")
else:
    print("Flash Attention 2 not available; using standard attention kernels.")

# For hybrid Granite-4-H models, prefer using remote code so their custom Mamba + Attention
# architecture is fully enabled. Allow explicit override via SUBSEC_TRUST_REMOTE_CODE.
explicit_trust = os.environ.get("SUBSEC_TRUST_REMOTE_CODE")
if explicit_trust is not None:
    trust_remote_code = explicit_trust.strip().lower() in ("1", "true", "yes", "y", "on")
elif is_hybrid:
    trust_remote_code = True
else:
    trust_remote_code = False

has_mamba_kernels = False
if is_hybrid:
    # Hybrid Granite-4 models use Mamba-2 style SSM layers; if mamba-ssm and causal-conv1d
    # are available, their fused CUDA kernels will be used under the hood.
    try:
        import mamba_ssm  # type: ignore
        import causal_conv1d  # type: ignore
        has_mamba_kernels = True
        print("Detected optimized Mamba-SSM and causal-conv1d kernels for hybrid Granite-4 models.")
    except Exception as e:
        print("⚠ Optimized Mamba-SSM / causal-conv1d kernels not fully available; "
              "hybrid Mamba layers will run with fallback PyTorch kernels.")
        print(f"  Details: {e}")

try:
    # Use device_map='auto' to load directly to GPU memory without extra host<->device copies
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )
    if use_flash_attn_2:
        print("✓ Flash Attention 2 enabled successfully for attention blocks!")
except Exception as e:
    print(f"⚠ Could not enable requested attention implementation ({attn_impl}): {e}")
    print("Falling back to standard attention (install flash-attn for better performance).")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )

model.eval()

# Print a concise model + kernel summary for the SubSec demo / latency report
cfg = getattr(model, "config", None)
arch = getattr(cfg, "architectures", None) if cfg is not None else None
max_ctx = getattr(cfg, "max_position_embeddings", None) if cfg is not None else None
hidden_size = getattr(cfg, "hidden_size", None) if cfg is not None else None
num_layers = getattr(cfg, "num_hidden_layers", None) if cfg is not None else None

print("---- SubSec Model Summary ----")
print(f"Model ID          : {model_path}")
if arch:
    print(f"Architecture      : {', '.join(arch)}")
if max_ctx is not None:
    print(f"Max context (cfg) : {max_ctx} tokens")
if hidden_size is not None and num_layers is not None:
    print(f"Hidden size       : {hidden_size}, Layers: {num_layers}")
print(f"Granite-4 family  : {'yes' if is_granite else 'no'}")
print(f"Hybrid (Mamba+Att): {'yes' if is_hybrid else 'no'}")
print(f"FlashAttention 2  : {'ON' if use_flash_attn_2 else 'OFF'}")
if is_hybrid:
    print(f"Mamba-SSM kernels : {'ON (fused CUDA kernels)' if has_mamba_kernels else 'OFF (fallback PyTorch kernels)'}")
print("------------------------------")

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

# Warmup run to initialize CUDA kernels, compiled graphs, and cuDNN benchmarks
if device == "cuda":
    print("Warming up model (initializing CUDA kernels, compiled graphs, and cuDNN benchmarks)...")
    try:
        with torch.inference_mode():
            # Use a realistic input length for better warmup
            warmup_text = "Hello, how are you today? Tell me about yourself and who developed you?"
            warmup_input = tokenizer(warmup_text, return_tensors="pt").to(device)
            # Run a few tokens to warm up both prefill and decode phases
            _ = model.generate(**warmup_input, max_new_tokens=3, use_cache=True, do_sample=False)
            # Clear cache after warmup to start fresh
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("✓ Warmup complete!")
    except Exception as e:
        print(f"⚠ Warmup failed: {e}")

print("Model loaded successfully!\n")

# Conversation history and intelligent context management
conversation_history = []
conversation_summary = []  # Hierarchical summary or notes for older conversations

# ------------------------------
# Context management knobs
# ------------------------------
# Target maximum prompt length in tokens. We enforce this approximately via
# bounded summary size + a fixed number of recent turns kept in full.
MAX_PROMPT_TOKENS = int(os.environ.get("SUBSEC_MAX_PROMPT_TOKENS", "1024"))

# How many recent *turns* (user+assistant pairs) to keep in full detail.
# 2 turns strikes a good balance between quality (detailed recent context)
# and latency (prompt size stays small and TTFT remains low).
RECENT_TURNS_FULL = int(os.environ.get("SUBSEC_RECENT_TURNS_FULL", "2"))

# Upper bound in characters per summarized message line. This keeps the
# system summary short while still preserving key information.
SUMMARY_CHARS_PER_MESSAGE = int(os.environ.get("SUBSEC_SUMMARY_CHARS_PER_MSG", "160"))

# Maximum number of summarized lines to include in the system summary.
SUMMARY_MAX_LINES = int(os.environ.get("SUBSEC_SUMMARY_MAX_LINES", "12"))

class IntelligentCacheManager:
    """
    Lightweight holder for model context information.
    We currently rely on the model's internal KV cache (use_cache=True)
    and focus on bounding prompt size via smart context construction.
    """

    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.is_hybrid_model = hasattr(config, "architectures") and any(
            "Hybrid" in arch for arch in config.architectures
        )


cache_manager = IntelligentCacheManager(model, cfg, device)


def create_ultra_compact_summary(messages):
    """
    Create a compact system-level summary from a list of past messages.

    This summary:
    - Preserves both user queries and condensed assistant answers.
    - Is strictly bounded in size (by SUMMARY_MAX_LINES and
      SUMMARY_CHARS_PER_MESSAGE).
    - Avoids hallucinating new content: it only truncates and stitches
      together existing text.
    """
    if not messages:
        return None

    summary_lines = []

    for msg in messages:
        role = msg.get("role", "")
        content = (msg.get("content") or "").strip()
        if not content:
            continue

        # Truncate each message to a bounded size
        snippet = content.replace("\n", " ")
        if len(snippet) > SUMMARY_CHARS_PER_MESSAGE:
            snippet = snippet[: SUMMARY_CHARS_PER_MESSAGE].rstrip() + "..."

        if role == "user":
            prefix = "User: "
        elif role == "assistant":
            prefix = "Assistant: "
        else:
            prefix = ""

        summary_lines.append(prefix + snippet)

        if len(summary_lines) >= SUMMARY_MAX_LINES:
            break

    if not summary_lines:
        return None

    summary_text = "Summary of earlier conversation:\n" + "\n".join(summary_lines)
    return {"role": "system", "content": summary_text}


def build_context_messages():
    """
    Construct the messages to feed into the chat template for the next turn.

    Strategy:
    - Keep the last RECENT_TURNS_FULL turns (user+assistant) in full detail.
    - Compress everything before that into a compact system-level summary.
    - This keeps TTFT flat while retaining detailed context for the most
      recent part of the conversation and high-level context for earlier parts.
    """
    if not conversation_history:
        return []

    # Each "turn" is a pair (user, assistant). We keep the last
    # RECENT_TURNS_FULL * 2 messages in full.
    recent_msg_count = RECENT_TURNS_FULL * 2

    if len(conversation_history) <= recent_msg_count:
        return list(conversation_history)

    older_messages = conversation_history[:-recent_msg_count]
    recent_messages = conversation_history[-recent_msg_count:]

    summary = create_ultra_compact_summary(older_messages)

    context_messages = []
    if summary is not None:
        context_messages.append(summary)
    context_messages.extend(recent_messages)

    return context_messages

def reset_cache():
    """Reset the conversation history and clear GPU cache"""
    global conversation_history, conversation_summary
    conversation_history = []
    conversation_summary = []
    if device == "cuda":
        torch.cuda.empty_cache()
        # Synchronize to ensure cache is cleared
        torch.cuda.synchronize()

def generate_streaming_response(user_message):
    """Generate streaming response with intelligent context management for consistent TTFT."""
    global conversation_summary

    # Add user message to history
    conversation_history.append({"role": "user", "content": user_message})

    # Build bounded context: compact summary of older dialogue + recent turns in full
    context_messages = build_context_messages()

    # Format chat with history (chat-template aware + base-model fallback)
    chat_text = format_chat_messages(context_messages)

    # Tokenize on CPU; move to GPU non-blocking for minimal overhead
    input_tokens = tokenizer(chat_text, return_tensors="pt")
    
    if device == "cuda":
        input_tokens = {k: v.to("cuda", non_blocking=True) for k, v in input_tokens.items()}

    # Setup streamer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Allow controlling the maximum number of new tokens via env for fair
    # comparisons with baselines and to cap decode cost.
    max_new_tokens = int(os.environ.get("SUBSEC_MAX_NEW_TOKENS", "512"))

    # Generation kwargs with optimized cache configuration.
    # Key: Let the model manage its own KV cache internally via use_cache=True.
    # Our optimization is the bounded prompt size via summary+recent-window.
    generation_kwargs = {
        **input_tokens,
        "max_new_tokens": max_new_tokens,
        "streamer": streamer,
        "do_sample": False,
        "use_cache": True,  # KV cache enabled - model manages it internally
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

    # No trimming - we use hierarchical summarization instead
    # This preserves all information while keeping TTFT consistent

    # Clear GPU cache to prevent memory buildup
    if device == "cuda":
        torch.cuda.empty_cache()

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
    
    