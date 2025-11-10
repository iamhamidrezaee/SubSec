import torch
import time
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "ibm-granite/granite-4.0-h-micro"

print(f"Loading model from {model_path}...")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load model with Flash Attention 2 for optimized inference
# Flash Attention 2 provides up to 2x speedup on attention layers
try:
    print("Attempting to load with Flash Attention 2...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        attn_implementation="flash_attention_2",
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    print("✓ Flash Attention 2 enabled successfully!")
except Exception as e:
    print(f"⚠ Flash Attention 2 not available: {e}")
    print("Falling back to standard attention (install flash-attn for better performance)")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
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

    # Format chat with history
    chat_text = tokenizer.apply_chat_template(
        conversation_history,
        tokenize=False,
        add_generation_prompt=True
    )
    input_tokens = tokenizer(chat_text, return_tensors="pt").to(device)

    # Setup streamer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Generation kwargs
    # The model will handle caching internally with use_cache=True
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

