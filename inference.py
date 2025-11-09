import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "ibm-granite/granite-4.0-h-micro"

print(f"Loading model from {model_path}...")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
model.eval()

print("Model loaded successfully!")

# Example inference
chat = [
    {"role": "user", "content": "Please write a long essay on the topic of 'The Future of AI', the importance of its regulation, and the every single detail of how AI works from technicals to social aspects."},
]

chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
input_tokens = tokenizer(chat, return_tensors="pt").to(device)

print("\nGenerating response...")
output = model.generate(**input_tokens, max_new_tokens=1000)
output = tokenizer.batch_decode(output)

print("\n" + "="*50)
print("OUTPUT:")
print("="*50)
print(output[0])

