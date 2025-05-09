from transformers import AutoTokenizer

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("unsloth/Mistral-Nemo-Base-2407-bnb-4bit")

text = "Hello, I am a language model."
tokens = tokenizer.encode(text)
print(f"Tokens : {tokens}")

decoded_text = tokenizer.decode(tokens)
print(f"Decoded text : {decoded_text}")