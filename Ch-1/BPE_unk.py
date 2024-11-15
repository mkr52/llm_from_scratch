import tiktoken

# Initialize the BPE tokenizer from the tiktoken library
bpe_tokenizer = tiktoken.get_encoding("gpt2")

# Define the unknown words
unknown_words = "Akwirw ier"

# Encode the unknown words into token IDs
token_ids = bpe_tokenizer.encode(unknown_words)
print("Token IDs:", token_ids)

# Decode each token ID to reproduce the mapping
decoded_tokens = [bpe_tokenizer.decode([token_id]) for token_id in token_ids]
print("Decoded Tokens:", decoded_tokens)

# Decode the entire list of token IDs to check if it reconstructs the original input
reconstructed_text = bpe_tokenizer.decode(token_ids)
print("Reconstructed Text:", reconstructed_text)