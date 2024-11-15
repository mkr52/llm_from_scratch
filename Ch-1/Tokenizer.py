from importlib.metadata import version
import re

# Print the versions of the 'torch' and 'tiktoken' packages
print("torch version:", version("torch"))
print("tiktoken version:", version("tiktoken"))

# Read the content of the file 'the-verdict.txt' into a string
with open("Ch-1/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    
# Print the total number of characters in the text and the first 99 characters
print("Total number of character:", len(raw_text))
print(raw_text[:99])

# Split the text into tokens based on specified punctuation and whitespace
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
# Remove any empty strings or whitespace-only strings from the list of tokens
preprocessed = [item.strip() for item in preprocessed if item.strip()]
# Print the first 30 tokens
print(preprocessed[:30])

# Create a sorted list of unique tokens
all_words = sorted(set(preprocessed))
# Calculate the size of the vocabulary
vocab_size = len(all_words)

# Print the size of the vocabulary
print(vocab_size)

# Create a dictionary mapping each token to a unique integer
vocab = {token:integer for integer,token in enumerate(all_words)}

# Create a list of all unique tokens and add special tokens for end of text and unknown tokens
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])

# Create a dictionary mapping each token (including special tokens) to a unique integer
vocab = {token:integer for integer,token in enumerate(all_tokens)}
# Print the vocabulary dictionary
print(vocab)

class SimpleTokenizerV2:
    """
    A simple tokenizer class that can encode text into a list of token IDs and decode a list of token IDs back into text.
    """
    def __init__(self, vocab):
        """
        Initialize the tokenizer with a vocabulary mapping tokens to integers.
        
        :param vocab: A dictionary mapping tokens to unique integers.
        """
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}
    
    def encode(self, text):
        """
        Encode a given text into a list of token IDs.
        
        :param text: The input text to encode.
        :return: A list of integers representing the token IDs.
        """
        # Split the text into tokens based on specified punctuation and whitespace
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        # Remove any empty strings or whitespace-only strings from the list of tokens
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        # Replace tokens not in the vocabulary with the unknown token
        preprocessed = [
            item if item in self.str_to_int 
            else "<|unk|>" for item in preprocessed
        ]

        # Convert tokens to their corresponding IDs
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        """
        Decode a list of token IDs back into text.
        
        :param ids: A list of integers representing the token IDs.
        :return: The decoded text as a string.
        """
        # Convert token IDs back to their corresponding tokens and join them into a string
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text
    
# Instantiate the tokenizer with the vocabulary
tokenizer = SimpleTokenizerV2(vocab)

# Example texts to encode and decode
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."

# Combine the example texts with a special end of text token
text = " <|endoftext|> ".join((text1, text2))

# Print the combined text
print(text)

# Encode the combined text and print the list of token IDs
print(tokenizer.encode(text))

# Decode the list of token IDs back into text and print it
print(tokenizer.decode(tokenizer.encode(text)))
