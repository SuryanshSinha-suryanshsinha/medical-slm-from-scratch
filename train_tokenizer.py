from tokenizers import ByteLevelBPETokenizer
import os

data_path = "data/raw/pubmed_abstracts.txt"
save_dir  = "tokenizer"
vocab_size = 32000
min_freq   = 2

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file not found: {data_path}")

file_size = os.path.getsize(data_path) / 1e6
print(f"Data file found: {data_path} ({file_size:.1f} MB)")

print("Creating tokenizer...")
tokenizer = ByteLevelBPETokenizer()

print(f"Training BPE tokenizer with vocab_size={vocab_size}...")
tokenizer.train(
    files=[data_path],
    vocab_size=vocab_size,
    min_frequency=min_freq,
    special_tokens=["<|endoftext|>", "<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>"]
)
print("Training complete.")

os.makedirs(save_dir, exist_ok=True)
tokenizer.save_model(save_dir)
print(f"Tokenizer saved to {save_dir}/")
print(f"  vocab.json  — {os.path.getsize(f'{save_dir}/vocab.json') / 1e3:.1f} KB")
print(f"  merges.txt  — {os.path.getsize(f'{save_dir}/merges.txt') / 1e3:.1f} KB")

