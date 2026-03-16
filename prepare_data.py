import numpy as np
import os
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm

data_path    = "data/raw/pubmed_abstracts.txt"
token_dir    = "data/tokenized"
tokenizer_dir = "tokenizer"
context_len  = 1024
val_split    = 0.1

# load tokenizer
  
print("Loading tokenizer...")
tokenizer = ByteLevelBPETokenizer(
    f"{tokenizer_dir}/vocab.json",
    f"{tokenizer_dir}/merges.txt"
)
eot_id = tokenizer.token_to_id("<|endoftext|>")
print(f"Tokenizer loaded. Vocab size: {tokenizer.get_vocab_size()}")
print(f"End-of-text token ID: {eot_id}")

# read and encode all abstracts 

print("Reading and encoding abstracts...")
all_token_ids = []

with open(data_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

for line in tqdm(lines, desc="Tokenizing abstracts"):
    line = line.strip()
    if len(line) == 0:
        continue
    encoded = tokenizer.encode(line)
    all_token_ids.extend(encoded.ids)
    all_token_ids.append(eot_id)

print(f"Total tokens: {len(all_token_ids):,}")

# split into train and val sets 

print("Splitting into train and validation sets...")
total_tokens = len(all_token_ids)
val_tokens = int(total_tokens * val_split)
train_tokens = total_tokens - val_tokens

print(f"Train tokens: {train_tokens:,}")
print(f"Validation tokens: {val_tokens:,}") 
       
train_ids = all_token_ids[:train_tokens]
val_ids   = all_token_ids[train_tokens:]    

# convert to numpy and save as binary 

os.makedirs(token_dir, exist_ok=True)

train_arr = np.array(train_ids, dtype=np.uint16)
val_arr   = np.array(val_ids,   dtype=np.uint16)

train_path = os.path.join(token_dir, "train.bin")
val_path   = os.path.join(token_dir, "val.bin")

train_arr.tofile(train_path)
val_arr.tofile(val_path)

print(f"\nSaved binary files:")
print(f"  train.bin — {os.path.getsize(train_path) / 1e6:.1f} MB  ({len(train_arr):,} tokens)")
print(f"  val.bin   — {os.path.getsize(val_path)   / 1e6:.1f} MB  ({len(val_arr):,} tokens)")
print(f"\nDone. Data pipeline complete.")
print(f"Each training chunk will be {context_len} tokens.")
print(f"Number of possible chunks: {len(train_arr) // context_len:,}")