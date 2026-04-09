# Medical SLM — From Scratch to Deployment

A 15M parameter small language model trained on PubMed biomedical abstracts.
Built end-to-end in PyTorch — custom BPE tokenizer, decoder-only transformer,
LoRA fine-tuning on MedQA, and local deployment via FastAPI.

---

## Architecture

| Component | Detail |
|---|---|
| Parameters | 15,276,288 (~15M) |
| Layers | 12 transformer blocks |
| Hidden dim | 256 |
| Attention | Grouped Query Attention (4Q, 2KV heads) |
| Head dim | 64 |
| FFN | SwiGLU (3 matrices, intermediate=512) |
| Positional encoding | Rotary (RoPE) — zero learnable params |
| Normalization | RMSNorm (pre-norm) |
| Context length | 1024 tokens |
| Vocabulary | 32,000 BPE tokens |
| Precision | bf16 mixed precision |
| Memory optimizations | Flash Attention 2, gradient checkpointing |

---

## Status

- [x] Phase 1 — Data download (500K+ NCBI PubMed abstracts)
- [x] Phase 1 — Custom BPE tokenizer (32K vocab, trained on medical text)
- [x] Phase 1 — Data preparation pipeline (70M+ tokens, binary format)
- [x] Phase 2 — Transformer architecture from scratch
- [x] Phase 2 — Pretraining loop 
- [ ] Phase 3 — HuggingFace port + LoRA fine-tuning
- [ ] Phase 4 — Quantization + deployment

---

## Phase 1 — Data & Tokenization

**Data**
- 500K+ PubMed abstracts downloaded from NCBI FTP
- Stored in `data/raw/pubmed_abstracts.txt`

**Tokenizer**
- BPE tokenizer trained from scratch on medical text
- Vocabulary size: 32,000
- Special tokens: `<|endoftext|>=0`, `<|pad|>=1`, `<|unk|>=2`, `<|bos|>=3`, `<|eos|>=4`
- Stored in `tokenizer/vocab.json` and `tokenizer/merges.txt`

**Prepared Data**
- `data/tokenized/train.bin` — 63,276,030 tokens (126.6MB)
- `data/tokenized/val.bin` — 7,030,669 tokens (14.1MB)
- Chunk size: 1024 tokens, 61,792 possible training chunks

---

## Phase 2 — Transformer Architecture

**Model components built from scratch:**

`RMSNorm` — normalization without mean subtraction, 256 learnable params per instance

`RotaryEmbedding` — RoPE positional encoding, zero learnable params, precomputed cos/sin tables

`GroupedQueryAttention` — 4 query heads share 2 KV heads, 2× KV cache reduction vs standard MHA, Flash Attention 2 via `F.scaled_dot_product_attention`

`SwiGLUFFN` — gated feed-forward with three weight matrices (W1, W2, W3), smoother gradients than ReLU

`TransformerBlock` — pre-norm residual architecture, gradient checkpointing for VRAM efficiency

`MedSLM` — full decoder-only model, weight tying between embedding and LM head

**Parameter breakdown:**

```

Embedding table:       8,192,000
12 × TransformerBlock:
Attention (GQA):       196,608
FFN (SwiGLU):          393,216
RMSNorm (×2):              512
Per block:             590,336
× 12 layers:         7,084,032
Final RMSNorm:               256
LM Head (weight tied):         0
─────────────────────────────────
Total:                15,276,288
```

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 2e-4 (peak) |
| Betas | (0.9, 0.95) |
| Weight decay | 0.1 |
| Gradient clipping | 1.0 |
| Warmup steps | 1000 |
| Max steps | 20,000 |
| Batch size | 4 |
| Grad accumulation | 16 (effective batch = 64) |
| Precision | bf16 |
| LR schedule | Cosine decay with linear warmup |

---

## Training History

| Run | Model size | Dataset | Steps | Train loss | Val loss | Perplexity |
|---|---|---|---|---|---|---|
| Run 1 | 92M params | 11M tokens | 10,000 | 0.15 | 8.59 | 5389 |
| Run 2 | 15M params | 70M tokens | 20,000 | in progress | — | — |

Run 1 showed severe overfitting — 92M parameters with only 11M tokens is heavily under-resourced.
Run 2 addresses this with a smaller model and 6.4× more data.

---

## Hardware

- RTX 5070 (mobile), 8GB VRAM
- Ryzen AI 9 HX 370, 24GB RAM, Windows 11
- PyTorch 2.12.0.dev+cu128, bf16 supported

---

## Setup

```bash
conda create -n slm_env python=3.11
conda activate slm_env
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install transformers tokenizers accelerate wandb tqdm numpy
```

---

## Project Structure

```

slm_medical/
├── data/
│   ├── raw/
│   │   └── pubmed_abstracts.txt
│   └── tokenized/
│       ├── train.bin
│       └── val.bin
├── tokenizer/
│   ├── vocab.json
│   └── merges.txt
├── model/
│   └── model.py          ← Phase 2 complete
├── train/
│   └── train.py          ← Phase 2 in progress
├── checkpoints/
├── deploy/               ← Phase 4
├── download_data.py
├── train_tokenizer.py
└── prepare_data.py
```

---

## About

Building a biomedical language model from scratch as a deep learning learning project.
Documenting progress publicly on GitHub and X (@sspub2).