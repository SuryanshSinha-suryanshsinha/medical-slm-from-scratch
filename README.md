# Medical SLM — From Scratch to Deployment

A 55M parameter small language model trained on PubMed biomedical abstracts.
Built end-to-end in PyTorch — custom BPE tokenizer, decoder-only transformer,
LoRA fine-tuning on MedQA, and local deployment via FastAPI.

## Status
Work in progress. Building phase by phase.

- [x] Phase 1 — Data download (NCBI PubMed)
- [x] Phase 1 — Custom BPE tokenizer (32K vocab)
- [ ] Phase 1 — Data preparation pipeline
- [ ] Phase 2 — Transformer architecture from scratch
- [ ] Phase 2 — Pretraining loop
- [ ] Phase 3 — HuggingFace port + LoRA fine-tuning
- [ ] Phase 4 — Quantization + deployment

## Setup

conda create -n slm_env python=3.11
conda activate slm_env
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install transformers tokenizers accelerate wandb tqdm numpy
