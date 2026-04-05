import os
import math
import time
from matplotlib.pyplot import step
import numpy as np
import torch
import wandb
from torch.cuda.amp import GradScaler
from contextlib import nullcontext
from model.model import ModelConfig, MedSLM
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # data
    data_dir:         str   = "data/tokenized"
    # model
    model_config:     ModelConfig = None
    # optimizer
    lr:               float = 2e-4
    betas:            tuple = (0.9, 0.95)
    weight_decay:     float = 0.1
    grad_clip:        float = 1.0
    # schedule
    warmup_steps:     int   = 1000
    max_steps:        int   = 10000
    # batch
    batch_size:       int   = 4
    grad_accum_steps: int   = 16
    # logging
    log_every:        int   = 100
    eval_every:       int   = 500
    save_every:       int   = 1000
    # paths
    checkpoint_dir:   str   = "checkpoints"
    run_name:         str   = "medslm-pretrain"
    # device
    device:           str   = "cuda"
    dtype:            str   = "bfloat16"


class DataLoader:
    def __init__(self, data_dir: str, split: str, batch_size: int, context_length: int, device: str):
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device
        path = os.path.join(data_dir, f"{split}.bin")
        self.data = np.memmap(path, dtype=np.uint16, mode='r')
        self.num_tokens = len(self.data)
        self.num_chunks = self.num_tokens // context_length
        print(f"Loaded {split}.bin — {self.num_tokens:,} tokens, {self.num_chunks:,} possible chunks")

    def get_batch(self):
        ix = torch.randint(self.num_tokens - self.context_length, (self.batch_size,))
        x = torch.stack([torch.from_numpy(self.data[i : i + self.context_length].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i + 1 : i + 1 + self.context_length].astype(np.int64)) for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        return x, y

def get_lr(step: int, cfg: TrainingConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step / cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress))