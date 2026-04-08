import os
import math
import time
import numpy as np
import torch
import wandb
from torch.cuda.amp import GradScaler
from contextlib import nullcontext
from model.model import ModelConfig, MedSLM
from dataclasses import dataclass
import torch.nn.functional as F

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
    max_steps:        int   = 20000
    # batch
    batch_size:       int   = 4
    grad_accum_steps: int   = 16
    # logging
    log_every:        int   = 100
    eval_every:       int   = 500
    save_every:       int   = 1000
    # paths
    checkpoint_dir:   str   = "checkpoints"
    run_name:         str   = "medslm-15m-pretrain"
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

def build_optimizer(model: MedSLM, cfg: TrainingConfig):
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]

    param_groups = [
        {'params': decay_params,   'weight_decay': cfg.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]

    num_decay = sum(p.numel() for p in decay_params)
    num_nodecay = sum(p.numel() for p in nodecay_params)
    print(f"Decay params: {num_decay:,} | No-decay params: {num_nodecay:,}")

    optimizer = torch.optim.AdamW(param_groups, lr=cfg.lr, betas=cfg.betas)
    return optimizer

def train_step(model, optimizer, scaler, loader, cfg, ctx):
    model.train()
    total_loss = 0.0

    for accum_step in range(cfg.grad_accum_steps):
        x, y = loader.get_batch()
        with ctx:
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, cfg.model_config.vocab_size),
                y.view(-1)
            ) / cfg.grad_accum_steps
        scaler.scale(loss).backward()
        total_loss += loss.item()

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    return total_loss

@torch.no_grad()
def evaluate(model, loader, cfg, ctx, num_batches=20):
    model.eval()
    total_loss = 0.0

    for _ in range(num_batches):
        x, y = loader.get_batch()
        with ctx:
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, cfg.model_config.vocab_size),
                y.view(-1)
            )
        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity

def train(cfg: TrainingConfig):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    device = cfg.device
    dtype = torch.bfloat16 if cfg.dtype == 'bfloat16' else torch.float32
    ctx = torch.amp.autocast(device_type='cuda', dtype=dtype)
    scaler = GradScaler()

    model = MedSLM(cfg.model_config).to(device)
    optimizer = build_optimizer(model, cfg)

    train_loader = DataLoader(cfg.data_dir, 'train', cfg.batch_size, cfg.model_config.context_length, device)
    val_loader   = DataLoader(cfg.data_dir, 'val',   cfg.batch_size, cfg.model_config.context_length, device)

    wandb.init(project='medslm', name=cfg.run_name, config=cfg.__dict__)

    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Training for {cfg.max_steps} steps")
    print(f"Effective batch size: {cfg.batch_size * cfg.grad_accum_steps}")

    t0 = time.time()

    for step in range(cfg.max_steps):
        lr = get_lr(step, cfg)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        loss = train_step(model, optimizer, scaler, train_loader, cfg, ctx)

        if step % cfg.log_every == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
            print(f"step {step:5d} | loss {loss:.4f} | lr {lr:.2e} | dt {dt:.2f}s")
            wandb.log({'train/loss': loss, 'train/lr': lr, 'train/grad_norm': grad_norm}, step=step)

        if step % cfg.eval_every == 0:
            val_loss, val_ppl = evaluate(model, val_loader, cfg, ctx)
            print(f"  val loss {val_loss:.4f} | perplexity {val_ppl:.2f}")
            wandb.log({'val/loss': val_loss, 'val/perplexity': val_ppl}, step=step)

        if step % cfg.save_every == 0 and step > 0:
            ckpt_path = os.path.join(cfg.checkpoint_dir, f'ckpt_step{step}.pt')
            torch.save({
                'step':                 step,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss':                 loss,
                'config':               cfg.model_config
            }, ckpt_path)
            print(f"  checkpoint saved → {ckpt_path}")

    final_path = os.path.join(cfg.checkpoint_dir, 'medslm_final.pt')
    torch.save({'model_state_dict': model.state_dict(), 'config': cfg.model_config}, final_path)
    print(f"Training complete. Final model saved → {final_path}")
    wandb.finish()


if __name__ == '__main__':
    model_cfg = ModelConfig()
    cfg = TrainingConfig(model_config=model_cfg)
    train(cfg)