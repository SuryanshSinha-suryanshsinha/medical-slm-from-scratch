import torch                                          
import torch.nn as nn                                 
import torch.nn.functional as F                       
import math                                           
from dataclasses import dataclass                     
from typing import Optional                           
from torch.utils.checkpoint import checkpoint        

@dataclass
class ModelConfig:
    vocab_size:       int   = 32000   
    context_length:   int   = 1024    
    hidden_dim:       int   = 512     
    n_layers:         int   = 20      
    n_heads_q:        int   = 8       
    n_heads_kv:       int   = 2       
    head_dim:         int   = 64      
    ffn_intermediate: int   = 2048    
    dropout:          float = 0.0     
    norm_eps:         float = 1e-6     
    max_batch_size:   int   = 4  


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / rms)
    
class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, context_length: int):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        theta = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(context_length).float()
        angles = torch.outer(positions, theta)
        cos = torch.cat([angles.cos(), angles.cos()], dim=-1)
        sin = torch.cat([angles.sin(), angles.sin()], dim=-1)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple:
        seq_len = q.shape[2]
        cos = self.cos[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin[:seq_len].unsqueeze(0).unsqueeze(0)
        q_rot = q * cos + self.rotate_half(q) * sin
        k_rot = k * cos + self.rotate_half(k) * sin
        return q_rot, k_rot
    
class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads_q = config.n_heads_q
        self.n_heads_kv = config.n_heads_kv
        self.head_dim = config.head_dim
        self.group_size = config.n_heads_q // config.n_heads_kv

        self.W_q = nn.Linear(config.hidden_dim, config.n_heads_q * config.head_dim, bias=False)
        self.W_k = nn.Linear(config.hidden_dim, config.n_heads_kv * config.head_dim, bias=False)
        self.W_v = nn.Linear(config.hidden_dim, config.n_heads_kv * config.head_dim, bias=False)
        self.W_o = nn.Linear(config.n_heads_q * config.head_dim, config.hidden_dim, bias=False)

        self.rope = RotaryEmbedding(config.head_dim, config.context_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        q = q.view(batch, seq_len, self.n_heads_q, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads_kv, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads_kv, self.head_dim).transpose(1, 2)

        q, k = self.rope(q, k)

        k = torch.repeat_interleave(k, self.group_size, dim=1)
        v = torch.repeat_interleave(v, self.group_size, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.n_heads_q * self.head_dim)
        return self.W_o(out)
    
