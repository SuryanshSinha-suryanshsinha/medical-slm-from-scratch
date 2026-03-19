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
    
