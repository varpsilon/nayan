"""sinusoidal time embeddings with learned scaling"""
import torch
import torch.nn as nn
import math


class SinusoidalPosEmb(nn.Module):
    """sinusoidal positional embedding with bias."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (b,)
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class LearnedSinusoidalPosEmb(nn.Module):
    """sinusoidal with learned weights."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.weights = nn.Parameter(torch.randn(dim))
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (b,)
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb * self.weights


class TimeEmbedding(nn.Module):
    """time embedding with gated linear units."""
    
    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        
        self.sinemb = LearnedSinusoidalPosEmb(dim)
        self.proj1 = nn.Linear(dim * 2, hidden_dim)  # sin + cos
        self.proj2 = nn.Linear(hidden_dim, hidden_dim)
        
        # projection to model dim
        self.to_model_dim = nn.Linear(hidden_dim, dim)
        
        self.act = nn.SiLU()
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (b,)
        emb = self.sinemb(t)  # (b, dim*2)
        emb = self.proj1(emb)
        emb = self.act(emb)
        emb = self.proj2(emb)
        emb = self.act(emb)
        return self.to_model_dim(emb)


# lazy import
from typing import Optional