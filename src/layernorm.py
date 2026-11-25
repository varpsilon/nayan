"""layer normalization block"""
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """channel-first layernorm."""
    
    def __init__(self, dim: int, scale: bool = True):
        super().__init__()
        self.dim = dim
        self.scale = scale
        
        self.weight = nn.Parameter(torch.ones(dim)) if scale else None
        self.bias = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, c, h, w)
        stats = x.mean([2, 3], keepdim=True)
        std = (x.var([2, 3], unbiased=False, keepdim=True) + 1e-5).sqrt()
        y = (x - stats) / std
        
        if self.weight is not None:
            y = y * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        else:
            y = y + self.bias.view(1, -1, 1, 1)
        
        return y