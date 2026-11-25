"""resnet blocks with group normalization and time conditioning"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ResBlock(nn.Module):
    """resnet block with group norm, siLU activation."""
    
    def __init__(
        self,
        dim: int,
        dim_out: int,
        time_emb_dim: int = None,
        dropout: float = 0.1,
        groups: int = 32
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.time_emb_dim = time_emb_dim
        
        self.norm1 = nn.GroupNorm(groups, dim)
        self.conv1 = nn.Conv2d(dim, dim_out, 3, padding=1)
        
        self.norm2 = nn.GroupNorm(groups, dim_out)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, padding=1)
        
        self.shortcut = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, dim_out)
            )
        
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        # x: (b, c, h, w)
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        
        if t is not None and self.time_emb_dim is not None:
            # add time embedding
            t_out = self.time_mlp(t)
            t_out = rearrange(t_out, 'b c -> b c 1 1')
            h = h + t_out
        
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        shortcut = self.shortcut(x)
        return h + shortcut


class ResBlockWithAttention(nn.Module):
    """resnet block with optional self-attention."""
    
    def __init__(
        self,
        dim: int,
        dim_out: int,
        time_emb_dim: int = None,
        dropout: float = 0.1,
        groups: int = 32,
        use_attention: bool = True,
        heads: int = 4,
        dim_head: int = 64
    ):
        super().__init__()
        
        self.res_block = ResBlock(dim, dim_out, time_emb_dim, dropout, groups)
        
        if use_attention:
            from .attention import MultiHeadAttention
            self.attn = MultiHeadAttention(
                dim_out, heads=heads, dim_head=dim_head, dropout=dropout
            )
            self.norm3 = nn.GroupNorm(groups, dim_out)
            self.act3 = nn.SiLU()
        else:
            self.attn = None
    
    def forward(self, x: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        # x: (b, c, h, w)
        h = self.res_block(x, t)
        
        if self.attn is not None:
            # apply attention
            h_attn = self.norm3(h)
            h_attn = self.act3(h_attn)
            h_attn = self.attn(h_attn)
            h = h + h_attn
        
        return h


class Downsample(nn.Module):
    """downsample with conv."""
    def __init__(self, dim: int, dim_out: int):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """upsample with nearest neighbor + conv."""
    def __init__(self, dim: int, dim_out: int):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)