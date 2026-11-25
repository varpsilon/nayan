"""multihead attention with relative positional encoding"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class RelativePositionBias(nn.Module):
    """relative position bias for attention."""
    
    def __init__(self, num_heads: int, max_distance: int = 128):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(2 * max_distance - 1, num_heads)
        )
        
        # distance to index mapping
        self.register_buffer(
            "relative_position_index",
            self._compute_relative_position_index(max_distance),
            persistent=False
        )
    
    def _compute_relative_position_index(self, max_distance: int) -> torch.Tensor:
        """compute relative position index for attention."""
        # create position indices
        positions = torch.arange(max_distance).unsqueeze(1) - torch.arange(max_distance).unsqueeze(0)
        positions = torch.clamp(positions, -max_distance + 1, max_distance - 1)
        positions = positions + max_distance - 1
        return positions
    
    def forward(self, sequence_length: int) -> torch.Tensor:
        """get relative position bias for sequence length."""
        # get relative position index
        relative_position_index = self._compute_relative_position_index(sequence_length)
        
        # get bias values
        relative_position_bias = self.relative_position_bias_table[relative_position_index]
        
        # reshape to (heads, seq, seq)
        return relative_position_bias.permute(2, 0, 1)


class MultiHeadAttention(nn.Module):
    """multihead attention with relative position bias."""
    
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.1,
        use_relative_pos: bool = True
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = dim_head * heads
        self.use_relative_pos = use_relative_pos
        
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        
        if use_relative_pos:
            self.rel_pos_bias = RelativePositionBias(heads)
        
        self.to_out = nn.Linear(self.inner_dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # scaling factor
        self.scale = 1 / (dim_head ** 0.5)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (b, n, c) or (b, c, h, w)
        original_shape = x.shape
        
        # flatten spatial dimensions if needed
        if x.dim() == 4:
            b, c, h, w = x.shape
            x = rearrange(x, 'b c h w -> b (h w) c')
        else:
            b, n, c = x.shape
        
        # compute qkv
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # reshape for multihead
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)
        
        # compute attention
        attn = torch.einsum('b h n d, b h m d -> b h n m', q, k) * self.scale
        
        # add relative position bias
        if self.use_relative_pos:
            seq_len = n
            rel_bias = self.rel_pos_bias(seq_len)
            attn = attn + rel_bias.unsqueeze(0)  # (1, heads, seq, seq)
        
        # softmax
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # apply to values
        out = torch.einsum('b h n m, b h m d -> b h n d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        # project output
        out = self.to_out(out)
        
        # reshape back if needed
        if len(original_shape) == 4:
            out = rearrange(out, 'b (h w) c -> b c h w', h=original_shape[2])
        
        return out


# lazy import
from typing import Optional