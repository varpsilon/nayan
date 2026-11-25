"""hybrid denoiser: stochastic depth resnet + attention pyramid"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List, Optional

from .timemb import TimeEmbedding
from .resnet import ResBlockWithAttention, Downsample, Upsample
from .layernorm import LayerNorm


class HybridDenoiser(nn.Module):
    """
    hybrid denoiser: learns both noise prediction and variance.
    
    key insight: instead of predicting x0 or epsilon directly,
    we predict (noise, log_variance, confidence) tuple.
    this enables better stochastic sampling.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dim_head: int = 64,
        time_emb_dim: int = 256,
        dropout: float = 0.1,
        stochastic_depth: float = 0.2,
        use_variance_prediction: bool = True,
        use_confidence: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.stochastic_depth = stochastic_depth
        self.use_variance_prediction = use_variance_prediction
        self.use_confidence = use_confidence
        
        self.time_emb = TimeEmbedding(time_emb_dim // 4, hidden_dim=time_emb_dim)
        
        # input projection
        self.input_proj = nn.Conv2d(in_channels, hidden_dim, 1)
        
        # build encoder-decoder pyramid
        self._build_pyramid()
        
        # output heads
        self.to_noise = nn.Conv2d(hidden_dim, in_channels, 1)
        
        if use_variance_prediction:
            self.to_variance = nn.Conv2d(hidden_dim, 1, 1)
        
        if use_confidence:
            self.to_confidence = nn.Conv2d(hidden_dim, 1, 1)
        
        self.norm = LayerNorm(hidden_dim)
    
    def _build_pyramid(self):
        """build encoder-decoder with skip connections."""
        dims = [self.hidden_dim, self.hidden_dim * 2, self.hidden_dim * 4, self.hidden_dim * 8]
        
        # encoder
        self.encoder_blocks = nn.ModuleList()
        self.encoder_downs = nn.ModuleList()
        
        for i in range(self.num_layers):
            self.encoder_blocks.append(
                nn.ModuleList([
                    ResBlockWithAttention(
                        dims[i], dims[i],
                        time_emb_dim=self.hidden_dim,
                        dropout=0.1,
                        use_attention=(i > 0)
                    ),
                    ResBlockWithAttention(
                        dims[i], dims[i],
                        time_emb_dim=self.hidden_dim,
                        dropout=0.1,
                        use_attention=(i > 0)
                    )
                ])
            )
            
            if i < self.num_layers - 1:
                self.encoder_downs.append(Downsample(dims[i], dims[i+1]))
        
        # decoder
        self.decoder_blocks = nn.ModuleList()
        self.decoder_ups = nn.ModuleList()
        
        for i in range(self.num_layers - 1, -1, -1):
            self.decoder_blocks.append(
                nn.ModuleList([
                    ResBlockWithAttention(
                        dims[i] * 2, dims[i],
                        time_emb_dim=self.hidden_dim,
                        dropout=0.1,
                        use_attention=(i > 0)
                    ),
                    ResBlockWithAttention(
                        dims[i], dims[i],
                        time_emb_dim=self.hidden_dim,
                        dropout=0.1,
                        use_attention=(i > 0)
                    )
                ])
            )
            
            if i > 0:
                self.decoder_ups.append(Upsample(dims[i] * 2, dims[i-1]))
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        return_features: bool = False
    ) -> dict:
        """
        forward pass.
        
        returns dict with:
        - noise: predicted noise epsilon
        - variance: predicted log variance (if enabled)
        - confidence: prediction confidence (if enabled)
        - features: intermediate features (if requested)
        """
        # time embedding
        t_emb = self.time_emb(t)
        
        # encode
        features = []
        x = self.input_proj(x)
        
        for i, (blocks, down) in enumerate(zip(self.encoder_blocks, self.encoder_downs)):
            x = blocks[0](x, t_emb)
            x = blocks[1](x, t_emb)
            features.append(x)
            x = down(x)
        
        # bottleneck
        features.append(x)
        
        # decode
        decoded_features = []
        for i, (blocks, up) in enumerate(zip(self.decoder_blocks, self.decoder_ups)):
            # upsample
            x = up(x)
            
            # concatenate skip (reverse order)
            skip = features[-(i+2)]
            x = torch.cat([x, skip], dim=1)
            
            x = blocks[0](x, t_emb)
            x = blocks[1](x, t_emb)
            decoded_features.append(x)
        
        # final processing
        x = self.norm(x)
        
        # output heads
        noise = self.to_noise(x)
        
        out = {'noise': noise}
        
        if self.use_variance_prediction:
            variance = self.to_variance(x)
            out['variance'] = variance
        
        if self.use_confidence:
            confidence = self.to_confidence(x)
            out['confidence'] = confidence
        
        if return_features:
            out['features'] = decoded_features
        
        return out