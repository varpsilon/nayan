#!/usr/bin/env python3
"""sample script - generate images from trained model"""
import argparse
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import load_config
from schedule import NoiseScheduler
from denoiser import HybridDenoiser
from sampler import sample_with_confidence


def build_model(config, checkpoint_path=None):
    """build and load model."""
    model = HybridDenoiser(
        in_channels=config.model.channels,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        dim_head=config.model.dim_head,
        time_emb_dim=config.model.time_emb_dim,
        dropout=config.model.dropout,
        use_variance_prediction=True,
        use_confidence=True
    )
    
    if checkpoint_path:
        print(f"loading checkpoint from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        
        if 'ema_state_dict' in ckpt:
            model.load_state_dict(ckpt['ema_state_dict'])
            print("using ema weights")
        else:
            model.load_state_dict(ckpt['model_state_dict'])
    
    return model


def interpolate_latents(z1, z2, t: float) -> torch.Tensor:
    """linear interpolation in latent space."""
    return (1 - t) * z1 + t * z2


def generate_grid(
    model,
    noise_scheduler,
    n_rows: int = 4,
    n_cols: int = 4,
    guidance_scale: float = 0.0,
    device: str = 'cuda'
) -> torch.Tensor:
    """generate a grid of samples."""
    model.eval()
    n = n_rows * n_cols
    
    sampler = NoiseScheduler(1000)  # dummy scheduler for sampling
    
    samples = []
    
    with torch.no_grad():
        for i in range(n_rows):
            row = []
            for j in range(n_cols):
                sample = sample_with_confidence(
                    model, noise_scheduler,
                    (1, 3, 64, 64),
                    num_steps=50,
                    guidance_scale=guidance_scale,
                    progress=False
                )
                row.append(sample)
            samples.append(torch.cat(row, dim=3))
    
    model.train()
    return torch.cat(samples, dim=2)


def main():
    parser = argparse.ArgumentParser(description='sample from diffuser')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='model checkpoint path')
    parser.add_argument('--config', type=str, default='configs/base.yaml',
                        help='config file')
    parser.add_argument('--output', type=str, default='samples.png',
                        help='output image path')
    parser.add_argument('--n', type=int, default=16,
                        help='number of samples')
    parser.add_argument('--guidance', type=float, default=0.0,
                        help='classifier-free guidance scale')
    parser.add_argument('--steps', type=int, default=50,
                        help='number of denoising steps')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed')
    
    args = parser.parse_args()
    
    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # load config and model
    config = load_config(args.config)
    model = build_model(config, args.checkpoint)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # build scheduler for sampling
    scheduler = NoiseScheduler(
        timesteps=config.sampling.num_steps,
        noise_type=config.sampling.noise_schedule
    )
    
    # generate samples
    print(f"generating {args.n} samples...")
    samples = []
    
    with torch.no_grad():
        for i in range(args.n):
            sample = sample_with_confidence(
                model, scheduler,
                (1, 3, config.model.image_size, config.model.image_size),
                num_steps=50,
                guidance_scale=args.guidance,
                progress=True
            )
            samples.append(sample)
    
    samples = torch.cat(samples, dim=0)
    
    # save as grid
    grid = torch.cat([
        torch.cat(samples[i:i+4], dim=2) 
        for i in range(0, len(samples), 4)
    ], dim=1)
    
    # denormalize and convert to uint8
    grid = (grid + 1) / 2  # [-1, 1] -> [0, 1]
    grid = grid.clamp(0, 1)
    grid = (grid * 255).byte()
    
    # to PIL
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    img = Image.fromarray(grid_np)
    img.save(args.output)
    
    print(f"saved to {args.output}")


if __name__ == '__main__':
    main()