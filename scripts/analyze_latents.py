#!/usr/bin/env python3
"""latent space exploration - for the curious mind"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from einops import rearrange

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import load_config
from schedule import NoiseScheduler
from denoiser import HybridDenoiser


class LatentExplorer:
    """explore the latent space of your diffusion model."""
    
    def __init__(self, model, noise_scheduler, device='cuda'):
        self.model = model
        self.scheduler = noise_scheduler
        self.device = device
        
        self.model.eval()
    
    @torch.no_grad()
    def denoise_to_latent(self, x: torch.Tensor, num_steps: int = 50, progress: bool = False) -> torch.Tensor:
        """reverse diffusion to get a 'latent' representation."""
        x = x.to(self.device)
        timesteps = torch.linspace(num_steps - 1, 0, num_steps - 1).long().to(self.device)
        
        for t in tqdm(timesteps, disable=not progress, desc="reverse diffusing"):
            t_batch = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)
            
            with torch.no_grad():
                out = self.model(x, t_batch)
                pred_noise = out['noise']
            
            alpha_t = self.scheduler.alphas_cumprod[t_batch].view(-1, 1, 1, 1)
            alpha_t_prev = self.scheduler.alphas_cumprod[t_batch - 1].view(-1, 1, 1, 1)
            
            x0_pred = (x - alpha_t.sqrt() * pred_noise) / (1 - alpha_t).sqrt()
            x0_pred = x0_pred.clamp(-1, 1)
            
            if t > 0:
                beta_t = self.scheduler.betas[t_batch].view(-1, 1, 1, 1)
                posterior_var = beta_t * (1 - alpha_t_prev) / (1 - alpha_t)
                std = posterior_var.sqrt()
                
                noise = torch.randn_like(x)
                x = alpha_t_prev.sqrt() * x0_pred + (1 - alpha_t_prev).sqrt() * noise
            else:
                x = x0_pred
        
        return x
    
    def interpolate(self, img1: torch.Tensor, img2: torch.Tensor, n_steps: int = 8) -> list:
        """interpolate between two images in latent space."""
        l1 = self.denoise_to_latent(img1, num_steps=20)
        l2 = self.denoise_to_latent(img2, num_steps=20)
        
        results = []
        for t in torch.linspace(0, 1, n_steps):
            latent = (1 - t) * l1 + t * l2
            
            # re-diffuse
            x = latent.clone()
            timesteps = torch.linspace(39, 0, 40).long().to(self.device)
            
            for ts in timesteps:
                t_batch = torch.full((x.shape[0],), ts, device=self.device, dtype=torch.long)
                out = self.model(x, t_batch)
                pred_noise = out['noise']
                
                alpha_t = self.scheduler.alphas_cumprod[t_batch].view(-1, 1, 1, 1)
                alpha_t_prev = self.scheduler.alphas_cumprod[t_batch - 1].view(-1, 1, 1, 1)
                
                x0_pred = (x - alpha_t.sqrt() * pred_noise) / (1 - alpha_t).sqrt()
                x0_pred = x0_pred.clamp(-1, 1)
                
                if ts > 0:
                    beta_t = self.scheduler.betas[t_batch].view(-1, 1, 1, 1)
                    posterior_var = beta_t * (1 - alpha_t_prev) / (1 - alpha_t)
                    std = posterior_var.sqrt()
                    noise = torch.randn_like(x)
                    x = alpha_t_prev.sqrt() * x0_pred + (1 - alpha_t_prev).sqrt() * noise
                else:
                    x = x0_pred
            
            results.append(x.cpu())
        
        return results
    
    def analogy(self, a1: torch.Tensor, a2: torch.Tensor, b1: torch.Tensor) -> torch.Tensor:
        """vector arithmetic in latent space: a2 - a1 + b1."""
        l_a1 = self.denoise_to_latent(a1)
        l_a2 = self.denoise_to_latent(a2)
        l_b1 = self.denoise_to_latent(b1)
        
        # analogy vector
        vec = l_a2 - l_a1
        result_latent = l_b1 + vec
        
        # re-diffuse
        x = result_latent.clone()
        timesteps = torch.linspace(39, 0, 40).long().to(self.device)
        
        for ts in timesteps:
            t_batch = torch.full((x.shape[0],), ts, device=self.device, dtype=torch.long)
            out = self.model(x, t_batch)
            pred_noise = out['noise']
            
            alpha_t = self.scheduler.alphas_cumprod[t_batch].view(-1, 1, 1, 1)
            alpha_t_prev = self.scheduler.alphas_cumprod[t_batch - 1].view(-1, 1, 1, 1)
            
            x0_pred = (x - alpha_t.sqrt() * pred_noise) / (1 - alpha_t).sqrt()
            x0_pred = x0_pred.clamp(-1, 1)
            
            if ts > 0:
                beta_t = self.scheduler.betas[t_batch].view(-1, 1, 1, 1)
                posterior_var = beta_t * (1 - alpha_t_prev) / (1 - alpha_t)
                std = posterior_var.sqrt()
                noise = torch.randn_like(x)
                x = alpha_t_prev.sqrt() * x0_pred + (1 - alpha_t_prev).sqrt() * noise
            else:
                x = x0_pred
        
        return x.cpu()


def main():
    parser = argparse.ArgumentParser(description='explore latent space')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--img1', type=str, help='first image for interpolation')
    parser.add_argument('--img2', type=str, help='second image for interpolation')
    parser.add_argument('--output', type=str, default='latent_grid.png')
    
    args = parser.parse_args()
    
    # load config and model
    config = load_config('configs/base.yaml')
    model = HybridDenoiser(
        in_channels=config.model.channels,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        dim_head=config.model.dim_head,
        time_emb_dim=config.model.time_emb_dim
    )
    
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['ema_state_dict'] if 'ema_state_dict' in ckpt else ckpt['model_state_dict'])
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    scheduler = NoiseScheduler(1000)
    explorer = LatentExplorer(model, scheduler, device)
    
    # interpolation
    if args.img1 and args.img2:
        img1 = torch.from_numpy(
            np.array(Image.open(args.img1).resize((64, 64))) / 127.5 - 1
        ).permute(2, 0, 1).unsqueeze(0).float()
        img2 = torch.from_numpy(
            np.array(Image.open(args.img2).resize((64, 64))) / 127.5 - 1
        ).permute(2, 0, 1).unsqueeze(0).float()
        
        results = explorer.interpolate(img1, img2, n_steps=8)
        
        # make grid
        grid = torch.cat(results, dim=3)
        grid = (grid + 1) / 2
        grid = grid.clamp(0, 1)
        grid_np = grid.permute(1, 2, 0).numpy()
        Image.fromarray((grid_np * 255).astype('uint8')).save(args.output)
        print(f"saved interpolation grid to {args.output}")


if __name__ == '__main__':
    main()