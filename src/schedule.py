"""noise schedule utilities"""
import math
import torch
from typing import Tuple


def linear_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> Tuple[torch.Tensor, torch.Tensor]:
    """linear noise schedule."""
    betas = torch.linspace(beta_start, beta_end, timesteps)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas_cumprod


def cosine_schedule(timesteps: int, s: float = 0.008) -> Tuple[torch.Tensor, torch.Tensor]:
    """cosine noise schedule from ddpm."""
    # f(t) = cos^2((t/T + s) / (1 + s) * pi/2)
    steps = torch.arange(0, timesteps + 1, dtype=torch.float32)
    f = torch.cos((steps / timesteps + s) / (1 + s) * math.pi / 2) ** 2
    
    alphas_cumprod = f / f[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = torch.clamp(betas, min=1e-8, max=0.999)
    
    alphas_cumprod = alphas_cumprod[1:]
    return betas, alphas_cumprod


def get_noise_schedule(timesteps: int, noise_type: str = "linear", **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """dispatch to appropriate schedule."""
    if noise_type == "linear":
        return linear_schedule(timesteps, **kwargs)
    elif noise_type == "cosine":
        return cosine_schedule(timesteps, **kwargs)
    else:
        raise ValueError(f"unknown noise schedule: {noise_type}")


class NoiseScheduler:
    """handles forward and reverse diffusion process."""
    
    def __init__(self, timesteps: int, noise_type: str = "linear", **kwargs):
        self.timesteps = timesteps
        self.noise_type = noise_type
        self.betas, self.alphas_cumprod = get_noise_schedule(timesteps, noise_type, **kwargs)
        
        self.alphas = 1 - self.betas
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # precompute coefficients for q sample
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        
        # for posterior
        self.posterior_variance = (
            self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        )
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """forward diffusion process: add noise to clean image."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # xt = sqrt(alpha_t) * x0 + sqrt(1 - alpha_t) * noise
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def posterior_mean(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """compute mean of q(x_{t-1} | x_t, x0)."""
        # this is simplified - full version needs x0 prediction
        return x  # placeholder
    
    def get_posterior(self, x_start: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """compute posterior distribution q(x_{t-1} | x_t, x0)."""
        # implementation in terms of x_start
        mean = (
            self.betas[t] * torch.sqrt(self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])) * x_start +
            torch.sqrt(self.alphas_cumprod_prev[t]) / torch.sqrt(self.alphas_cumprod[t]) * 
            (1 - self.alphas_cumprod[t]) / (1 - self.alphas_cumprod_prev[t]) * x_start
        )
        variance = self.posterior_variance[t]
        
        return mean, variance


# lazy import for optional
import math
from typing import Optional