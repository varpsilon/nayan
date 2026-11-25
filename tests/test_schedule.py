"""test noise schedule"""
import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from schedule import NoiseScheduler, linear_schedule, cosine_schedule


def test_linear_schedule():
    """test linear noise schedule."""
    betas, alphas_cumprod = linear_schedule(100)
    
    assert len(betas) == 100
    assert len(alphas_cumprod) == 100
    assert betas[0] < betas[-1]
    assert torch.all(alphas_cumprod >= 0)
    assert torch.all(alphas_cumprod <= 1)


def test_cosine_schedule():
    """test cosine noise schedule."""
    betas, alphas_cumprod = cosine_schedule(100)
    
    assert len(betas) == 100
    assert len(alphas_cumprod) == 100
    assert torch.all(betas >= 0) and torch.all(betas <= 1)


def test_noise_scheduler_q_sample():
    """test forward diffusion process."""
    scheduler = NoiseScheduler(100)
    
    x0 = torch.randn(4, 3, 32, 32)
    t = torch.tensor([50] * 4)
    
    xt = scheduler.q_sample(x0, t)
    
    assert xt.shape == x0.shape
    # xt should be between x0 and noise
    assert not torch.allclose(xt, x0)


def test_noise_scheduler_get_posterior():
    """test posterior computation."""
    scheduler = NoiseScheduler(100)
    
    x0 = torch.randn(2, 3, 32, 32)
    t = torch.tensor([50] * 2)
    
    mean, var = scheduler.get_posterior(x0, t)
    
    assert mean.shape == x0.shape
    assert var.shape == (2,)