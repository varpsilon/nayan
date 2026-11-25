"""test denoiser model"""
import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from denoiser import HybridDenoiser


@pytest.mark.parametrize("hidden_dim", [64, 128])
@pytest.mark.parametrize("num_layers", [2, 3])
def test_denoiser_forward(hidden_dim, num_layers):
    """test model forward pass."""
    model = HybridDenoiser(
        in_channels=3,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=4,
        time_emb_dim=128,
        use_variance_prediction=True,
        use_confidence=True
    )
    
    x = torch.randn(2, 3, 32, 32)
    t = torch.tensor([50] * 2)
    
    out = model(x, t)
    
    assert 'noise' in out
    assert out['noise'].shape == x.shape
    assert 'variance' in out
    assert out['variance'].shape == (2, 1, 32, 32)
    assert 'confidence' in out
    assert out['confidence'].shape == (2, 1, 32, 32)


def test_denoiser_different_timesteps():
    """test model handles different timesteps."""
    model = HybridDenoiser(
        in_channels=3,
        hidden_dim=64,
        num_layers=2,
        num_heads=2,
        time_emb_dim=128
    )
    
    x = torch.randn(2, 3, 32, 32)
    
    # different timesteps
    t1 = torch.tensor([0] * 2)
    t2 = torch.tensor([100] * 2)
    
    out1 = model(x, t1)
    out2 = model(x, t2)
    
    # outputs should be different for different timesteps
    assert not torch.allclose(out1['noise'], out2['noise'])