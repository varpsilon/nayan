"""loss functions for diffusion training"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


def normal_kl(mean0: torch.Tensor, std0: torch.Tensor, mean1: torch.Tensor, std1: torch.Tensor) -> torch.Tensor:
    """kl divergence between two normal distributions."""
    kl = torch.log(std1 / std0) + (std0 ** 2 + (mean0 - mean1) ** 2) / (2 * std1 ** 2) - 0.5
    return kl.sum(dim=[1, 2, 3]).mean()


def vae_loss(
    x: torch.Tensor,
    pred_noise: torch.Tensor,
    noise_scheduler,
    t: torch.Tensor,
    pred_variance: Optional[torch.Tensor] = None,
    pred_confidence: Optional[torch.Tensor] = None,
    confidence_weight: float = 0.1,
    variance_weight: float = 0.1
) -> torch.Tensor:
    """
    hybrid loss for diffusion training.
    
    combines:
    1. mse for noise prediction
    2. variational loss for variance prediction
    3. confidence-aware weighting
    """
    batch_size = x.shape[0]
    
    # sample noise and add to clean image
    noise = torch.randn_like(x)
    x_noisy = noise_scheduler.q_sample(x, t, noise)
    
    # main mse loss for noise prediction
    mse_loss = F.mse_loss(pred_noise, noise)
    
    total_loss = mse_loss
    
    # variance loss (optional)
    if pred_variance is not None:
        # predict variance of the denoising distribution
        # target is the true posterior variance
        log_var = pred_variance.squeeze(1)  # (b, h, w)
        
        # simple variance fitting loss
        var_target = torch.zeros_like(log_var)  # placeholder
        var_loss = F.mse_loss(log_var, var_target)
        
        total_loss = total_loss + variance_weight * var_loss
    
    # confidence loss (optional) - encourages model to know what it knows
    if pred_confidence is not None:
        # confidence should be high where prediction is good
        # use gradient-based measure of prediction quality
        confidence = pred_confidence.squeeze(1)
        
        # simple supervision: confidence should correlate with -mse
        target_conf = -mse_loss.detach() * torch.ones_like(confidence)
        conf_loss = F.mse_loss(confidence, target_conf)
        
        total_loss = total_loss + confidence_weight * conf_loss
    
    return total_loss


class DiffusionLoss(nn.Module):
    """trainable loss module."""
    
    def __init__(
        self,
        confidence_weight: float = 0.1,
        variance_weight: float = 0.1,
        timesteps: int = 1000
    ):
        super().__init__()
        self.confidence_weight = confidence_weight
        self.variance_weight = variance_weight
        self.timesteps = timesteps
    
    def forward(
        self,
        pred_dict: Dict[str, torch.Tensor],
        noise_scheduler,
        x: torch.Tensor,
        t: torch.Tensor,
        x0: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """compute training loss."""
        pred_noise = pred_dict['noise']
        
        if 'variance' in pred_dict and 'confidence' in pred_dict:
            loss = vae_loss(
                x, pred_noise, noise_scheduler, t,
                pred_variance=pred_dict['variance'],
                pred_confidence=pred_dict['confidence'],
                confidence_weight=self.confidence_weight,
                variance_weight=self.variance_weight
            )
        else:
            loss = vae_loss(x, pred_noise, noise_scheduler, t)
        
        # compute metrics for logging
        with torch.no_grad():
            noise = torch.randn_like(x)
            x_noisy = noise_scheduler.q_sample(x, t, noise)
            mse = F.mse_loss(pred_noise, noise).item()
        
        metrics = {'mse': mse}
        
        return loss, metrics