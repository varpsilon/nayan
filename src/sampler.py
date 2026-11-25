"""advanced sampling with learned variance"""
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Callable
from tqdm import tqdm


def cosine_beta_schedule(timesteps: int, s: float = 0.008):
    """cosine schedule for betas."""
    import math
    steps = torch.arange(0, timesteps + 1, dtype=torch.float32)
    f = torch.cos((steps / timesteps + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = f / f[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clamp(betas, min=1e-8, max=0.999)


class Sampler:
    """advanced sampler using learned variance and confidence."""
    
    def __init__(
        self,
        noise_scheduler,
        num_steps: int = 50,
        sampler_type: str = "dpm",
        eta: float = 0.0
    ):
        self.noise_scheduler = noise_scheduler
        self.num_steps = num_steps
        self.sampler_type = sampler_type
        self.eta = eta
        
        self._setup_timesteps()
    
    def _setup_timesteps(self):
        """setup timestep schedule for sampling."""
        if self.sampler_type == "linear":
            self.timesteps = torch.linspace(0, self.num_steps - 1, self.num_steps - 1).long()
        elif self.sampler_type == "cosine":
            # cosine annealing
            alphas = torch.cos(torch.linspace(0, self.num_steps * 1e-4, self.num_steps))**2
            self.timesteps = (alphas[1:] / alphas[:-1]).log().cumsum(0).long()
        elif self.sampler_type == "dpm":
            # DPM-style discretization
            self.timesteps = torch.linspace(0, self.num_steps - 1, self.num_steps - 1).long()
            self.timesteps = torch.flip(self.timesteps, dims=[0])
    
    def p_sample(
        self,
        model,
        x: torch.Tensor,
        t: torch.Tensor,
        t_index: int
    ) -> torch.Tensor:
        """single denoising step."""
        betas = self.noise_scheduler.betas
        alphas_cumprod = self.noise_scheduler.alphas_cumprod
        
        # predict noise
        with torch.no_grad():
            out = model(x, t)
            pred_noise = out['noise']
        
        # get model output
        alpha_t = alphas_cumprod[t].view(-1, 1, 1, 1)
        alpha_t_prev = self._get_alpha(t - 1).view(-1, 1, 1, 1)
        beta_t = betas[t].view(-1, 1, 1, 1)
        
        # compute posterior mean
        # q(x_{t-1} | x_t, x_0) ~ N(mean, beta_tilde * I)
        # where we use predicted x0
        
        # predict x0 from noise
        x0_pred = (x - alpha_t.sqrt() * pred_noise) / (1 - alpha_t).sqrt()
        x0_pred = x0_pred.clamp(-1, 1)
        
        # compute mean
        mean = (
            alpha_t_prev.sqrt() * (1 - alpha_t) / (1 - alpha_t) * x0_pred +
            alpha_t.sqrt() / alpha_t.sqrt() * (1 - alpha_t_prev) / (1 - alpha_t) * x
        )
        
        # add noise
        if t_index > 0:
            # use learned variance if available
            if 'variance' in out:
                log_var = out['variance']
                std = torch.exp(0.5 * log_var) * self.eta
            else:
                # standard posterior variance
                posterior_var = beta_t * (1 - alpha_t_prev) / (1 - alpha_t)
                std = posterior_var.sqrt() * self.eta
            
            noise = torch.randn_like(x)
            return mean + std * noise
        else:
            return mean
    
    def _get_alpha(self, t: torch.Tensor) -> torch.Tensor:
        """get alpha for timestep (with padding for t=-1)."""
        batch_size = t.shape[0]
        result = torch.zeros(batch_size, device=t.device)
        
        valid_mask = t >= 0
        result[valid_mask] = self.noise_scheduler.alphas_cumprod[t[valid_mask]]
        
        return result
    
    @torch.no_grad()
    def sample(
        self,
        model,
        shape: Tuple[int, ...],
        progress: bool = True,
        callback: Optional[Callable] = None
    ) -> torch.Tensor:
        """generate samples."""
        device = next(model.parameters()).device
        x = torch.randn(shape, device=device)
        
        timesteps = self.timesteps.to(device)
        
        iterator = range(len(timesteps) - 1, -1, -1)
        if progress:
            iterator = tqdm(iterator, desc="sampling")
        
        for i in iterator:
            t = torch.full((shape[0],), timesteps[i], device=device, dtype=torch.long)
            
            x = self.p_sample(model, x, t, i)
            
            if callback is not None:
                callback(i, x)
        
        return x


def sample_with_confidence(
    model,
    noise_scheduler,
    shape: Tuple[int, ...],
    num_steps: int = 50,
    guidance_scale: float = 1.0,
    cond: Optional[dict] = None,
    progress: bool = True
) -> torch.Tensor:
    """classifier-free guidance sampling."""
    device = next(model.parameters()).device
    
    x = torch.randn(shape, device=device)
    
    # setup timesteps
    timesteps = torch.linspace(num_steps - 1, 0, num_steps - 1).long().to(device)
    
    for i in tqdm(range(num_steps - 1), desc="guidance sampling", disable=not progress):
        t = torch.full((shape[0],), timesteps[i], device=device, dtype=torch.long)
        
        with torch.no_grad():
            if cond is not None and guidance_scale != 1.0:
                # unconditional prediction
                model_output_uncond = model(x, t)
                noise_uncond = model_output_uncond['noise']
                
                # conditional prediction
                model_output_cond = model(x, t, **cond)
                noise_cond = model_output_cond['noise']
                
                # guidance
                pred_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            else:
                model_output = model(x, t)
                pred_noise = model_output['noise']
        
        # denoise step
        alpha_t = noise_scheduler.alphas_cumprod[t].view(-1, 1, 1, 1)
        alpha_t_prev = noise_scheduler.alphas_cumprod[t - 1].view(-1, 1, 1, 1)
        
        x0_pred = (x - alpha_t.sqrt() * pred_noise) / (1 - alpha_t).sqrt()
        x0_pred = x0_pred.clamp(-1, 1)
        
        if i > 0:
            beta_t = noise_scheduler.betas[t].view(-1, 1, 1, 1)
            posterior_var = beta_t * (1 - alpha_t_prev) / (1 - alpha_t)
            std = posterior_var.sqrt()
            
            noise = torch.randn_like(x)
            x = alpha_t_prev.sqrt() * x0_pred + (1 - alpha_t_prev).sqrt() * noise
        else:
            x = x0_pred
    
    return x