"""trainer with ema, lr scheduling, and cool logging"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import wandb
from pathlib import Path
from typing import Optional, Dict, Tuple
from datetime import datetime
import json
import os
import math

from .loss import DiffusionLoss
from .schedule import NoiseScheduler
from .sampler import Sampler


class EMA:
    """exponential moving average of model weights."""
    
    def __init__(self, model: nn.Module, decay: float = 0.999, device: str = 'cuda'):
        self.decay = decay
        self.device = device
        
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(device)
    
    def update(self, model: nn.Module):
        """update shadow weights."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = (1 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self, model: nn.Module):
        """apply shadow weights to model."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()
    
    def restore(self, model: nn.Module):
        """restore original weights."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name].clone()


class Trainer:
    """trainer with all the cool stuff."""
    
    def __init__(
        self,
        model: nn.Module,
        noise_scheduler: NoiseScheduler,
        loss_fn: DiffusionLoss,
        config,
        device: str = 'cuda',
        use_amp: bool = True,
        use_ema: bool = True,
        ema_decay: float = 0.999
    ):
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.loss_fn = loss_fn
        self.config = config
        self.device = device
        self.use_amp = use_amp and device != 'cpu'
        
        # optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            betas=config.optimizer.betas
        )
        
        # lr scheduler
        self.scheduler = self._get_scheduler()
        
        # ema
        self.use_ema = use_ema
        if use_ema:
            self.ema = EMA(model, decay=ema_decay, device=device)
        
        # amp
        self.scaler = GradScaler() if use_amp else None
        
        # logging
        self.writer = None
        self.wandb_run = None
        self._setup_logging()
        
        # state
        self.step = 0
        self.epoch = 0
    
    def _get_scheduler(self):
        """get learning rate scheduler."""
        num_steps = self.config.training.num_steps
        
        if self.config.training.lr_schedule == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=num_steps, eta_min=self.config.training.learning_rate * 0.01
            )
        elif self.config.training.lr_schedule == 'constant':
            return optim.lr_scheduler.ConstantLR(
                self.optimizer, factor=1.0, total_iters=num_steps
            )
        else:
            return optim.lr_scheduler.PolynomialDecay(
                self.optimizer, total_steps=num_steps, power=1.0
            )
    
    def _setup_logging(self):
        """setup tensorboard and wandb."""
        log_dir = Path(self.config.logging.checkpoint_dir) / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(str(log_dir / f'session_{datetime.now().strftime("%Y%m%d_%H%M%S")}'))
        
        # wandb
        if os.getenv('WANDB_API_KEY'):
            wandb.init(
                project='diffuser',
                name=f'diffuser-{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                config=self._config_to_dict()
            )
            self.wandb_run = wandb.run
    
    def _config_to_dict(self) -> dict:
        """convert config to dict for wandb."""
        return {
            'model': {
                'image_size': self.config.model.image_size,
                'hidden_dim': self.config.model.hidden_dim,
                'num_layers': self.config.model.num_layers,
            },
            'training': {
                'batch_size': self.config.training.batch_size,
                'lr': self.config.training.learning_rate,
            }
        }
    
    def train_step(self, batch: Tuple[torch.Tensor, dict]) -> Dict[str, float]:
        """single training step."""
        x = batch[0].to(self.device)
        
        # sample random timesteps
        t = torch.randint(0, self.config.training.num_timesteps, (x.shape[0],), device=self.device)
        
        # forward pass with amp
        with autocast(enabled=self.use_amp):
            out = self.model(x, t)
            loss, metrics = self.loss_fn(out, self.noise_scheduler, x, t)
        
        # backward
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        
        # ema update
        if self.use_ema:
            self.ema.update(self.model)
        
        # zero grad
        self.optimizer.zero_grad()
        
        # scheduler step
        self.scheduler.step()
        self.step += 1
        
        return metrics
    
    def log_metrics(self, metrics: dict, prefix: str = 'train'):
        """log metrics to tensorboard and wandb."""
        for key, value in metrics.items():
            self.writer.add_scalar(f'{prefix}/{key}', value, self.step)
        
        if self.wandb_run:
            wandb.log(metrics, step=self.step)
    
    def validate(self, val_loader) -> Dict[str, float]:
        """run validation."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= 10:  # only 10 batches
                    break
                
                x = batch[0].to(self.device)
                t = torch.randint(0, self.config.training.num_timesteps, (x.shape[0],), device=self.device)
                
                with autocast(enabled=self.use_amp):
                    out = self.model(x, t)
                    loss, _ = self.loss_fn(out, self.noise_scheduler, x, t)
                
                total_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        return {'val_loss': total_loss / num_batches}
    
    def save_checkpoint(self, name: Optional[str] = None):
        """save model checkpoint."""
        save_dir = Path(self.config.logging.checkpoint_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if name is None:
            name = f'step_{self.step}'
        
        ckpt = {
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        
        if self.use_ema:
            ckpt['ema_state_dict'] = self.ema.shadow
        
        torch.save(ckpt, save_dir / f'{name}.pt')
    
    def load_checkpoint(self, path: str):
        """load model checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if 'ema_state_dict' in ckpt and self.use_ema:
            self.ema.shadow = ckpt['ema_state_dict']
    
    def sample(self, n: int = 16) -> torch.Tensor:
        """sample n images."""
        self.model.eval()
        sampler = Sampler(self.noise_scheduler, num_steps=self.config.sampling.num_steps)
        
        with torch.no_grad():
            samples = sampler.sample(
                self.model if not self.use_ema else None,
                (n, 3, self.config.model.image_size, self.config.model.image_size),
                progress=True
            )
        
        self.model.train()
        return samples


def train(
    model: nn.Module,
    noise_scheduler: NoiseScheduler,
    loss_fn: DiffusionLoss,
    config,
    train_loader,
    val_loader: Optional[DataLoader] = None,
    device: str = 'cuda',
    max_steps: Optional[int] = None
) -> Trainer:
    """main training loop."""
    from torch.utils.data import DataLoader
    
    trainer = Trainer(
        model, noise_scheduler, loss_fn, config, device=device
    )
    
    model.train()
    
    print(f"training for {max_steps or config.training.num_steps} steps...")
    
    pbar = tqdm(range(max_steps or config.training.num_steps))
    
    for step in pbar:
        # get batch (cycle through loader)
        batch = next(iter(train_loader))
        
        # train step
        metrics = trainer.train_step(batch)
        trainer.step = step
        
        # log
        if step % config.logging.log_every == 0:
            trainer.log_metrics(metrics)
            pbar.set_description(f"loss: {metrics['mse']:.4f}")
        
        # validate
        if val_loader and step % config.logging.val_every == 0:
            val_metrics = trainer.validate(val_loader)
            trainer.log_metrics(val_metrics, prefix='val')
        
        # save samples
        if step > 0 and step % config.logging.save_every == 0:
            trainer.save_checkpoint()
            # sample and log
            samples = trainer.sample(4)
            grid = torchvision.utils.make_grid(samples, nrow=2, normalize=True)
            trainer.writer.add_image('samples', grid, step)
            if trainer.wandb_run:
                wandb.log({'samples': wandb.Image(grid)}, step=step)
        
        # early stop
        if max_steps and step >= max_steps - 1:
            break
    
    return trainer


# lazy import
from .data import DataLoader
import torchvision.utils