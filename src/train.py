#!/usr/bin/env python3
"""main training script - the entry point"""
import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

# add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import load_config
from schedule import NoiseScheduler
from denoiser import HybridDenoiser
from loss import DiffusionLoss
from data import get_dataloader
from trainer import Trainer


def build_model(config):
    """build model from config."""
    model = HybridDenoiser(
        in_channels=config.model.channels,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        dim_head=config.model.dim_head,
        time_emb_dim=config.model.time_emb_dim,
        dropout=config.model.dropout,
        stochastic_depth=0.2,
        use_variance_prediction=True,
        use_confidence=True
    )
    return model


def build_scheduler(config):
    """build noise scheduler from config."""
    scheduler = NoiseScheduler(
        timesteps=config.training.num_timesteps,
        noise_type=config.training.noise_type,
        beta_start=config.training.beta_start,
        beta_end=config.training.beta_end
    )
    return scheduler


def build_loss_fn(config):
    """build loss function."""
    loss_fn = DiffusionLoss(
        confidence_weight=0.1,
        variance_weight=0.1,
        timesteps=config.training.num_timesteps
    )
    return loss_fn


def get_device():
    """get the best available device."""
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def main():
    parser = argparse.ArgumentParser(description='train diffuser')
    parser.add_argument('--config', type=str, default='configs/base.yaml',
                        help='path to config file')
    parser.add_argument('--device', type=str, default=None,
                        help='device to train on')
    parser.add_argument('--steps', type=int, default=None,
                        help='override max training steps')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='override batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='override learning rate')
    
    args = parser.parse_args()
    
    # load config
    print(f"loading config from {args.config}...")
    config = load_config(args.config)
    
    # override config
    if args.steps:
        config.training.num_steps = args.steps
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    
    # device
    device = args.device or get_device()
    print(f"training on {device}...")
    
    # build components
    print("building model...")
    model = build_model(config)
    model = model.to(device)
    
    print(f"model params: {sum(p.numel() for p in model.parameters()):,} tokens")
    
    print("building noise scheduler...")
    scheduler = build_scheduler(config)
    
    print("building loss function...")
    loss_fn = build_loss_fn(config)
    
    print("building data loader...")
    train_loader = get_dataloader(config, augment=True, num_workers=config.data.num_workers)
    val_loader = None  # add validation loader later
    
    if args.steps is None:
        args.steps = config.training.num_steps
    
    # create output dirs
    Path(config.logging.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.logging.output_dir).mkdir(parents=True, exist_ok=True)
    
    # train
    trainer = Trainer(
        model, scheduler, loss_fn, config, device=device
    )
    
    model.train()
    
    from tqdm import tqdm
    
    print(f"\ntraining for {args.steps} steps...")
    pbar = tqdm(range(args.steps))
    
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
        
        # save checkpoint
        if step > 0 and step % config.logging.save_every == 0:
            trainer.save_checkpoint()
            print(f"\ncheckpoint saved at step {step}")
        
        # early stop
        if step >= args.steps - 1:
            break
    
    # final save
    trainer.save_checkpoint('final')
    print("\ntraining complete!")
    
    # generate final samples
    print("generating samples...")
    samples = trainer.sample(16)
    
    save_path = Path(config.logging.output_dir) / f'samples_step_{args.steps}.pt'
    torch.save(samples, save_path)
    print(f"samples saved to {save_path}")


if __name__ == '__main__':
    main()