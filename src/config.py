"""config loader"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import yaml
from omegaconf import OmegaConf


@dataclass
class ModelConfig:
    channels: int = 3
    image_size: int = 64
    hidden_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    dim_head: int = 64
    ff_mult: int = 4
    time_emb_dim: int = 256
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_steps: int = 1000000
    lr_schedule: str = "cosine"
    noise_type: str = "linear"
    num_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02


@dataclass
class SamplingConfig:
    num_steps: int = 50
    noise_schedule: str = "cosine"


@dataclass
class DataConfig:
    name: str = "lavender"
    path: str = "data/lavender"
    num_workers: int = 4


@dataclass
class LoggingConfig:
    log_every: int = 100
    val_every: int = 1000
    save_every: int = 5000
    checkpoint_dir: str = "checkpoints/"
    output_dir: str = "outputs/"


@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    sampling: SamplingConfig
    data: DataConfig
    logging: LoggingConfig


def load_config(path: str) -> Config:
    """load and validate config from yaml file."""
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)
    
    cfg = OmegaConf.create(raw)
    
    return Config(
        model=ModelConfig(**cfg.model),
        training=TrainingConfig(**cfg.training),
        sampling=SamplingConfig(**cfg.sampling),
        data=DataConfig(**cfg.data),
        logging=LoggingConfig(**cfg.logging)
    )