<div align="center">

# 🌫️ diffuser

**a from-scratch diffusion model that learns to denoise images by predicting not just noise, but also its uncertainty and confidence**

[![CI](https://img.shields.io/badge/build-passing-brightgreen?style=flat-square)]()
[![license](https://img.shields.io/badge/license-MIT-blue?style=flat-square)]()
[![Python](https://img.shields.io/badge/python-3.9%2B-blue?style=flat-square)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=flat-square)]()

</div>

## Overview

diffuser is a hands-on implementation of a diffusion probabilistic model built entirely from scratch. Unlike black-box frameworks, every component is implemented with transparency and research-first principles.

The model uses a hybrid architecture where the U-Net denoiser simultaneously predicts:
- **ε** - the noise added to the image
- **logσ** - the learned variance of the denoising distribution  
- **c** - a confidence score indicating prediction reliability

This enables more informed sampling and better uncertainty quantification.

## Key Features

**hybrid denoiser** - predicts noise + variance + confidence in a single forward pass

**learned time embeddings** - sinusoidal positional encodings with learnable scaling

**relative position bias** - attention that understands spatial relationships

**vae-style training** - combines mse loss with uncertainty-aware objectives

**advanced sampling** - cosine schedule with classifier-free guidance

**full logging** - wandb + tensorboard integration with EMA weights

## Installation

```bash
git clone https://github.com/varpsilon/nayan
cd nayan
pip install -r requirements.txt
```

## Training

```bash
# basic training
python src/train.py --config configs/base.yaml

# with custom settings
python src/train.py --config configs/base.yaml --steps 100000 --lr 5e-5
```

Training config controls:
- `image_size`: 64, 128, or 256
- `hidden_dim`: model capacity (64, 128, 256)
- `num_layers`: depth of the network
- `batch_size`: training batch size
- `learning_rate`: optimiser learning rate

## Sampling

```bash
# generate 16 samples
python scripts/sample.py --checkpoint checkpoints/final.pt --output samples.png --n 16

# with guidance
python scripts/sample.py --guidance 1.5 --checkpoint checkpoints/final.pt
```

## Latent Space Exploration

```bash
# interpolate between two images
python scripts/analyze_latents.py --img1 a.png --img2 b.png --output interp.png
```

## Project Structure

```
diffuser/
├── src/*.py           # core implementation
│   ├── config.py      # dataclasses + yaml loader
│   ├── schedule.py    # linear/cosine noise schedules
│   ├── denoiser.py    # hybrid u-net model
│   ├── loss.py        # vae-style hybrid loss
│   ├── sampler.py     # advanced diffusion sampler
│   └── train.py       # entry point
├── configs/base.yaml  # model + training configuration
├── scripts/           # utilities
├── tests/             # pytest suite
├── Dockerfile         # container build
└── README.md
```

## License

MIT — feel free to experiment, modify, and build upon this code.