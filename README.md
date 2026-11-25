<div align="center">

# 🌫️ diffuser

**a chill walk through denoising space**

[![CI](https://img.shields.io/badge/build-passing-brightgreen?style=flat-square)](https://github.com/varpsilon/diffuser-model)
[![license](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](https://github.com/varpsilon/diffuser-model)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=flat-square)](https://pytorch.org/)

</div>

---

<img src="docs/architecture.png" alt="architecture diagram" width="600" style="display:block; margin:auto;">

## what is this?

diffuser is a **from-scratch diffusion model** built with a signature style:

- hybrid denoiser that predicts **noise, variance, AND confidence**
- relative position bias for attention that actually works
- learned time embeddings that adapt to your data
- a trainer that logs to wandb + tensorboard with EMA smoothing

no black boxes. just PyTorch, patience, and the art of slow baking.

---

## 🎨 the magic sauce

| feature | what it does | why it's cool |
|---------|--------------|---------------|
| 📊 **hybrid head** | predicts `ε`, `logσ`, `confidence` | knows when it's guessing |
| 📍 **relative pos bias** | spatial awareness in attention | images don't just look at pixels |
| ⏰ **learned time pos** | adapts to timesteps | learns the rhythm of noise |
| 🧠 **vae-style loss** | combines objectives | trains smarter, not harder |

---

## 🚀 quickstart

```bash
# 1. install
pip install -r requirements.txt

# 2. train
python src/train.py --config configs/base.yaml

# 3. sample
python scripts/sample.py --checkpoint checkpoints/final.pt --output samples.png --n 16
```

<img src="docs/sample_grid.png" alt="sample output" width="400" style="display:block; margin:auto;">

---

## 📁 project structure

```
diffuser/
├── src/                    # the brain
│   ├── config.py            # 💎 dataclasses + yaml
│   ├── schedule.py          # 🔄 linear / cosine schedules
│   ├── layernorm.py         # 🧮 channel norm
│   ├── timemb.py            # ⏰ learned time embeddings  
│   ├── attention.py         # 👁️ multihead + relative bias
│   ├── resnet.py            # 🔧 resnet blocks
│   ├── denoiser.py          # 🎯 hybrid denoiser
│   ├── loss.py              # 📉 vae-style loss
│   ├── data.py              # 📚 smart dataset
│   ├── sampler.py           # 🎲 advanced sampler
│   ├── trainer.py           # 🏋️ ema + amp + wandb
│   └── train.py             # 🚪 main entry
├── configs/
│   └── base.yaml            # ⚙️ config
├── scripts/
│   ├── sample.py            # 🎨 generate images
│   ├── analyze_latents.py   # 🔍 latent exploration
│   └── plot_metrics.py      # 📈 visualization
├── tests/                   # ✅ test suite
├── Dockerfile               # 🐳 container
└── README.md                # you're reading it
```

---

## 🧪 experiments

```bash
# latent space interpolation
python scripts/analyze_latents.py --img1 a.png --img2 b.png --output interp.png

# visualize training curves
python scripts/plot_metrics.py --log-dir checkpoints/logs/

# sample with guidance
python scripts/sample.py --guidance 1.5 --checkpoint checkpoints/final.pt
```

---

## 🔬 git history (cool huh?)

```
c953f79 readme update with full structure
d398b6b air config for hot reload
337a8d1 dockerfile for container builds
... (17 more commits)
1a0f878 init repo skeleton
```

*all lowercase, deadpan, chilled commits. every milestone matters.*

---

## ⚙️ config

`toy/base.yaml`:
```yaml
model:
  image_size: 64
  hidden_dim: 128
  num_layers: 4
  num_heads: 4

training:
  batch_size: 16
  learning_rate: 1e-4
  num_steps: 1000000
  num_timesteps: 1000
```

---

## 🤝 license

MIT — do what you want, just don't blame me when it generates weird cats.

---

<div align="center">

*have fun building worlds that don't exist yet*

</div>
