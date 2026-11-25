<div align="center">

# 🌫️ diffuser

**a from-scratch diffusion model that learns to denoise images by predicting not just noise, but also its uncertainty and confidence**

[![CI](https://img.shields.io/badge/build-passing-brightgreen?style=flat-square)]()
[![license](https://img.shields.io/badge/license-MIT-blue?style=flat-square)]()
[![Python](https://img.shields.io/badge/python-3.9%2B-blue?style=flat-square)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=flat-square)]()

</div>

---

<img src="docs/architecture.png" alt="architecture" width="500" style="display:block; margin:auto;">

## quickstart

```bash
pip install -r requirements.txt
python src/train.py --config configs/base.yaml
```

## structure

```
diffuser/
├── src/*.py           # model, schedule, loss, data, sampler, trainer
├── configs/base.yaml  # model + training config
├── scripts/           # sample, explore, viz
├── tests/             # test suite
├── Dockerfile
└── README.md
```

## license

MIT