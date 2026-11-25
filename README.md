# diffuser
personal diffusion model from scratch

a chill walk through denoising space.

## structure
```
diffuser/
├── src/
│   ├── config.py         # dataclasses + yaml loader
│   ├── schedule.py       # linear / cosine noise schedules  
│   ├── layernorm.py      # channel-first layernorm
│   ├── timemb.py         # learned sinusoidal time embeddings
│   ├── attention.py      # multihead with relative pos bias
│   ├── resnet.py         # resnet blocks with time cond
│   ├── denoiser.py       # hybrid denoiser (variance + confidence heads)
│   ├── loss.py           # vae-style hybrid loss
│   ├── data.py           # smart dataset with dynamic augment
│   ├── sampler.py        # advanced sampler with guidance
│   ├── trainer.py        # ema + amp + wandb logging
│   └── train.py          # main entry point
├── configs/
│   └── base.yaml         # model + training config
├── scripts/
│   ├── sample.py         # inference script
│   ├── analyze_latents.py # latent space interpolation
│   └── plot_metrics.py   # training visualization
├── tests/
│   ├── test_schedule.py
│   └── test_denoiser.py
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```

## key innovations
- **hybrid denoiser**: predicts noise + variance + confidence
- **relative position bias**: better attention modeling
- **learned time embeddings**: adapts to data
- **vae-style loss**: combines mse with variance/confidence training

## quickstart
```bash
pip install -r requirements.txt
python src/train.py --config configs/base.yaml
```

## sampling
```bash
python scripts/sample.py --checkpoint checkpoints/final.pt --output samples.png
```

## philosophy
build slow, smoke slow. every commit is a milestone. lowercase commits only.