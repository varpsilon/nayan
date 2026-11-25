# diffuser
personal diffusion model from scratch

a chill walk through denoising space.

## structure
```
diffuser/
├── src/
│   ├── configs/          # model & training configs
│   ├── models/           # neural network blocks
│   ├── utils/            # helpers & transforms
│   └── train.py          # main training loop
├── scripts/
└── tests/
```

## requirements
see `requirements.txt`

## quickstart
```bash
pip install -r requirements.txt
python src/train.py --config configs/base.yaml
```

## philosophy
build slow, smoke slow. every commit is a milestone. lowercase commits only.