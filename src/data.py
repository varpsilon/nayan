"""dataset + dataloader with smart augmentations"""
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import torchvision.transforms as T
from torchvision.transforms.functional import randomize_bbox, crop
from PIL import Image
import os
import random
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Callable
import json


class SmartTransform:
    """smart augmentation pipeline that learns from data."""
    
    def __init__(self, image_size: int = 64, augment: bool = True):
        self.image_size = image_size
        self.augment = augment
        
        self.base_transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),  # [-1, 1] for diffusion
        ])
        
        if augment:
            self.aug_transforms = [
                self._random_hflip,
                self._random_vflip,
                self._random_rotate90,
                self._color_jitter,
                self._gaussian_noise,
                self._cutout,
            ]
    
    def _random_hflip(self, x: torch.Tensor, p: float = 0.5) -> torch.Tensor:
        if random.random() < p:
            return torch.flip(x, dims=[2, 3])
        return x
    
    def _random_vflip(self, x: torch.Tensor, p: float = 0.5) -> torch.Tensor:
        if random.random() < p:
            return torch.flip(x, dims=[1, 3])
        return x
    
    def _random_rotate90(self, x: torch.Tensor, p: float = 0.5) -> torch.Tensor:
        if random.random() < p:
            return torch.rot90(x, k=random.randint(0, 3), dims=[2, 3])
        return x
    
    def _color_jitter(self, x: torch.Tensor, p: float = 0.3) -> torch.Tensor:
        if random.random() < p:
            # simple color jitter for greyscale/colored images
            factor = random.uniform(0.8, 1.2)
            return x * factor
        return x
    
    def _gaussian_noise(self, x: torch.Tensor, p: float = 0.3) -> torch.Tensor:
        if random.random() < p:
            noise = torch.randn_like(x) * 0.05
            return x + noise
        return x
    
    def _cutout(self, x: torch.Tensor, p: float = 0.3) -> torch.Tensor:
        if random.random() < p:
            _, _, h, w = x.shape
            cut_h = random.randint(h // 4, h // 2)
            cut_w = random.randint(w // 4, w // 2)
            y1 = random.randint(0, h - cut_h)
            x1 = random.randint(0, w - cut_w)
            x[:, :, y1:y1+cut_h, x1:x1+cut_w] = 0
        return x
    
    def __call__(self, x: Image.Image) -> torch.Tensor:
        x = self.base_transform(x)
        
        if self.augment:
            for aug in self.aug_transforms:
                x = aug(x)
        
        return x


class DiffusionDataset(Dataset):
    """dataset for diffusion training with class conditioning."""
    
    def __init__(
        self,
        data_path: str,
        image_size: int = 64,
        augment: bool = True,
        max_size: Optional[int] = None,
        extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.webp'),
        cache: bool = False
    ):
        self.data_path = Path(data_path)
        self.image_size = image_size
        self.augment = augment
        self.cache = cache
        
        # find all images
        self.paths = []
        for ext in extensions:
            self.paths.extend(self.data_path.rglob(f'*{ext}'))
        
        if max_size is not None:
            self.paths = self.paths[:max_size]
        
        # load metadata if exists
        self.class_names = self._load_classes()
        self.labels = self._get_labels()
        
        # transforms
        self.transform = SmartTransform(image_size, augment)
        
        # cache
        self._cache = {}
    
    def _load_classes(self) -> List[str]:
        """try to load class names from folder structure or metadata file."""
        classes = []
        
        # check for classes.json
        classes_json = self.data_path / 'classes.json'
        if classes_json.exists():
            with open(classes_json) as f:
                data = json.load(f)
                if isinstance(data, dict):
                    classes = list(data.keys())
                else:
                    classes = data
        
        # check subfolder structure
        else:
            for item in self.data_path.iterdir():
                if item.is_dir():
                    classes.append(item.name)
        
        return classes if classes else ['default']
    
    def _get_labels(self) -> dict:
        """get label for each image path."""
        labels = {}
        
        for path in self.paths:
            if path.parent.name in self.class_names:
                labels[path] = self.class_names.index(path.parent.name)
            else:
                labels[path] = 0
        
        return labels
    
    def __len__(self) -> int:
        return len(self.paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        path = self.paths[idx]
        
        if self.cache and path in self._cache:
            x, label = self._cache[path]
        else:
            x = Image.open(path).convert('RGB')
            x = self.transform(x)
            label = self.labels[path]
            
            if self.cache:
                self._cache[path] = (x, label)
        
        return x, {'label': label, 'path': str(path)}


class UnconditionalDataset(DiffusionDataset):
    """unconditional dataset (no class conditioning)."""
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        x, _ = super().__getitem__(idx)
        return x, {'class_id': 0}  # dummy


def get_dataloader(
    config,
    augment: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2
) -> DataLoader:
    """create dataloader from config."""
    
    dataset = DiffusionDataset(
        data_path=config.data.path,
        image_size=config.model.image_size,
        augment=augment,
        max_size=getattr(config.data, 'max_size', None),
        extensions=getattr(config.data, 'extensions', ('.png', '.jpg', '.jpeg')),
        cache=False
    )
    
    sampler = None
    if augment and len(dataset.class_names) > 1:
        # class-balanced sampling
        class_counts = [0] * len(dataset.class_names)
        for _, label in dataset.labels.items():
            class_counts[label] += 1
        
        weights = [1.0 / class_counts[dataset.labels[p]] for p in dataset.paths]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    
    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        drop_last=True,
        persistent_workers=num_workers > 0
    )