"""
Dataset utilities for FGMNet
Contains a simple dataset loader for COD-like datasets and a dummy dataset for smoke tests.
"""

import os
import glob
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
import numpy as np


class DummyCODDataset(Dataset):
    """A dataset that uses real images if present under dataset_path/images and masks under dataset_path/masks.
    If no files are found, it generates random tensors for smoke testing.
    """
    def __init__(self, dataset_path, split='train', img_size=352, transform=None):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.transform = transform
        self.split = split

        images_dir = os.path.join(dataset_path, 'images')
        masks_dir = os.path.join(dataset_path, 'masks')

        if os.path.isdir(images_dir):
            self.image_paths = sorted(glob.glob(os.path.join(images_dir, '*')))
        else:
            self.image_paths = []

        if os.path.isdir(masks_dir):
            self.mask_paths = sorted(glob.glob(os.path.join(masks_dir, '*')))
        else:
            self.mask_paths = []

        if len(self.image_paths) == 0 or len(self.mask_paths) == 0:
            # No data found — use synthetic data
            self.use_synthetic = True
            self.length = 100
        else:
            self.use_synthetic = False
            # If counts mismatch, use the shorter one
            self.length = min(len(self.image_paths), len(self.mask_paths))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.use_synthetic:
            # Generate random image and mask
            img = (np.random.rand(3, self.img_size, self.img_size) * 255).astype('uint8')
            mask = (np.random.rand(1, self.img_size, self.img_size) > 0.5).astype('float32')
            img = Image.fromarray(np.transpose(img, (1, 2, 0)))
            mask = Image.fromarray((mask[0] * 255).astype('uint8'))
        else:
            img_path = self.image_paths[idx]
            mask_path = self.mask_paths[idx]
            img = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')

        if self.transform is not None:
            img = self.transform(img)
            # For mask we need to convert to tensor manually
            mask = mask.resize((self.img_size, self.img_size))
            mask = np.array(mask).astype('float32') / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0)
        else:
            # Convert to tensors
            img = np.array(img).astype('float32').transpose(2,0,1) / 255.0
            img = torch.from_numpy(img)
            mask = np.array(mask).astype('float32') / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0)

        return img, mask
