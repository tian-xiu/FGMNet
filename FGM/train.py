"""
Training script for FGMNet (minimal runnable)
Provides a small training loop that works with the DummyCODDataset when real data is not available.
"""

import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import FGMNet
from utils import get_transforms
from utils.losses import CombinedLoss
from dataset import DummyCODDataset


def train_cli(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # dataset and dataloader
    transform = get_transforms(split='train', img_size=args.img_size)
    dataset = DummyCODDataset(args.dataset_path, split='train', img_size=args.img_size, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # model
    model = FGMNet()
    model = model.to(device)

    # optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = CombinedLoss()

    # training loop
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_components = {'bce': 0.0, 'iou': 0.0, 'ssim': 0.0, 'dice': 0.0}
        for i, (imgs, masks) in enumerate(dataloader):
            imgs = imgs.to(device).float()
            masks = masks.to(device).float()

            preds = model(imgs)
            loss, components = criterion(preds, masks, return_components=True)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            for k in epoch_components:
                epoch_components[k] += components[k].item()

            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{args.epochs}] Step [{i+1}/{len(dataloader)}] '
                      f'Total: {loss.item():.4f} | BCE: {components["bce"].item():.4f} '
                      f'IoU: {components["iou"].item():.4f} SSIM: {components["ssim"].item():.4f} '
                      f'Dice: {components["dice"].item():.4f}')

        n = len(dataloader)
        avg_loss = epoch_loss / n
        avg_components = {k: v / n for k, v in epoch_components.items()}
        print(f'Epoch [{epoch+1}] Avg Total: {avg_loss:.4f} | '
              f'BCE: {avg_components["bce"]:.4f} IoU: {avg_components["iou"]:.4f} '
              f'SSIM: {avg_components["ssim"]:.4f} Dice: {avg_components["dice"]:.4f}')

        # save checkpoint
        ckpt_path = os.path.join(args.checkpoint_dir, f'fgmnet_epoch{epoch+1}.pth')
        torch.save(model.state_dict(), ckpt_path)
        print(f'Saved checkpoint: {ckpt_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train FGMNet (minimal)')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--dataset_path', type=str, default='./dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--img_size', type=int, default=352)
    args = parser.parse_args()
    train_cli(args)
