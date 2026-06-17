"""
Testing and Evaluation script for FGMNet
"""

import argparse
import torch
from torch.utils.data import DataLoader

from models import FGMNet
from utils import get_transforms, MAE, F_measure, S_measure


def test():
    """Main testing and evaluation function"""
    parser = argparse.ArgumentParser(description='Test FGMNet')
    parser.add_argument('--model_path', type=str, default='./checkpoints/fgmnet_best.pth', 
                        help='Path to trained model')
    parser.add_argument('--dataset_path', type=str, default='./dataset/test/', 
                        help='Dataset path')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--img_size', type=int, default=352, help='Input image size')
    
    args = parser.parse_args()
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Model initialization
    model = FGMNet()
    
    # Load checkpoint
    if torch.cuda.is_available():
        checkpoint = torch.load(args.model_path)
    else:
        checkpoint = torch.load(args.model_path, map_location='cpu')
    
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
    # Get test transforms
    transform = get_transforms(split='test', img_size=args.img_size)
    
    # Testing loop
    print(f'Starting evaluation...')
    print(f'Model: {args.model_path}')
    print(f'Dataset: {args.dataset_path}')
    # Evaluation code to be implemented


if __name__ == '__main__':
    test()
