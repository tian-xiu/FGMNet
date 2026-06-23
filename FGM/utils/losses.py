"""
Loss functions for FGMNet.
Implements the combined loss:
    L_total = L_BCE + L_IoU + L_SSIM + L_Dice
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def bce_loss(pred, target):
    """Binary Cross-Entropy loss."""
    return F.binary_cross_entropy_with_logits(pred, target)


def dice_loss(pred, target, smooth=1.0):
    """
    Dice loss: 1 - Dice coefficient.
    pred: logits [B, 1, H, W]
    target: binary mask [B, 1, H, W]
    """
    pred_prob = torch.sigmoid(pred)
    pred_prob = pred_prob.contiguous().view(pred_prob.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)

    intersection = (pred_prob * target).sum(dim=1)
    union = pred_prob.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def iou_loss(pred, target, smooth=1.0):
    """
    IoU loss: 1 - IoU (Jaccard index).
    pred: logits [B, 1, H, W]
    target: binary mask [B, 1, H, W]
    """
    pred_prob = torch.sigmoid(pred)
    pred_prob = pred_prob.contiguous().view(pred_prob.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)

    intersection = (pred_prob * target).sum(dim=1)
    union = pred_prob.sum(dim=1) + target.sum(dim=1) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return 1.0 - iou.mean()


class SSIMLoss(nn.Module):
    """
    SSIM loss for image segmentation.
    Uses a simplified 1D Gaussian kernel to compute local statistics.
    L_SSIM = 1 - SSIM(pred, target)
    """
    def __init__(self, window_size=11, sigma=1.5, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_gaussian_window(window_size, sigma)

    def _create_gaussian_window(self, window_size, sigma):
        """Create a 2D Gaussian window."""
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        # 2D window
        window = gauss[:, None] * gauss[None, :]
        window = window.expand(1, 1, window_size, window_size).contiguous()
        return window

    def forward(self, pred, target):
        """
        pred: logits [B, 1, H, W]
        target: binary mask [B, 1, H, W]
        """
        pred_prob = torch.sigmoid(pred)

        # Ensure window is on the same device
        if self.window.device != pred_prob.device:
            self.window = self.window.to(pred_prob.device)

        # Compute local means
        mu_pred = F.conv2d(pred_prob, self.window, padding=self.window_size // 2, groups=1)
        mu_target = F.conv2d(target, self.window, padding=self.window_size // 2, groups=1)

        # Compute local variances and covariance
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target

        sigma_pred_sq = F.conv2d(pred_prob ** 2, self.window, padding=self.window_size // 2, groups=1) - mu_pred_sq
        sigma_target_sq = F.conv2d(target ** 2, self.window, padding=self.window_size // 2, groups=1) - mu_target_sq
        sigma_pred_target = F.conv2d(pred_prob * target, self.window, padding=self.window_size // 2, groups=1) - mu_pred_target

        # Clamp to avoid negative values from numerical issues
        sigma_pred_sq = torch.clamp(sigma_pred_sq, min=0.0)
        sigma_target_sq = torch.clamp(sigma_target_sq, min=0.0)

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2.0 * mu_pred_target + C1) * (2.0 * sigma_pred_target + C2)) / \
                    ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))

        if self.size_average:
            return 1.0 - ssim_map.mean()
        else:
            return 1.0 - ssim_map.mean(dim=(1, 2, 3))


class CombinedLoss(nn.Module):
    """
    Combined loss function:
        L_total = L_BCE + L_IoU + L_SSIM + L_Dice
    """
    def __init__(self, weights=None):
        super(CombinedLoss, self).__init__()
        if weights is None:
            weights = {'bce': 1.0, 'iou': 1.0, 'ssim': 1.0, 'dice': 1.0}
        self.weights = weights
        self.ssim_loss = SSIMLoss()

    def forward(self, pred, target, return_components=False):
        """
        pred: logits [B, 1, H, W]
        target: binary mask [B, 1, H, W] in [0, 1]
        If return_components=True, returns (total_loss, dict_of_components)
        """
        loss_bce = bce_loss(pred, target)
        loss_iou = iou_loss(pred, target)
        loss_ssim = self.ssim_loss(pred, target)
        loss_dice = dice_loss(pred, target)

        components = {
            'bce': loss_bce,
            'iou': loss_iou,
            'ssim': loss_ssim,
            'dice': loss_dice,
        }

        total = (self.weights['bce'] * loss_bce +
                 self.weights['iou'] * loss_iou +
                 self.weights['ssim'] * loss_ssim +
                 self.weights['dice'] * loss_dice)

        if return_components:
            return total, components
        return total


# Keep alias for backward compatibility
DiceLoss = dice_loss
