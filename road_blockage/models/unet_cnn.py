"""
U-Net Pure CNN Architecture for Road Blockage Semantic Segmentation
Strictly Transformer-Free: Fully Convolutional Encoder-Decoder with Skip Connections.
Calculates road blockage percentage via direct pixel-area integration and auxiliary regression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNetRoadBlockageCNN(nn.Module):
    """
    Pure CNN U-Net Architecture:
    - Segments road into 3 classes: [0: background, 1: road_clear, 2: road_blocked]
    - Obstacle Multi-Label Classifier Head pooled from bottleneck
    - Blockage regression head + geometrical mask area calculation
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_mask_classes: int = 3,
        num_obstacles: int = 6,
        base_features: int = 32
    ):
        super().__init__()
        self.model_name = "U-Net (Pure CNN Semantic Segmentation)"
        bf = base_features

        # Encoder (Contracting Path)
        self.inc = DoubleConv(in_channels, bf)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bf, bf * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bf * 2, bf * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bf * 4, bf * 8))

        # Bottleneck
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bf * 8, bf * 16))

        # Decoder (Expanding Path with Skip Connections)
        self.up1 = nn.ConvTranspose2d(bf * 16, bf * 8, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(bf * 16, bf * 8)

        self.up2 = nn.ConvTranspose2d(bf * 8, bf * 4, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(bf * 8, bf * 4)

        self.up3 = nn.ConvTranspose2d(bf * 4, bf * 2, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(bf * 4, bf * 2)

        self.up4 = nn.ConvTranspose2d(bf * 2, bf, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(bf * 2, bf)

        # Output Segmentation Mask Head (1x1 Conv)
        self.out_mask = nn.Conv2d(bf, num_mask_classes, kernel_size=1)

        # Bottleneck Auxiliary Multi-Task Heads
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.obstacle_head = nn.Sequential(
            nn.Linear(bf * 16, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_obstacles)
        )
        self.blockage_head = nn.Sequential(
            nn.Linear(bf * 16, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        b = self.bottleneck(x4)

        # Decoder with skip connections
        d1 = self.up1(b)
        d1 = torch.cat([d1, x4], dim=1)
        d1 = self.conv_up1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2, x3], dim=1)
        d2 = self.conv_up2(d2)

        d3 = self.up3(d2)
        d3 = torch.cat([d3, x2], dim=1)
        d3 = self.conv_up3(d3)

        d4 = self.up4(d3)
        d4 = torch.cat([d4, x1], dim=1)
        d4 = self.conv_up4(d4)

        mask_logits = self.out_mask(d4)

        # Auxiliary heads from bottleneck
        b_pooled = torch.flatten(self.global_pool(b), 1)
        obstacle_logits = self.obstacle_head(b_pooled)
        reg_blockage = self.blockage_head(b_pooled) * 100.0

        # Pixel area based blockage calculation from predicted mask
        pred_labels = torch.argmax(mask_logits, dim=1)  # (B, H, W)
        clear_pixels = (pred_labels == 1).sum(dim=(1, 2)).float()
        blocked_pixels = (pred_labels == 2).sum(dim=(1, 2)).float()
        total_road = clear_pixels + blocked_pixels

        # Geometric blockage percentage
        pixel_blockage = torch.where(
            total_road > 0,
            (blocked_pixels / (total_road + 1e-6)) * 100.0,
            torch.zeros_like(total_road)
        ).unsqueeze(1)

        # Ensemble blend of regression head and pixel count
        final_blockage = 0.5 * reg_blockage + 0.5 * pixel_blockage

        return {
            "mask_logits": mask_logits,
            "blockage_pct": final_blockage,
            "pixel_blockage_pct": pixel_blockage,
            "reg_blockage_pct": reg_blockage,
            "obstacle_logits": obstacle_logits
        }
