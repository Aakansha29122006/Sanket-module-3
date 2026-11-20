"""
EfficientNet Pure CNN Architecture for Road Blockage Assessment
Strictly Transformer-Free: Depthwise Separable Convolutions + MBConv + Squeeze-and-Excitation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class SqueezeExcitation(nn.Module):
    """Pure CNN Squeeze-and-Excitation block."""
    def __init__(self, in_channels: int, reduced_dim: int):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, reduced_dim, kernel_size=1)
        self.fc2 = nn.Conv2d(reduced_dim, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.mean(x, dim=(2, 3), keepdim=True)
        scale = F.silu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Convolution block (pure CNN)."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: int = 4,
        stride: int = 1,
        kernel_size: int = 3
    ):
        super().__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True)
            ])

        # Depthwise Conv
        padding = kernel_size // 2
        layers.extend([
            nn.Conv2d(
                hidden_dim, hidden_dim,
                kernel_size=kernel_size, stride=stride, padding=padding,
                groups=hidden_dim, bias=False
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            SqueezeExcitation(hidden_dim, max(1, in_channels // 4)),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class EfficientNetRoadBlockageCNN(nn.Module):
    """
    Multi-Task EfficientNet Pure CNN:
    - Head 1: Road Blockage Percentage Regression (0 - 100%)
    - Head 2: Obstacle Multi-Label Classification Logits (6 classes)
    - Head 3: Blockage Severity Category Classification (5 classes)
    """
    def __init__(
        self,
        num_obstacles: int = 6,
        num_severities: int = 5,
        in_channels: int = 3
    ):
        super().__init__()
        self.model_name = "EfficientNet-B0 (Pure CNN)"

        # Stem Conv
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )

        # MBConv Stages
        self.stage1 = MBConvBlock(32, 16, expand_ratio=1, stride=1, kernel_size=3)
        self.stage2 = nn.Sequential(
            MBConvBlock(16, 24, expand_ratio=6, stride=2, kernel_size=3),
            MBConvBlock(24, 24, expand_ratio=6, stride=1, kernel_size=3)
        )
        self.stage3 = nn.Sequential(
            MBConvBlock(24, 40, expand_ratio=6, stride=2, kernel_size=5),
            MBConvBlock(40, 40, expand_ratio=6, stride=1, kernel_size=5)
        )
        self.stage4 = nn.Sequential(
            MBConvBlock(40, 80, expand_ratio=6, stride=2, kernel_size=3),
            MBConvBlock(80, 80, expand_ratio=6, stride=1, kernel_size=3)
        )
        self.stage5 = nn.Sequential(
            MBConvBlock(80, 112, expand_ratio=6, stride=1, kernel_size=5),
            MBConvBlock(112, 112, expand_ratio=6, stride=1, kernel_size=5)
        )
        self.stage6 = nn.Sequential(
            MBConvBlock(112, 192, expand_ratio=6, stride=2, kernel_size=5),
            MBConvBlock(192, 192, expand_ratio=6, stride=1, kernel_size=5)
        )
        self.stage7 = MBConvBlock(192, 320, expand_ratio=6, stride=1, kernel_size=3)

        # Head Conv
        self.head_conv = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU(inplace=True)
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.2)

        # Multi-task heads
        # 1. Blockage Percentage Head (scaled to [0, 100])
        self.blockage_head = nn.Sequential(
            nn.Linear(1280, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # outputs in [0, 1], multiplied by 100 in forward
        )

        # 2. Obstacle multi-label classification head
        self.obstacle_head = nn.Sequential(
            nn.Linear(1280, 256),
            nn.SiLU(),
            nn.Linear(256, num_obstacles)
        )

        # 3. Blockage severity categorical head
        self.severity_head = nn.Sequential(
            nn.Linear(1280, 128),
            nn.SiLU(),
            nn.Linear(128, num_severities)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.head_conv(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return x

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.extract_features(x)
        # Blockage percentage: scaled from [0, 1] to [0, 100]%
        blockage_pct = self.blockage_head(features) * 100.0
        obstacle_logits = self.obstacle_head(features)
        severity_logits = self.severity_head(features)

        return {
            "blockage_pct": blockage_pct,
            "obstacle_logits": obstacle_logits,
            "severity_logits": severity_logits,
            "features": features
        }
