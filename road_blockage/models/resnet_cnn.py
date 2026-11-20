"""
ResNet-34 Pure CNN Architecture for Road Blockage Assessment
Strictly Transformer-Free: Deep Residual Convolutional Network.
"""

import torch
import torch.nn as nn
from typing import Dict, List


class BasicResidualBlock(nn.Module):
    """Pure CNN Residual block."""
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNetRoadBlockageCNN(nn.Module):
    """ResNet-34 Pure CNN with multi-task road blockage heads."""
    def __init__(self, num_obstacles: int = 6, in_channels: int = 3):
        super().__init__()
        self.model_name = "ResNet-34 (Pure CNN)"
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.blockage_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.obstacle_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_obstacles)
        )
        self.severity_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 5)
        )

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicResidualBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        feat = torch.flatten(x, 1)

        blockage_pct = self.blockage_head(feat) * 100.0
        obstacle_logits = self.obstacle_head(feat)
        severity_logits = self.severity_head(feat)

        return {
            "blockage_pct": blockage_pct,
            "obstacle_logits": obstacle_logits,
            "severity_logits": severity_logits,
            "features": feat
        }
