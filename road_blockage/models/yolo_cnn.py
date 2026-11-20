"""
YOLO Pure CNN Architectures for Road Blockage Detection
Includes YOLOv8-CNN (anchor-free C2f CSPDarknet) and YOLOv5-CNN (anchor-based C3 CSPDarknet).
Strictly Transformer-Free: 100% Convolutional Backbone, SPPF, and PANet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class ConvBNSiLU(nn.Module):
    """Standard YOLO Conv2D + BatchNorm2d + SiLU."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class YOLOBottleneck(nn.Module):
    """Standard YOLO pure CNN Bottleneck."""
    def __init__(self, c1: int, c2: int, shortcut: bool = True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = ConvBNSiLU(c1, c_, kernel_size=1, stride=1, padding=0)
        self.cv2 = ConvBNSiLU(c_, c2, kernel_size=3, stride=1, padding=1)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2fBlock(nn.Module):
    """YOLOv8 C2f module (pure CNN faster CSP with multi-branch gradient flow)."""
    def __init__(self, c1: int, c2: int, n: int = 2, shortcut: bool = True):
        super().__init__()
        self.c = int(c2 * 0.5)
        self.cv1 = ConvBNSiLU(c1, 2 * self.c, kernel_size=1, stride=1, padding=0)
        self.cv2 = ConvBNSiLU((2 + n) * self.c, c2, kernel_size=1, stride=1, padding=0)
        self.m = nn.ModuleList([YOLOBottleneck(self.c, self.c, shortcut) for _ in range(n)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3Block(nn.Module):
    """YOLOv5 C3 module (pure CNN CSP bottleneck)."""
    def __init__(self, c1: int, c2: int, n: int = 2, shortcut: bool = True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = ConvBNSiLU(c1, c_, kernel_size=1, stride=1, padding=0)
        self.cv2 = ConvBNSiLU(c1, c_, kernel_size=1, stride=1, padding=0)
        self.cv3 = ConvBNSiLU(2 * c_, c2, kernel_size=1, stride=1, padding=0)
        self.m = nn.Sequential(*(YOLOBottleneck(c_, c_, shortcut) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling Fast (pure CNN multi-scale spatial receptive field)."""
    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = ConvBNSiLU(c1, c_, kernel_size=1, stride=1, padding=0)
        self.cv2 = ConvBNSiLU(c_ * 4, c2, kernel_size=1, stride=1, padding=0)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class YOLORoadBlockageCNN(nn.Module):
    """
    YOLOv8-CNN Architecture (Pure CNN):
    - Backbone: CSP-DarkNet with C2f modules and SPPF
    - Neck: PANet Feature Aggregation
    - Head: Decoupled Multi-Task Road Blockage & Obstacle Predictor
    """
    def __init__(
        self,
        num_obstacles: int = 6,
        in_channels: int = 3,
        version: str = "v8"
    ):
        super().__init__()
        self.version = version
        self.model_name = f"YOLO{version}-CNN (Pure CNN)"

        # P1 & P2
        self.conv1 = ConvBNSiLU(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = ConvBNSiLU(32, 64, kernel_size=3, stride=2, padding=1)

        # Choose C2f (v8) or C3 (v5)
        BlockClass = C2fBlock if version == "v8" else C3Block

        # P3 stage
        self.stage3 = BlockClass(64, 64, n=1)
        self.down3 = ConvBNSiLU(64, 128, kernel_size=3, stride=2, padding=1)

        # P4 stage
        self.stage4 = BlockClass(128, 128, n=2)
        self.down4 = ConvBNSiLU(128, 256, kernel_size=3, stride=2, padding=1)

        # P5 stage + SPPF
        self.stage5 = BlockClass(256, 256, n=1)
        self.sppf = SPPF(256, 256, k=5)

        # Global Pooling & Multi-Task Decoupled Heads
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Blockage Percentage Head (0-100%)
        self.blockage_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Obstacle Multi-Label Classification Head
        self.obstacle_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, num_obstacles)
        )

        # Severity Classification Head
        self.severity_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, 5)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.stage3(x)
        x = self.down3(x)
        x = self.stage4(x)
        x = self.down4(x)
        x = self.stage5(x)
        x = self.sppf(x)
        feat = torch.flatten(self.global_pool(x), 1)
        return feat

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.extract_features(x)
        blockage_pct = self.blockage_head(feat) * 100.0
        obstacle_logits = self.obstacle_head(feat)
        severity_logits = self.severity_head(feat)

        return {
            "blockage_pct": blockage_pct,
            "obstacle_logits": obstacle_logits,
            "severity_logits": severity_logits,
            "features": feat
        }


def build_yolov5_cnn(num_obstacles: int = 6) -> YOLORoadBlockageCNN:
    """Build YOLOv5 CSP-DarkNet pure CNN."""
    return YOLORoadBlockageCNN(num_obstacles=num_obstacles, version="v5")


def build_yolov8_cnn(num_obstacles: int = 6) -> YOLORoadBlockageCNN:
    """Build YOLOv8 CSP-DarkNet pure CNN."""
    return YOLORoadBlockageCNN(num_obstacles=num_obstacles, version="v8")
