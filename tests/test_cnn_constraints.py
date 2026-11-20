"""
Pure CNN Constraint Verification Tests
Ensures zero transformer layers exist across all models (Transformers Strictly Forbidden).
"""

import unittest
import torch
import torch.nn as nn

from road_blockage.models.efficientnet_cnn import EfficientNetRoadBlockageCNN
from road_blockage.models.unet_cnn import UNetRoadBlockageCNN
from road_blockage.models.yolo_cnn import build_yolov5_cnn, build_yolov8_cnn
from road_blockage.models.resnet_cnn import ResNetRoadBlockageCNN


FORBIDDEN_KEYWORDS = [
    "attention",
    "transformer",
    "multihead",
    "vit",
    "swin",
    "selfattention",
    "crossattention"
]


class TestCNNArchitectureConstraints(unittest.TestCase):
    """Verify that models are 100% Convolutional Neural Networks without attention/transformers."""

    def _verify_no_transformers(self, model: nn.Module, model_name: str):
        for name, module in model.named_modules():
            mod_type = type(module).__name__.lower()
            mod_name = name.lower()

            for kw in FORBIDDEN_KEYWORDS:
                self.assertNotIn(
                    kw, mod_type,
                    f"Forbidden keyword '{kw}' found in module type '{type(module).__name__}' of model '{model_name}'"
                )
                self.assertNotIn(
                    kw, mod_name,
                    f"Forbidden keyword '{kw}' found in layer name '{name}' of model '{model_name}'"
                )

        # Confirm presence of pure Conv2d layers
        conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        self.assertGreater(
            len(conv_layers), 5,
            f"Model '{model_name}' must be built with Convolutional layers (found {len(conv_layers)})"
        )

    def test_efficientnet_is_pure_cnn(self):
        model = EfficientNetRoadBlockageCNN()
        self._verify_no_transformers(model, "EfficientNet-B0")

    def test_unet_is_pure_cnn(self):
        model = UNetRoadBlockageCNN()
        self._verify_no_transformers(model, "U-Net")

    def test_yolov8_is_pure_cnn(self):
        model = build_yolov8_cnn()
        self._verify_no_transformers(model, "YOLOv8-CNN")

    def test_yolov5_is_pure_cnn(self):
        model = build_yolov5_cnn()
        self._verify_no_transformers(model, "YOLOv5-CNN")

    def test_resnet_is_pure_cnn(self):
        model = ResNetRoadBlockageCNN()
        self._verify_no_transformers(model, "ResNet-34")


if __name__ == "__main__":
    unittest.main()
