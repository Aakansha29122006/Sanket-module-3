"""
Model Forward Pass and Dimension Verification Tests
Tests tensor shapes, output keys, and value bounds across all candidate architectures.
"""

import unittest
import torch

from road_blockage.models.efficientnet_cnn import EfficientNetRoadBlockageCNN
from road_blockage.models.unet_cnn import UNetRoadBlockageCNN
from road_blockage.models.yolo_cnn import build_yolov5_cnn, build_yolov8_cnn
from road_blockage.models.resnet_cnn import ResNetRoadBlockageCNN


class TestModelForwardPass(unittest.TestCase):
    """Test forward passes for batch_size=2 on 224x224 RGB inputs."""

    def setUp(self):
        self.batch_size = 2
        self.dummy_input = torch.randn(self.batch_size, 3, 224, 224)

    def _assert_valid_multi_task_output(self, output: dict, model_name: str):
        self.assertIn("blockage_pct", output, f"{model_name} missing blockage_pct")
        self.assertIn("obstacle_logits", output, f"{model_name} missing obstacle_logits")

        pct = output["blockage_pct"]
        obs = output["obstacle_logits"]

        self.assertEqual(pct.shape, (self.batch_size, 1), f"{model_name} invalid blockage shape")
        self.assertEqual(obs.shape, (self.batch_size, 6), f"{model_name} invalid obstacle shape")

        # Range bounds for percentage
        self.assertTrue(torch.all(pct >= 0.0), f"{model_name} blockage pct has negative values")
        self.assertTrue(torch.all(pct <= 100.0), f"{model_name} blockage pct exceeds 100")

    def test_efficientnet_forward(self):
        model = EfficientNetRoadBlockageCNN()
        model.eval()
        with torch.no_grad():
            out = model(self.dummy_input)
        self._assert_valid_multi_task_output(out, "EfficientNet")
        self.assertIn("severity_logits", out)
        self.assertEqual(out["severity_logits"].shape, (self.batch_size, 5))

    def test_unet_forward(self):
        model = UNetRoadBlockageCNN(base_features=16)
        model.eval()
        with torch.no_grad():
            out = model(self.dummy_input)
        self._assert_valid_multi_task_output(out, "U-Net")
        self.assertIn("mask_logits", out)
        self.assertEqual(out["mask_logits"].shape, (self.batch_size, 3, 224, 224))

    def test_yolov8_forward(self):
        model = build_yolov8_cnn()
        model.eval()
        with torch.no_grad():
            out = model(self.dummy_input)
        self._assert_valid_multi_task_output(out, "YOLOv8")

    def test_yolov5_forward(self):
        model = build_yolov5_cnn()
        model.eval()
        with torch.no_grad():
            out = model(self.dummy_input)
        self._assert_valid_multi_task_output(out, "YOLOv5")

    def test_resnet_forward(self):
        model = ResNetRoadBlockageCNN()
        model.eval()
        with torch.no_grad():
            out = model(self.dummy_input)
        self._assert_valid_multi_task_output(out, "ResNet")


if __name__ == "__main__":
    unittest.main()
